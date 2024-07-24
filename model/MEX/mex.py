import math
import torch
from torch import nn, randn
import torch.nn.functional as F
from utils.utils import distributed_rank
from einops import rearrange
from transformers import AutoImageProcessor, Swinv2Model
from model.common.func import MLP,CausalSelfAttention
from model.common.position_embedding import build
import clip
from clip.model import CLIP, convert_weights
from clip.model import AttentionPool2d
import os

def tokenize(text):
    token = clip.tokenize(text)
    return token

class MyCLIP(CLIP):
    def __init__(self, *args):
        super(MyCLIP, self).__init__(*args)

    def encode_text_2(self, text, truncation=10):
        
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        hidden = x[torch.arange(x.shape[0]), :truncation] @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return hidden, x

def load_clip(model_path, device=None,input_resolution=None):
    state_dict = torch.jit.load(model_path).state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    if input_resolution is not None:
        if input_resolution != image_resolution:
            del state_dict['visual.attnpool.positional_embedding']
        image_resolution = input_resolution

    model = MyCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    ).to(device)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)

    return model

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, device=None,is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False,device=device)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False,device=device)
        self.bn1 = nn.BatchNorm2d(c_out,device=device)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False,device=device)
        self.bn2 = nn.BatchNorm2d(c_out,device=device)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False,device=device),
                nn.BatchNorm2d(c_out,device=device)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False,device=device),
                nn.BatchNorm2d(c_out,device=device)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times,device=None, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample,device=device), ]
        else:
            blocks += [BasicBlock(c_out, c_out,device=device), ]
    return nn.Sequential(*blocks)

class FFN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mlp = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.mlp(x)
        x = x + self.drop(y)
        x = self.norm(x)
        return x
    
class AddNorm(nn.Module):
    def __init__(self, d_model,device, dropout=0.01):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model,device=device)

    def forward(self, Y, X):
        return self.layer_norm(self.dropout(Y) + X)
    
class TextSelfAttentionBlock(nn.Module):
    def __init__(self, d_model,seq_length,num_heads=4, device=None, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.ln_1 = nn.LayerNorm(self.d_model, bias=True,device=device)
        self.attn = CausalSelfAttention(d_model,seq_length, num_heads, dropout=dropout,device=device)
        self.ln_2 = nn.LayerNorm(self.d_model, bias=True,device=device)
        self.mlp = MLP(d_model, dropout=dropout,device=device)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    
class FusionAttention(nn.Module):
    def __init__(self, d_model, n_heads, device=None,dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "The hidden size is not a multiple of the number of attention heads"

        self.d_model = d_model
        self.n_heads = n_heads
        
        self.attention_head_size = int(d_model / n_heads)
        self.all_head_size = n_heads * self.attention_head_size

        # global_local block (glb)
        self.glb_q = nn.Linear(d_model, self.all_head_size,device=device)
        self.glb_k = nn.Linear(d_model, self.all_head_size,device=device)
        self.glb_v = nn.Linear(d_model, self.all_head_size,device=device)

        # prompt_local block (plb)
        self.plb_q = nn.Linear(d_model, self.all_head_size,device=device)
        self.plb_k = nn.Linear(d_model, self.all_head_size,device=device)
        self.plb_v = nn.Linear(d_model, self.all_head_size,device=device)

        self.glb_dropout = nn.Dropout(p=dropout)
        self.plb_dropout = nn.Dropout(p=dropout)


        self.dense = nn.Linear(d_model, d_model,device=device)
        self.mlp=MLP(d_model,dropout,device=device)
        self.mlp_linear =nn.Linear(d_model, d_model,device=device)

        self.global_embedding = build(d_model, dropout=dropout)
        self.local_embedding_1 = build(d_model, dropout=dropout)
        self.local_embedding_2 = build(d_model, dropout=dropout)


        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,global_feat,local_feat,text_feat,batch_first=False):
        '''
        global_feat: [Batch_size x Seq_length x Hidden_size]
        local_feat: [Batch_size x Seq_length x Hidden_size]
        text_feat: [Batch_size x Seq_length x Hidden_size] 
        '''
        if batch_first == False:
            global_feat = global_feat.permute(1, 0, 2)  # [Batch_size x Seq_length x Hidden_size]
            local_feat = local_feat.permute(1, 0, 2)
            text_feat = text_feat.permute(1, 0, 2)

        duplicate_local = local_feat.clone()
        duplicate_local= self.local_embedding_1(duplicate_local)
        local_feat = self.local_embedding_2(local_feat)
        global_feat = self.global_embedding(global_feat)

        # global_local block (glb)
        glb_mixed_q_layer = self.glb_q(global_feat)  # [Batch_size x Seq_length x Hidden_size]
        glb_mixed_k_layer = self.glb_k(local_feat)  # [Batch_size x Seq_length x Hidden_size]
        glb_mixed_v_layer = self.glb_v(local_feat)  # [Batch_size x Seq_length x Hidden_size]

        # prompt_local block (plb)
        plb_mixed_q_layer = self.plb_q(duplicate_local)
        plb_mixed_k_layer = self.plb_k(text_feat)
        plb_mixed_v_layer = self.plb_v(text_feat)


        # global_local block (glb)
        glb_q_layer = self.transpose_for_scores(
            glb_mixed_q_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        glb_k_layer = self.transpose_for_scores(glb_mixed_k_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        glb_v_layer = self.transpose_for_scores(
            glb_mixed_v_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        # prompt_local block (plb)
        plb_q_layer = self.transpose_for_scores(plb_mixed_q_layer)
        plb_k_layer = self.transpose_for_scores(plb_mixed_k_layer)
        plb_v_layer = self.transpose_for_scores(plb_mixed_v_layer)

        # global_local block (glb)
        glb_attention_scores = torch.matmul(glb_q_layer, glb_k_layer.transpose(-1,-2)) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        glb_attention_scores = glb_attention_scores / math.sqrt(self.attention_head_size) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        glb_attention_probs = nn.Softmax(dim=-1)(glb_attention_scores) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        glb_attention_probs = self.glb_dropout(glb_attention_probs) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        bridge_value=glb_attention_probs

        # prompt_local block (plb)
        plb_attention_scores = torch.matmul(plb_q_layer, plb_k_layer.transpose(-1,-2))
        plb_attention_scores = plb_attention_scores / math.sqrt(self.attention_head_size)
        plb_attention_probs = nn.Softmax(dim=-1)(plb_attention_scores)
        plb_attention_probs = self.plb_dropout(plb_attention_probs)

        plb_context_enhanced=torch.matmul(bridge_value,plb_attention_probs)

        # add value
        glb_context_layer = torch.matmul(glb_attention_probs, glb_v_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        plb_context_layer = torch.matmul(plb_context_enhanced, plb_v_layer)


        # glb context layer
        glb_context_layer = glb_context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        glb_new_context_layer_shape = glb_context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        glb_context_layer = glb_context_layer.view(*glb_new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]


        # plb context layer
        plb_context_layer = plb_context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        plb_new_context_layer_shape = plb_context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        plb_context_layer = plb_context_layer.view(*plb_new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        last_context_layer = glb_context_layer + plb_context_layer
        output = self.dense(last_context_layer)
        output = self.mlp(output)
        output = self.mlp_linear(output)

        return output



class Id(AttentionPool2d):
    def __init__(self, x=0,y=0,z=0):
        super(Id, self).__init__(x,y,z)
    def forward(self, x):
        x = x.cuda()
        return x

class MEX(nn.Module):
    def __init__(self,config: dict):
        super().__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = 4
        self.dropout = 0.1
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.swinv2_model =Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        self.config=config

        self.clip = load_clip(
            os.path.join(config["CLIP_CHECKPOINT_DIR"], "RN50.pt"),
            device=self.device,
            input_resolution=224,
        )
        self.clip = self.clip.float()

        self.clip.visual.attnpool = Id().to(self.device)
        self._freeze_text_encoder()
        self._freeze_clip()
           #reprocess image

        self.cnn_image_local=nn.Sequential(*[make_layers(768, 512, 2,device=self.device, is_downsample=False),
                                        make_layers(512, 256, 2,device=self.device, is_downsample=True)])
        self.cnn_image_global=nn.Sequential(*[make_layers(768, 512, 2, device=self.device,is_downsample=False),
                                        make_layers(512, 256, 2,device=self.device, is_downsample=True)])
        #reprocess text
        self.feature_dim=256
        self.img_dim = 256
        self.text_dim = 1024
        self.img_fc = self.get_img_fc(use_ln=False).to(self.device)
        self.text_fc = self.get_text_fc(use_ln=True).to(self.device)
        self.seq_length=config['TRUNCATION']
       
        
        local_reso = 4 * 4
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso))
        global_reso = 4 * 4
        global_scale = global_reso ** -0.5
        self.pos_emb_global = nn.Parameter(global_scale * randn(global_reso))

        self.fusion_fc = nn.Linear(self.text_dim, self.img_dim,device=self.device)
        self.fusion_ffn = FFN(self.img_dim, 0.1)

        self.global_attn_ = nn.MultiheadAttention(self.feature_dim, self.num_heads, dropout=self.dropout,device=self.device)
        self.global_add_norm_ = AddNorm(self.feature_dim, dropout=self.dropout,device=self.device)
        self.local_attn_ = nn.MultiheadAttention(self.feature_dim, self.num_heads, dropout=self.dropout,device=self.device)
        self.local_add_norm_ = AddNorm(self.feature_dim, dropout=self.dropout,device=self.device)
        self.text_attn_ = TextSelfAttentionBlock(self.feature_dim,self.seq_length, self.num_heads,device=self.device, dropout=self.dropout)

        self.weird_attn = FusionAttention(self.feature_dim, self.num_heads,device=self.device, dropout=self.dropout)

        transformer_width =768
        #stolen from clip
        self.encoder_attention_mask=self.build_attention_mask()
        self.text_projection =nn.Parameter( torch.empty(transformer_width, self.feature_dim))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.seq_length, self.seq_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _freeze_text_encoder(self):
        """
        These parameters are not frozen:
        - list(self.clip.token_embedding.parameters())
        - [self.clip.positional_embedding]
        """
        for p in list(self.swinv2_model.parameters()):
            p.requires_grad = False
     
        # self.bert_model.eval()
        self.swinv2_model.eval()

    def encode_images(self,local_img,global_img):
        b, t = global_img.size()[:2]
        local_img = rearrange(local_img, 'b t c h w -> (b t) c h w')
        local_feat =  self.process_image(local_img);  # [bt,c,7,7]
      
        global_img = rearrange(global_img, 'B T C H W -> (B T) C H W')
        global_feat =  self.process_image(global_img); 
      
        # rearrange
        local_feat = rearrange(local_feat, 'bt (h w) c -> bt c h w',h=8)
        local_feat = self.cnn_image_local(local_feat)

        global_feat = rearrange(global_feat, 'BT (H W) C-> BT C H W',H=8)
        global_feat = self.cnn_image_global(global_feat)

        local_feat = rearrange(local_feat, 'bt c h w -> bt c (h w)')
        global_feat = rearrange(global_feat, 'bt C H W -> bt C (H W)')

        local_feat = local_feat + self.pos_emb_local.to(self.device)
        global_feat = global_feat + self.pos_emb_global.to(self.device)

        local_feat = rearrange(local_feat, 'bt c hw -> bt hw c')
        global_feat = rearrange(global_feat, 'bt C HW -> bt HW C')
        return local_feat,global_feat

    def self_attentions(self, global_feat,local_feat,text_feat):
        # self attention
        y1,_= self.global_attn_(global_feat,global_feat,global_feat)
        y1 = self.global_add_norm_(y1,global_feat)

        y2,_= self.local_attn_(local_feat,local_feat,local_feat)
        y2 = self.local_add_norm_(y2,local_feat)

        y3= self.text_attn_(text_feat)
        return y1,y2,y3
    def _freeze_clip(self):
        for p in list(self.clip.transformer.parameters()) + \
                 list(self.clip.ln_final.parameters()) + \
                 [self.clip.text_projection, ]:
            p.requires_grad = False
    
    def textual_encoding_clip(self, tokens):
        x_hidden, x = self.clip.encode_text_2(tokens, self.config["TRUNCATION"])
        x = self.text_fc(x)
        if self.training:
            return x_hidden, x
        else:
            return x_hidden, F.normalize(x, p=2, dim=-1)

    def forward(self, x):
        output = dict()
        imgs= x['local_images']
        texts = x['sentences']
        b,n = imgs.size()[:2]
        # textual_hidden,text_feat= self.encode_text_2(texts)
        exp =  tokenize(texts).cuda()
        text_feat,textual_hidden = self.textual_encoding_clip(exp)
        local_feat,global_feat = self.encode_images(x['local_images'],x['global_image'])

        #text-guided
        assert len(text_feat.size()) == 3
        # get textual embeddings
        text_feat = text_feat.unsqueeze(1)  # [b,l,c]->[b,1,l,c]
        text_feat = text_feat.repeat([1, n, 1, 1])
        text_feat = rearrange(text_feat, 'b t l c -> (b t) l c')
        text_feat = self.fusion_fc(text_feat)

        global_feat,local_feat,text_feat = self.self_attentions(global_feat,local_feat,text_feat)

        visual_feat = self.weird_attn(global_feat,local_feat,text_feat,batch_first=True) 
        visual_feat = visual_feat * local_feat
        vis_feat = rearrange(visual_feat, "bt l c -> bt c l")
        vis_feat = self.st_pooling(vis_feat, bs=b)
        if not self.training:
            vis_feat = F.normalize(vis_feat, p=2, dim=-1)

        # visual_feat = rearrange(visual_feat,'(b t) c -> t b c',b=b)
        k1 = rearrange(vis_feat,"(b n) c -> n b c",b=b)
        k2 = rearrange(textual_hidden,"(b n) c -> n b c",b=b)
        scores = torch.mean(F.cosine_similarity(k1, k2, dim=-1),0)
            

        output['scores'] = scores
        output['vis_feat'] = vis_feat
        output['text_feat'] = text_feat
        return output

    def st_pooling(self, feat, bs):
        # spatial pooling
        feat = F.adaptive_avg_pool1d(feat, 1)  # [bt,c,l]->[bt,c]
       # this file change
        feat=rearrange(feat, '(b t) c l -> b (c l) t',b=bs)
        feat = F.adaptive_max_pool1d(feat, 1)  # [b,c]
        feat = rearrange(feat, 'b c t -> b (t c)')

        # projection
        feat = self.img_fc(feat)
        return feat


    def get_img_fc(self, use_ln=True):
            return nn.Linear(self.img_dim, self.feature_dim)

    def get_text_fc(self, use_ln=True):
        if use_ln:
            return nn.Sequential(
                nn.Linear(self.text_dim, self.text_dim),
                nn.ReLU(),
                nn.Linear(self.text_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim, eps=1e-12),
            )
    
    def process_image(self,image):

        temp=self.image_encoder(image)
        # padded_temp = F.pad(temp, (0, 256, 0, 0, 0, 0), mode='constant', value=0)
        return temp
    
    def image_encoder(self, image): # [1,49,768]
        inputs = self.image_processor(image, return_tensors="pt",do_rescale=False).to(self.device)    
        outputs = self.swinv2_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states 
    

def build_MEX(config: dict):

    model = MEX(config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model