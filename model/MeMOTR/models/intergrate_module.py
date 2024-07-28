import torch
from utils.box_ops import box_cxcywh_to_xyxy

import torchvision.transforms.functional as F
import torchvision.transforms as T

class SquarePad:
    """Reference:
    https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
    """
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


config=[(224, 224), (448, 448), (672, 672)]
random_crop_ratio = [0.8, 1.0]

def get_transform(mode, idx):
    if mode == 'train':
        return T.Compose([
            SquarePad(),
            T.RandomResizedCrop(
                config[idx],
                ratio=random_crop_ratio
            ),
            T.ToTensor(),
            # T.Normalize(opt["norm_mean"], opt["norm_std"]),
        ])
    elif mode == 'test':
        return T.Compose([
            SquarePad(),
            T.Resize(config[idx]),
            T.ToTensor(),
            # T.Normalize(opt["norm_mean"], opt['norm_std']),
        ])
    
transform = {idx: get_transform("test", idx) for idx in (0, 1, 2)}

def preprocess_image(model, images,global_images, caption):
    processed_local_image = torch.stack(
                [transform[0](
                    image
                ) for image in images],
                dim=0,
            )
    processed_global_image = torch.stack(
        [transform[2](img) for img in global_images],
        dim=0,
    )
    caption= caption
    captions=[caption for _ in range(len(images))]
    # forward
    inputs = dict(
        local_images=processed_local_image.unsqueeze(1).to(model.device),
        global_image=processed_global_image.unsqueeze(1).to(model.device),
        sentences=captions,
    )
    model_outputs =model(inputs)
    scores = model_outputs['scores']
    # print("scores: ",scores)

    return scores


def cut_image_from_bbox( image, boxes):
    left = torch.clamp(boxes[:, 0], 0).tolist()
    top = torch.clamp(boxes[:, 1], 0).tolist()
    right = torch.clamp(boxes[:, 2], 0).tolist()
    bottom = torch.clamp(boxes[:, 3], 0).tolist()
    cropped_image =[]
    
    for i in range(len(left)):
        img=image.copy().crop((left[i], top[i], right[i], bottom[i]))
        cropped_image.append(img)

    return cropped_image

class INTERGRATE_MODULE():
    def __init__(self, model, device,config=None):
        self.model = model
        self.device = device
        self.is_IKUN=False
        if config["MODULE_NAME"] == "IKUN":
            self.is_IKUN=True
    def predict(self, caption, new_bbox,width,height,temp_img,old_images=[],old_global_images=[]):
        boxes = box_cxcywh_to_xyxy(new_bbox)
        boxes = (boxes * torch.as_tensor([width,height, width, height], dtype=torch.float).to(boxes.device))
        

        crop_image=cut_image_from_bbox(temp_img,boxes)
        crop_image2=crop_image+ old_images
        global_images= [temp_img for _ in range(len(crop_image)) ] + old_global_images
        # print("local-image new: {}, old: {}".format(len(crop_image),len(old_images)))

        masks = []
        if len(crop_image2)>0:
            if self.is_IKUN:
                probs = self.model(crop_image2,global_images, caption)
            else:
                probs = preprocess_image(self.model, crop_image2,global_images, caption)
          
            masks = probs
        return masks,crop_image