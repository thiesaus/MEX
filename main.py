import yaml
# from model.filter_module import FilterModule
from model.utils.utils import update_config
from train_engine import train
from submit_engine import submit
from inference_engine import inference
from inference_submit_engine import submit_inference
import argparse 


def yaml_to_dict(path: str):
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


def parse_option():
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    # Running mode, Training? Evaluation? or ?
    parser.add_argument("--mode", type=str, help="Running mode. train/submit/inf/submitinf")

    # Yaml Config Path:
    parser.add_argument("--config_module", type=str, help="Module config file path.",default="./configs/mex.yaml")
    parser.add_argument("--data_root", type=str, help="Space config file path.",default="./data_dir/Refer_Kitti")

     # basic parameters
    parser.add_argument('--img_hw', nargs='+', type=tuple,
                                default=[(224, 224), (448, 448), (672, 672)])
    parser.add_argument('--norm_mean', type=list, default=[0.48145466, 0.4578275, 0.40821073])
    parser.add_argument('--norm_std', type=list, default=[0.26862954, 0.26130258, 0.27577711])
    parser.add_argument('--random_crop_ratio', nargs='+', type=float, default=[0.8, 1.0])

    #train step
    parser.add_argument("--train_bs", type=int, help="Batch size for training.",default=8)
    parser.add_argument("--test_bs", type=int, help="Batch size for testing.",default=1)
    parser.add_argument("--num_workers", type=int, help="Number of workers for dataloader.",default=4)

    # model parameters
    parser.add_argument('--sample_expression_num', type=int, default=1)
    parser.add_argument('--sample_frame_stride', type=int, default=2)
    parser.add_argument("--truncation", type=int, help="Truncation for the model.",default=10)
    parser.add_argument("--sample_frame_len", type=int, help="Truncation for the model.",default=8)
    parser.add_argument("--sample_frame_num", type=int, help="Truncation for the model.",default=2)


    #checkpoint
    parser.add_argument("--clip_checkpoint_dir", type=str, help="Pretrained model path.",default="./checkpoints/CLIP")
    parser.add_argument("--resume", type=str, help="Resume checkpoint path.")
    parser.add_argument("--clip_model", type=str, help="Resume clip checkpoint path.",default="RN50.pt")
    parser.add_argument("--track_root", type=str, help="Track root path.",default="./track-dataset")    
    parser.add_argument("--track_name", type=str, help="Track data path.",default="NeuralSORT")
    parser.add_argument("--epochs_per_eval", type=int, help="epochs per eval",default=10)
    parser.add_argument("--epochs_per_checkpoint", type=int, help="epochs per eval",default=50)
    parser.add_argument("--module_checkpoint", type=str, help="Check point of module: IKUN | MEX",default="./checkpoints/MEX/MEX_99.pth")

    #submit
    parser.add_argument("--submit_save_root", type=str, help="Submit save root",default="./submit_outputs")    
    parser.add_argument("--module_name", type=str, help="Module type: IKUN | MEX",default="MEX")

    #inference
    parser.add_argument("--video_path", type=str, help="Video path for inference.")
    parser.add_argument("--caption", type=str, help="Caption for inference.")
    parser.add_argument("--memotr_config", type=str, help="Memotr config path.",default="./configs/memotr_bdd100k.yaml")
    parser.add_argument("--memotr_checkpoint", type=str, help="Memotr checkpoint path.",default="./checkpoints/MeMOTR/memotr_bdd100k.pth")
    parser.add_argument("--inference_save_root", type=str, help="Inference save root",default="./inference_outputs")
    parser.add_argument("--module_threshold", type=float, help="Module threshold",default=0.)
    parser.add_argument("--video_src", type=str, help="Video src for submit inference",default="D:/Thesis/DamnShit/Hello/MeMOTR_IKUN/models/mines")
    parser.add_argument("--tracker_threshold", type=float, help="Tracker threshold",default=0.2)
    parser.add_argument("--inf_w_mem", type=bool, help="Run inference with mem",default=True)
    parser.add_argument("--filter_type", type=str, help="Filter type: active | post",default="active")

    #testing
    parser.add_argument('--img_encoder', type=str, default='swintv2', help='image encoder to swintv2 | clip ')
    parser.add_argument("--outputs_dir", type=str, help="Checkpoint output dir",default="./checkpoints")
    parser.add_argument('--wandb', type=bool, help="Use wandb for logging", default=False)


 
    return parser.parse_args()



if __name__ == "__main__":
    opt = parse_option()   
    config = yaml_to_dict(opt.config_module)
    merge_config =update_config(config,opt)

    if config["MODE"] == "train":
        train(config=merge_config)
    elif config["MODE"] == "submit":
        submit(config)
    elif config["MODE"] == "inf":
        inference(config)
    elif config["MODE"] == "submitinf":
        submit_inference(config)

