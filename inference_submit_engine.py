
import cv2
import torchvision.transforms.functional as F
import torch
import numpy as np
from model.MeMOTR.structures.track_instances import TrackInstances
import time
import shutil
import gc
import os 
import numpy as np
from model.MeMOTR.models import build_model
from model.MeMOTR.models.utils import load_checkpoint
from utils.utils import yaml_to_dict
from PIL import Image
from typing import List
from model.MeMOTR.models.runtime_tracker import RuntimeTracker
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy
from model.MeMOTR.models.intergrate_module import INTERGRATE_MODULE

def merge_dicts(dict1, dict2):
    for key in dict2:
        dict1[key] = dict2[key]
    return dict1


def process_image(image):
    ori_image = image.copy()
    h, w = image.shape[:2]
    scale = 800 / min(h, w)
    if max(h, w) * scale > 1536:
        scale = 1536 / max(h, w)
    target_h = int(h * scale)
    target_w = int(w * scale)
    image = cv2.resize(image, (target_w, target_h))
    image = F.normalize(F.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image, ori_image


def filter_by_score(tracks: TrackInstances, thresh: float = 0.7):
    keep = torch.max(tracks.scores, dim=-1).values > thresh
    return tracks[keep]

def filter_by_area(tracks: TrackInstances, thresh: int = 100):
    assert len(tracks.area) == len(tracks.ids), f"Tracks' 'area' should have the same dim with 'ids'"
    keep = tracks.area > thresh
    return tracks[keep]


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None,caption="",thresh=0.7,probs=None):
    # Thanks to https://github.com/noahcao/OC_SORT
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = 1
    text_thickness = 1
    line_thickness = 2

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=1)
    cv2.putText(im, 'Caption: {} , thresh {}'.format(caption,thresh),( 0, int(30 * text_scale)) , cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=1)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}: {:.4f}'.format(int(obj_id), probs[i])
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration
    

class SUBMIT_INFERENCE:
    def __init__(self,config:dict):
        self.config=config
        is_IKUN=False
        if config["MODULE_NAME"] == "IKUN":
            is_IKUN=True
        self.model = build_model(config,is_IKUN=is_IKUN)
        load_checkpoint(
            model=self.model,
            path=config["MEMOTR_CHECKPOINT"],
        )
        self.model.eval()
        self.inter_module=INTERGRATE_MODULE(device=self.model.module.device, model=self.model.module,config=config)
     
        self.threshold=config["MODULE_THRESHOLD"]
        self.captions=[]
        self.track_threshold=config["TRACKER_THRESHOLD"]

   
        

    def clean_up(self):
        gc.collect()
        torch.cuda.empty_cache()
        
    def generate_submit(self,video_path,caption): # caption: men-in-black
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        assert total_frames > 0, "Video is empty."
        video_name=os.path.basename(video_path).split(".")[0]
        infernce_submit_path="./inference_submit"
        video_output_path=os.path.join(infernce_submit_path,video_name)
        os.makedirs(video_output_path,exist_ok=True)
        video_caption=os.path.join(video_output_path,caption)
        os.makedirs(video_caption,exist_ok=True)
        predict_path=os.path.join(video_caption,"predict.txt")
        if not os.path.exists(predict_path):
            with open(predict_path, "w") as f:
                f.write("")

        edited_caption= ' '.join(caption.split("-"))
        frame_id = 0
        tracks = [TrackInstances(
        hidden_dim=self.model.memotr_model.hidden_dim,
        num_classes=self.model.memotr_model.num_classes,
        use_dab=self.config["USE_DAB"],
        ).to("cuda")]
        tracker = RuntimeTracker(
            det_score_thresh=0.5, 
            track_score_thresh=0.5,
            miss_tolerance=30,
            use_motion=False,
            motion_min_length=0, 
            motion_max_length=0,
            visualize=False, 
            use_dab=self.config["USE_DAB"],
            with_mem=self.config["INF_W_MEM"],
        )
        with torch.no_grad():
            while frame_id <total_frames:
                if frame_id % 20 == 0:
                    print(f"Processing frame {frame_id}/{total_frames}, video {video_name}, caption {caption}")
                ret_val, ret_frame = cap.read()
                if ret_val:
                    image = process_image(ret_frame)
                    frame = tensor_list_to_nested_tensor([image[0]]).to("cuda")
                    temp_img=Image.fromarray( np.asarray(cv2.cvtColor(ret_frame, cv2.COLOR_BGR2RGB)))

                    res = self.model(frame=frame, tracks=tracks)
                    previous_tracks, new_tracks = tracker.update(
                        model_outputs=res,
                        tracks=tracks,
                        temp_img=temp_img,
                        inter_module=self.inter_module,
                        caption=edited_caption,
                        threshold=self.threshold,
                        width=width,
                        height=height,
                    )
                
                    tracks: List[TrackInstances] = self.model.memotr_model.postprocess_single_frame(previous_tracks, new_tracks, None)
            
                    tracks_result = tracks[0].to(torch.device("cpu"))
                    # ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
                    ori_h, ori_w = height, width
                    # box = [x, y, w, h]
                    tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                        tracks_result.boxes[:, 3] * ori_h
                    tracks_result = filter_by_score(tracks_result, thresh=self.track_threshold)
                    tracks_result = filter_by_area(tracks_result)
                    # to xyxy:
                    tracks_result.boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
                    tracks_result.boxes = (tracks_result.boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float))
                    online_tlwhs, online_ids = [], []
                    for i in range(len(tracks_result)):
                        x1, y1, x2, y2 = tracks_result.boxes[i].tolist()
                        w, h = x2 - x1, y2 - y1
                        online_tlwhs.append([x1, y1, w, h])
                        online_ids.append(tracks_result.ids[i].item())
                        with open(predict_path, "a") as f:
                            f.write(f"{frame_id+1},{tracks_result.ids[i].item()},{x1},{y1},{w},{h},1.0,-1,-1,-1\n")
                 
                else:
                    break
                frame_id += 1
        self.clean_up()
        return predict_path

    def get_video_caption(self):
        output= dict()
        gt_path=os.path.join(self.config["TRACK_ROOT"],"gt_template")
        video_list=os.listdir(gt_path)
        for video in video_list:
            video_path=os.path.join(gt_path,video)
            caption_list=os.listdir(video_path)
            output[video]=caption_list
        return output

    def run(self):
        video_caption=self.get_video_caption()
        for video in video_caption:
            idx=0
            for caption in video_caption[video]:
                idx=idx+1
                print("Video: ",video)
                print("Caption: ",caption, " Process {}/{}".format(idx,len(video_caption[video])))
                predict_path=self.generate_submit(video_path=os.path.join(self.config["VIDEO_SRC"],f"{video}.mp4"),caption=caption)
                print("Predict path: ",predict_path)
                #copy gt to submit folder
                gt_path=os.path.join(self.config["TRACK_ROOT"],"gt_template",video,caption,"gt.txt")
                submit_path=predict_path.replace("predict.txt","gt.txt")
                shutil.copyfile(gt_path,submit_path)
            
                print("Done")
        shutil.make_archive("./inference_submit", 'zip', "./inference_submit")
        print("All done")



def submit_inference(config):
    memotr_config=config["MEMOTR_CONFIG"]
    memotr_dict = yaml_to_dict(memotr_config)
    merge_config=merge_dicts(memotr_dict,config)
    submit_inference=SUBMIT_INFERENCE(config=merge_config)
    submit_inference.run()