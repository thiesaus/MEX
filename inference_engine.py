
import cv2
import torchvision.transforms.functional as F
import torch
import numpy as np
from model.MeMOTR.structures.track_instances import TrackInstances
import time
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
    text_thickness = 2
    line_thickness = 2

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=1)
    cv2.putText(im, 'Caption: {} , thresh {}'.format(caption,thresh),( 0, int(30 * text_scale)) , cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=1)

    alpha = 0.2

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        # id_text = '{}: {:.4f}'.format(int(obj_id), probs[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, color,
                    thickness=text_thickness)
        overlay = im.copy()
        cv2.rectangle(overlay, intbox[0:2], intbox[2:4], color=color, thickness=-1)
        cv2.addWeighted(overlay, alpha, im, 1-alpha, 0, im)
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
    

def demo_processing(
        config:dict,
        video_path: str,
        caption:str
):
    is_IKUN=False
    if config["MODULE_NAME"] == "IKUN":
        is_IKUN=True
    assert config["MODULE_NAME"] in ["IKUN","MEX"], "Module name should be either IKUN or MEX"
    model = build_model(config,is_IKUN=is_IKUN)
    load_checkpoint(
        model=model,
        path=config["MEMOTR_CHECKPOINT"],
    )
    model.eval()
    print("Model loaded.")
    current_time = time.localtime()
    cap = cv2.VideoCapture(video_path)
    print("Input video "+video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = os.path.join(config["INFERENCE_SAVE_ROOT"], timestamp)
    save_path = os.path.join(save_folder, "output.mp4")
    os.makedirs(save_folder, exist_ok=True)
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"MP4V"), 10, (int(width), int(height))
    )
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total frames: ", total_frames)
    assert total_frames > 0, "Video is empty."
    print((int(width), int(height)))
    tracker_threshold = config["TRACKER_THRESHOLD"]
    timer = Timer()
    frame_id = 0

    tracks = [TrackInstances(
        hidden_dim=model.memotr_model.hidden_dim,
        num_classes=model.memotr_model.num_classes,
        use_dab=config["USE_DAB"]
    ).to("cuda")]
    inter_module=INTERGRATE_MODULE(device=model.module.device, model=model.module,config=config)

    tracker = RuntimeTracker(
        det_score_thresh=0.5, 
        track_score_thresh=0.5,
        miss_tolerance=30,
        use_motion=False,
        motion_min_length=0, 
        motion_max_length=0,
        visualize=False, 
        inter_module=inter_module,
        use_dab=config["USE_DAB"],
        with_mem=config["INF_W_MEM"],
        filter_type=config["FILTER_TYPE"]
    )
    print("Caption " + caption)
    threshold=config["MODULE_THRESHOLD"]
    mems_report = []
    fps_report = []
    with torch.no_grad():
        while frame_id<total_frames:
            mem=torch.cuda.max_memory_allocated(device=model.module.device) / 1048576

            if frame_id % 20 == 0:
                print('Processing frame {} ({:.2f} fps) {:.2f}%'.format(frame_id, 1. / max(1e-5, timer.average_time), frame_id / total_frames * 100))
                print("Memory used: ", mem)

            # caption = "two person walking along the street"
            # if (frame_id >80 and frame_id < 160) or (frame_id > 240 and frame_id < 320):
            mems_report.append(mem)
            if frame_id != 0:
                fps_report.append(1. / max(1e-5, timer.average_time))

            caption =caption
            ret_val, ret_frame = cap.read()
            if ret_val:
                image = process_image(ret_frame)
                frame = tensor_list_to_nested_tensor([image[0]]).to("cuda")
                temp_img=Image.fromarray( np.asarray(cv2.cvtColor(ret_frame, cv2.COLOR_BGR2RGB)))

                timer.tic()
                res = model(frame=frame, tracks=tracks)
                previous_tracks, new_tracks = tracker.update(
                    model_outputs=res,
                    tracks=tracks,
                    temp_img=temp_img,
                    caption=caption,
                    threshold=threshold,
                    width=width,
                    height=height,
                )
               
                tracks: List[TrackInstances] = model.memotr_model.postprocess_single_frame(previous_tracks, new_tracks, None)
                
                if config["FILTER_TYPE"]=="post":
                    plot_track=tracker.post_process_filter(tracks,temp_img,caption,threshold,width,height)
                else:
                    plot_track=tracks
                    
                tracks_result = plot_track[0].to(torch.device("cpu"))
                # ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
                ori_h, ori_w = height, width
                # box = [x, y, w, h]
                tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                                     tracks_result.boxes[:, 3] * ori_h
                tracks_result = filter_by_score(tracks_result, thresh=tracker_threshold)
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
                timer.toc()
                if len(online_tlwhs) > 0:
                    online_im = plot_tracking(
                        ret_frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time,caption=caption,thresh=threshold,probs=tracks_result.probs
                    )
                else:
                    online_im = ret_frame
                vid_writer.write(online_im)
                # cv2.imsave(os.path.join(save_folder, "{:6d}.jpg".format(frame_id)), online_im)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1
    
    return os.path.join(save_folder, "output.mp4"),mems_report,fps_report

def inference(config:dict):
    memotr_config=config["MEMOTR_CONFIG"]
    video_path=config["VIDEO_PATH"]
    caption= ' '.join(config["CAPTION"].split("-"))
    memotr_dict = yaml_to_dict(memotr_config)
    merge_config=merge_dicts(memotr_dict,config)
    print(merge_config)
    output_path,mems_report,fps_report = demo_processing(
        config=merge_config,
        video_path=video_path,
        caption=caption,
    )
    print("Inference finished. Output video saved at: ", output_path)
