
import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from collections import defaultdict
from data.dataloader import get_dataloader,EXPRESSIONS
from model.MEX.mex import MEX,build_MEX
import torch
from torch import nn
import torch.nn.functional as F
from utils.similarity_calibration import similarity_calibration

def dd():
    return defaultdict(list)
def ddd():
    return defaultdict(dd)
def dddd():
    return defaultdict(ddd)
def multi_dim_dict(n, types):
   
    return defaultdict(dddd)
FRAMES = {
    '0005': (0, 296),
    '0011': (0, 372),
    '0013': (0, 339),
}  # 视频起止帧


def test_tracking(model, dataloader):
    print('========== Testing Tracking ==========')
    model.eval()
    OUTPUTS = multi_dim_dict(4, list)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
            # forward
            inputs = dict(
                local_images=data['cropped_images'].cuda(),
                global_image=data['global_images'].cuda(),
                sentences=data['expression_new'],
            )
            similarity = model(inputs)['scores'].cpu()
            for idx in range(len(data['video'])):
                for frame_id in range(data['start_frame'][idx], data['stop_frame'][idx] + 1):
                    frame_dict = OUTPUTS[data['video'][idx]][int(data['obj_id'][idx])][int(frame_id)]
                    frame_dict[data['expression_raw'][idx]].append(similarity[idx].cpu().numpy().tolist())
    return OUTPUTS


def generate_final_results(cls_dict, template_dir, track_dir, save_dir, thr_score=0.):
    """
    给定`test_tracking`输出的结果，生成最终跟踪结果
    - cls_dict: video->id->frame->exp->
    """
    if exists(save_dir):
        shutil.rmtree(save_dir)
    for video in os.listdir(template_dir):
        if video not in cls_dict:
            continue
        video_dir_in = join(template_dir, video)
        video_dir_out = join(save_dir, video)
        MIN_FRAME, MAX_FRAME = FRAMES[video]
        # symbolic link for `gt.txt`
        for exp in os.listdir(video_dir_in):
            exp_dir_in = join(video_dir_in, exp)
            exp_dir_out = join(video_dir_out, exp)
            os.makedirs(exp_dir_out, exist_ok=True)
            gt_path_in = join(exp_dir_in, 'gt.txt')
            gt_path_out = join(exp_dir_out, 'gt.txt' )
            if not exists(gt_path_out):
                os.symlink(gt_path_in, gt_path_out)
        # load tracks
        # noinspection PyBroadException
        try:
            tracks = np.loadtxt(join(track_dir, video, 'all', 'gt.txt'), delimiter=',')
        except:
            tracks_1 = np.loadtxt(join(track_dir, video, 'car', 'predict.txt'), delimiter=',')
            if len(tracks_1.shape) == 2:
                tracks = tracks_1
                max_obj_id = max(tracks_1[:, 1])
            else:
                tracks = np.empty((0, 10))
                max_obj_id = 0
            tracks_2 = np.loadtxt(join(track_dir, video, 'pedestrian', 'predict.txt'), delimiter=',')
            if len(tracks_2.shape) == 2:
                tracks_2[:, 1] += max_obj_id
                tracks = np.concatenate((tracks, tracks_2), axis=0)
        # generate `predict.txt`
        video_dict = cls_dict[video]
        for obj_id, obj_dict in video_dict.items():
            for frame_id, frame_dict in obj_dict.items():
                for exp in EXPRESSIONS[video]:
                    if exp in EXPRESSIONS['dropped']:
                        continue
                    if exp not in frame_dict:  # TODO:可删
                        continue
                    exp_dir_out = join(video_dir_out, exp)
                    score = np.mean(frame_dict[exp])
                    with open(join(exp_dir_out, 'predict.txt'), 'a') as f:
                        if score > thr_score:
                            bbox = tracks[
                                (tracks[:, 0] == int(frame_id)) *
                                (tracks[:, 1] == int(obj_id))
                            ][0]
                            assert bbox.shape in ((9, ), (10, ))
                            if MIN_FRAME < bbox[0] < MAX_FRAME:  # TODO
                                # the min/max frame is not included in `gt.txt`
                                f.write(','.join(list(map(str, bbox))) + '\n')

def submit(opt: dict):
    model = build_MEX(config=opt).cuda()
    load_state = torch.load(opt["MODULE_CHECKPOINT"], map_location="cpu")
    model.load_state_dict(load_state["model"])
    dataloader = get_dataloader('test', opt, 'Track_Dataset')
    print(
    '========== Testing (Text-Guided {}) =========='
        .format('ON')
    )
    output_path = join(opt["SUBMIT_SAVE_ROOT"],opt["MODULE_NAME"], f'results_{opt["MODULE_NAME"]}_{opt["TRACK_NAME"]}.json')

    if not exists(output_path):

        output = test_tracking(model, dataloader)
        os.makedirs(join(opt["SUBMIT_SAVE_ROOT"], opt["MODULE_NAME"]), exist_ok=True)
        json.dump(
            output,
            open(output_path, 'w')
        )

    SAVE_DIR = join(opt["SUBMIT_SAVE_ROOT"], opt["MODULE_NAME"], f'results_{opt["MODULE_NAME"]}_{opt["TRACK_NAME"]}')
    CLS_DICT = json.load(open(output_path))

    if True:
        TEXT_FEAT_DICT = json.load(open(os.path.join(opt["DATA_ROOT"], 'textual_features.json')))
        CLS_DICT = similarity_calibration(
            TEXT_FEAT_DICT,
            CLS_DICT,
            a=8,
            b=-0.1,
            tau=100
        )

    generate_final_results(
        cls_dict=CLS_DICT,
        template_dir=os.path.join(opt["TRACK_ROOT"], "gt_template"),
        track_dir=os.path.join(opt["TRACK_ROOT"],opt["TRACK_NAME"]),
        save_dir=SAVE_DIR,
    )