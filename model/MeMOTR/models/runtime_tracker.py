# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os

import torch

from typing import List, Dict
from .utils import logits_to_scores
from .motion import Motion
from PIL import Image 

from model.MeMOTR.structures.track_instances import TrackInstances

class RuntimeTracker:
    def __init__(self, det_score_thresh: float = 0.7, track_score_thresh: float = 0.6,
                 miss_tolerance: int = 5,
                 use_motion: bool = False, motion_min_length: int = 3, motion_max_length: int = 5,
                 visualize: bool = False, use_dab: bool = True,with_mem:bool=True,inter_module=None,filter_type:str="active"):
        self.det_score_thresh = det_score_thresh
        self.track_score_thresh = track_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.use_motion = use_motion
        self.visualize = visualize
        self.motion_min_length = motion_min_length
        self.motion_max_length = motion_max_length
        self.motions: Dict[Motion] = {}
        self.use_dab = use_dab
        self.with_mem=with_mem
        self.inter_module=inter_module
        self.filter_type=filter_type
        

    def active_filter(self,tracks:List[TrackInstances],new_tracks:List[TrackInstances],
                      temp_img:Image,caption:str,threshold:float,width:int,height:int):
        if self.with_mem:
            old_images=tracks[-1].first_images
        else:
            old_images=[]
        _probs,crop_image= self.inter_module.predict(caption=caption, temp_img=temp_img, new_bbox=new_tracks.boxes,width=width,height=height,old_images=old_images)
        if len(_probs)==0:
            _probs=torch.Tensor([]).to(new_tracks.logits.device)
        mask=_probs>=threshold
        new_image_len=len(new_tracks.boxes)
        previous_image_len=len(tracks[-1].boxes)
        probs = _probs.tolist()
        # print("true_objects : "+ str(sum(mask) ))
        latency=0
        if self.with_mem:
            new_tracks.setFirstImage(crop_image,probs[:new_image_len])
        for i in range(new_image_len):
            if mask[i] == False:
                new_tracks.removePosition(i-latency)
                latency+=1
        latency=0
        tracks[-1].setProbs(probs[new_image_len:])
        for i in range(previous_image_len):
            if mask[i+new_image_len] == False:
                tracks[-1].removePosition(i-latency)
                latency+=1
        return new_tracks,tracks
    
    def post_process_filter(self,tracks:List[TrackInstances],  temp_img:Image,caption:str,threshold:float,width:int,height:int):
        _probs,crop_image= self.inter_module.predict(caption=caption, temp_img=temp_img, new_bbox=tracks.boxes,width=width,height=height,old_images=[])
        if len(_probs)==0:
            _probs=torch.Tensor([]).to(tracks.logits.device)
        mask=_probs>=threshold
        result=tracks.copy()
        for i in range(len(mask)):
            if mask[i] == False:
                result[0].removePosition(i)
        return result

    def update(self, model_outputs: dict, tracks: List[TrackInstances],
               temp_img=None,inter_module=None,caption="",
               threshold=0.,width=None,height=None):
        assert len(tracks) == 1
        model_outputs["scores"] = logits_to_scores(model_outputs["pred_logits"])
        n_dets = len(model_outputs["det_query_embed"])

        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/runtime_tracker/", exist_ok=True)
            visualize_ids = tracks[0].ids.cpu().tolist()

        # Update tracks.
        tracks[0].boxes = model_outputs["pred_bboxes"][0][n_dets:]
        tracks[0].logits = model_outputs["pred_logits"][0][n_dets:]
        tracks[0].output_embed = model_outputs["outputs"][0][n_dets:]
        tracks[0].scores = logits_to_scores(tracks[0].logits)
        for i in range(len(tracks[0])):
            if tracks[0].scores[i][tracks[0].labels[i]] < self.track_score_thresh:
                tracks[0].disappear_time[i] += 1
            else:
                if self.use_motion and tracks[0].disappear_time[i] > 0:
                    self.motions[tracks[0].ids[i].item()].clear()
                tracks[0].disappear_time[i] = 0
                if self.use_motion:
                    self.motions[tracks[0].ids[i].item()].add_box(tracks[0].boxes[i].cpu())
                    tracks[0].last_appear_boxes[i] = tracks[0].boxes[i]
            if tracks[0].disappear_time[i] >= self.miss_tolerance:
                tracks[0].ids[i] = -1

        # Add newborn targets.
        new_tracks = TrackInstances(hidden_dim=tracks[0].hidden_dim, num_classes=tracks[0].num_classes)
        new_tracks_idxes = torch.max(model_outputs["scores"][0][:n_dets], dim=-1).values >= self.det_score_thresh
        new_tracks.logits = model_outputs["pred_logits"][0][:n_dets][new_tracks_idxes]
        new_tracks.boxes = model_outputs["pred_bboxes"][0][:n_dets][new_tracks_idxes]
        new_tracks.ref_pts = model_outputs["last_ref_pts"][0][:n_dets][new_tracks_idxes]
        new_tracks.scores = model_outputs["scores"][0][:n_dets][new_tracks_idxes]
        new_tracks.output_embed = model_outputs["outputs"][0][:n_dets][new_tracks_idxes]
        # new_tracks.query_embed = model_outputs["aux_outputs"][-1]["queries"][0][:n_dets][new_tracks_idxes]
        if self.use_dab:
            new_tracks.query_embed = model_outputs["aux_outputs"][-1]["queries"][0][:n_dets][new_tracks_idxes]
        else:
            new_tracks.query_embed = torch.cat(
                (
                    model_outputs["det_query_embed"][new_tracks_idxes][:, :256],    # hack
                    model_outputs["aux_outputs"][-1]["queries"][0][:n_dets][new_tracks_idxes]
                ),
                dim=-1
            )
        new_tracks.disappear_time = torch.zeros((len(new_tracks.logits), ), dtype=torch.long)
        new_tracks.labels = torch.max(new_tracks.scores, dim=-1).indices

        # We do not use this post-precess motion module in our final version,
        # this will bring a slight improvement,
        # but makes un-elegant.
        if self.use_motion:
            new_tracks.last_appear_boxes = model_outputs["pred_bboxes"][0][:n_dets][new_tracks_idxes]
        ids = []
        if self.inter_module is not None and self.filter_type=="active":
           # ================== INTER MODEL ==================
            new_tracks,tracks=self.active_filter(tracks,new_tracks,temp_img,caption,threshold,width,height)
            # ================== INTER MODEL ==================
          
        for i in range(len(new_tracks)):
            ids.append(self.max_obj_id)
            self.max_obj_id += 1
        new_tracks.ids = torch.as_tensor(ids, dtype=torch.long)
        new_tracks = new_tracks.to(new_tracks.logits.device)
        for _ in range(len(new_tracks)):
            self.motions[new_tracks.ids[_].item()] = Motion(
                min_record_length=self.motion_min_length,
                max_record_length=self.motion_max_length
            )
            self.motions[new_tracks.ids[_].item()].add_box(new_tracks.boxes[_].cpu())

        if self.visualize:
            visualize_ids += ids
            torch.save(torch.as_tensor(visualize_ids),
                       "./outputs/visualize_tmp/runtime_tracker/ids.tensor")
            
     
        return tracks, [new_tracks]
