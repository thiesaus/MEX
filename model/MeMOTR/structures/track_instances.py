# Copyright (c) Ruopeng Gao. All Rights Reserved.
import torch

from typing import List


class TrackInstances:
    """
    Tracked Instances.
    """
    def __init__(self, frame_height: float = 1.0, frame_width: float = 1.0,
                 hidden_dim: int = 256, num_classes: int = 1, use_dab: bool = False):
        self.use_dab = use_dab
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        if self.use_dab:
            self.ref_pts = torch.zeros((0, 4))
            self.query_embed = torch.zeros((0, hidden_dim))
        else:
            # self.ref_pts = torch.zeros((0, 2))
            self.ref_pts = torch.zeros((0, 4))
            self.query_embed = torch.zeros((0, 2 * hidden_dim))
        self.ids = torch.zeros((0,), dtype=torch.long)
        self.boxes = torch.zeros((0, 4))
        self.labels = torch.zeros((0,), dtype=torch.long)
        self.logits = torch.zeros((0, self.num_classes))
        self.matched_idx = torch.zeros((0, ), dtype=torch.long)
        self.output_embed = torch.zeros((0, self.hidden_dim))
        self.disappear_time = torch.zeros((0,), dtype=torch.long)
        self.scores = torch.zeros((0,), dtype=torch.float)
        self.area = torch.zeros((0,), dtype=torch.float)
        self.iou = torch.zeros((0,), dtype=torch.float)
        self.last_output = torch.zeros((0, self.hidden_dim), dtype=torch.float)
        self.long_memory = torch.zeros((0, self.hidden_dim), dtype=torch.float)
        self.last_appear_boxes = torch.zeros((0, 4))
        self.probs = []
        self.first_images=[]
        self.first_global_images=[]

    
    def removePosition(self, idx: int):
        self.ref_pts = torch.cat((self.ref_pts[:idx], self.ref_pts[idx + 1:]))
        self.query_embed = torch.cat((self.query_embed[:idx], self.query_embed[idx + 1:]))
        self.ids = torch.cat((self.ids[:idx], self.ids[idx + 1:]))
        self.boxes = torch.cat((self.boxes[:idx], self.boxes[idx + 1:]))
        self.labels = torch.cat((self.labels[:idx], self.labels[idx + 1:]))
        self.logits = torch.cat((self.logits[:idx], self.logits[idx + 1:]))
        self.matched_idx = torch.cat((self.matched_idx[:idx], self.matched_idx[idx + 1:]))
        self.output_embed = torch.cat((self.output_embed[:idx], self.output_embed[idx + 1:]))
        self.disappear_time = torch.cat((self.disappear_time[:idx], self.disappear_time[idx + 1:]))
        self.scores = torch.cat((self.scores[:idx], self.scores[idx + 1:]))
        self.area = torch.cat((self.area[:idx], self.area[idx + 1:]))
        self.iou = torch.cat((self.iou[:idx], self.iou[idx + 1:]))
        self.last_output = torch.cat((self.last_output[:idx], self.last_output[idx + 1:]))
        self.long_memory = torch.cat((self.long_memory[:idx], self.long_memory[idx + 1:]))
        self.last_appear_boxes = torch.cat((self.last_appear_boxes[:idx], self.last_appear_boxes[idx + 1:]))
        self.probs.pop(idx)
        self.first_images.pop(idx)
        self.first_global_images.pop(idx)
    def setFirstImage(self,first_images,probs):
        self.first_images=first_images
        self.probs=probs
    def setProbs(self,probs):
        self.probs=probs

    def setFirstGlobalImage(self,first_global_images):
        self.first_global_images=first_global_images
        
    def to(self, device):
        res = TrackInstances(frame_height=self.frame_height, frame_width=self.frame_width,
                             hidden_dim=self.hidden_dim, num_classes=self.num_classes)
        for k, v in vars(self).items():
            if hasattr(v, "to"):
                v = v.to(device)
            res.__setattr__(k, v)
        return res

    def __len__(self):
        assert self.ref_pts.shape[0] == self.query_embed.shape[0]
        return max(self.query_embed.shape[0], self.labels.shape[0])

    def __getitem__(self, item: int | slice | torch.BoolTensor) -> "TrackInstances":
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("TrackInstances index out of range!")
            else:
                item = slice(item, None, len(self))
        res = TrackInstances(frame_height=self.frame_height, frame_width=self.frame_width,
                             hidden_dim=self.hidden_dim, num_classes=self.num_classes)
        for k, v in vars(self).items():
            if k == "first_images" or k == "probs" or k == "first_global_images":
                result=[]
                for i in range(len(item)):
                    flag = item[i].item() if type(item) == torch.Tensor else item[i]
                    if flag:
                        result.append(v[i])
                res.__setattr__(k,result)
            else:
                if hasattr(v, "__getitem__") and v.shape[0] != 0:
                    res.__setattr__(k, v[item])
                else:
                    res.__setattr__(k, v)
        return res

    @staticmethod
    def init_tracks(batch: dict, hidden_dim: int, num_classes: int, device="cpu", use_dab: bool = False):
        """
        Init tracks for a batch.
        """
        tracks_list = []
        h_max, w_max = 0, 0
        for i in range(len(batch["imgs"])):
            h_max = max(batch["imgs"][i][0].shape[-2], h_max)
            w_max = max(batch["imgs"][i][0].shape[-1], w_max)
        for i in range(len(batch["imgs"])):
            tracks_list.append(TrackInstances(
                frame_height=float(batch["imgs"][i][0].shape[-2] / h_max),
                frame_width=float(batch["imgs"][i][0].shape[-1] / w_max),
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                use_dab=use_dab
            ).to(device))
        return tracks_list

    @staticmethod
    def cat_tracked_instances(tracked1: "TrackInstances", tracked2: "TrackInstances"):
        res = TrackInstances(frame_height=tracked1.frame_height, frame_width=tracked1.frame_width)

        for k, v in vars(tracked1).items():
            if type(v) is torch.Tensor:
                res.__setattr__(k, torch.cat((getattr(tracked1, k), getattr(tracked2, k))))
            if type(v) is list:
                res.__setattr__(k, getattr(tracked1, k)+getattr(tracked2, k))
        return res

    @staticmethod
    def tracks_to_meta_tensors(tracks: List["TrackInstances"]):
        keys = [k for k in vars(tracks[0]) if type(getattr(tracks[0], k)) is torch.Tensor]
        meta = {
            "frame_height": [],
            "frame_width": [],
            "hidden_dim": [],
            "num_classes": [],
            "keys": keys
        }
        tensors = []
        for b in range(len(tracks)):
            meta["frame_height"].append(tracks[b].frame_height)
            meta["frame_width"].append(tracks[b].frame_width)
            meta["hidden_dim"].append(tracks[b].hidden_dim)
            meta["num_classes"].append(tracks[b].num_classes)
            for k in keys:
                tensors.append(getattr(tracks[b], k))
        return meta, tensors

    @staticmethod
    def meta_tensors_to_tracks(meta: dict, tensors: List[torch.Tensor]):
        tracks = []
        for b in range(len(meta["frame_height"])):
            track = TrackInstances(
                frame_height=meta["frame_height"][b],
                frame_width=meta["frame_width"][b],
                hidden_dim=meta["hidden_dim"][b],
                num_classes=meta["num_classes"][b]
            )
            for i, k in enumerate(meta["keys"]):
                setattr(track, k, tensors[i + b * len(meta["keys"])])
            tracks.append(track)
        return tracks
