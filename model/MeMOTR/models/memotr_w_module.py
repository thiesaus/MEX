from torch import nn
from .memotr import MeMOTR
from utils.nested_tensor import NestedTensor
from model.MeMOTR.structures.track_instances import TrackInstances
from model.IKUN.ikun import IKUN,load_from_ckpt
from model.MEX.mex import MEX
from model.utils.utils import load_checkpoint


class MeMOTR_MODULE(nn.Module):
    def __init__(self, memotr_model:MeMOTR, config: dict,is_IKUN:bool=True):
        super().__init__()
        self.memotr_model = memotr_model
        checkpoint= config["MODULE_CHECKPOINT"]
        if is_IKUN:
            self.module= IKUN(config=config)
            self.module,_ = load_from_ckpt(self.module,checkpoint)
            self.module.to(self.module.device)
        
        else:
            self.module = MEX(config)
            load_checkpoint(self.module, checkpoint)

            

    def forward(self,  frame: NestedTensor, tracks: list[TrackInstances]):
        memotr_output = self.memotr_model(frame=frame,tracks=tracks)

        return  memotr_output


def build(memotr_model:MeMOTR,config: dict,is_IKUN:bool=True):
    return MeMOTR_MODULE(memotr_model=memotr_model,config=config,is_IKUN=is_IKUN)