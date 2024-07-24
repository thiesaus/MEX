
import torch

from utils.utils import distributed_rank
from .memotr import build as build_memotr
from .memotr_w_module import build as build_memotr_w_module
from .utils import load_pretrained_model


def build_model(config: dict,is_IKUN=False):
    model = build_memotr(config=config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    if config["MEMOTR_CHECKPOINT"] is not None:
        model = load_pretrained_model(model, config["MEMOTR_CHECKPOINT"], show_details=False)


    my_model = build_memotr_w_module(memotr_model=model, config=config, is_IKUN=is_IKUN)
    return my_model
