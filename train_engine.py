import os
import torch
import wandb
from logger import Logger,MetricLog
from torch.optim import  AdamW
from utils.utils import   distributed_rank, set_seed,distributed_world_size,is_main_process
from typing import List, Tuple, Dict
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import time
from model.utils.utils import get_model, save_checkpoint, load_checkpoint

from model.MEX.mex import MEX,build_MEX
from torch.utils.data import DataLoader
from model.common.criterion import ModuleCriterion,build_criterion
from model.common.loss import SimilarityLoss
from model.common.accuracy import test_accuracy
from data.dataloader import get_dataloader

# from model.test_hook import test

sim_loss = SimilarityLoss(
    rho=None,
    gamma=2.0,
    reduction="sum",
)


def train(config: dict):
    train_logger = Logger(logdir=os.path.join( config["OUTPUTS_DIR"], "train"), only_main=True)
    train_logger.show(head="Configs:", log=config)
    train_logger.write(log=config, filename="config.yaml", mode="w")
    train_logger.tb_add_git_version(git_version=config["GIT_VERSION"])

    set_seed(config["SEED"])

    model = build_MEX(config=config)
    
    # hook_test=test(model)

    # Load Pretrained Model
 
    # Data process
    dataloader_train = get_dataloader('train', config, 'RMOT_Dataset', show=True)
    dataloader_test = get_dataloader('test', config, 'RMOT_Dataset', show=False)


    # Criterion
    criterion = build_criterion(config=config)
    criterion.set_device(torch.device("cuda", distributed_rank()))

    # Optimizer
    param_groups, lr_names = get_param_groups(config=config, model=model)
    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

    # Wandb
    if config["WANDB"] == True:
        wandb.init(
        project="module_space", 
        config={
        "architecture": "Transformer",
        "epochs": config["EPOCHS"],

        })
    


    # Scheduler
    if config["LR_SCHEDULER"] == "MultiStep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config["LR_DROP_MILESTONES"],
            gamma=config["LR_DROP_RATE"]
        )
    elif config["LR_SCHEDULER"] == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config["EPOCHS"]
        )
    else:
        raise ValueError(f"Do not support lr scheduler '{config['LR_SCHEDULER']}'")

    # Training states
    train_states = {
        "start_epoch": 0,
        "global_iters": 0
    }

    # Resume
    if config["RESUME"] is not None:
        if config["RESUME_SCHEDULER"]:
            load_checkpoint(model=model, path=config["RESUME"], states=train_states,
                            optimizer=optimizer, scheduler=scheduler)
        else:
            load_checkpoint(model=model, path=config["RESUME"], states=train_states)
            for _ in range(train_states["start_epoch"]):
                scheduler.step()

    # Set start epoch
    start_epoch = train_states["start_epoch"]

    # if is_distributed():
    #     model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=False)

    multi_checkpoint = "MULTI_CHECKPOINT" in config and config["MULTI_CHECKPOINT"]
    # visualizer=Visualize()
    # Training:

    for epoch in range(start_epoch, config["EPOCHS"]):

        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        assert len(lrs) == len(lr_names)
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        train_logger.show(head=f"[Epoch {epoch}] lr={lr_info}")
        train_logger.write(head=f"[Epoch {epoch}] lr={lr_info}")
        default_lr_idx = -1
        for _ in range(len(lr_names)):
            if lr_names[_] == "lr":
                default_lr_idx = _
        train_logger.tb_add_scalar(tag="lr", scalar_value=lrs[default_lr_idx], global_step=epoch, mode="epochs")

        no_grad_frames = None
        if "NO_GRAD_FRAMES" in config:
            for i in range(len(config["NO_GRAD_STEPS"])):
                if epoch >= config["NO_GRAD_STEPS"][i]:
                    no_grad_frames = config["NO_GRAD_FRAMES"][i]
                    break
             
        output_dict=train_one_epoch(
            model=model,
            train_states=train_states,
            max_norm=config["CLIP_MAX_NORM"],
            dataloader=dataloader_train,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            # metric_log=train_metric_log,
            logger=train_logger,
            accumulation_steps=config["ACCUMULATION_STEPS"],
            multi_checkpoint=multi_checkpoint,
        )
        
        if (epoch+1) % config["EPOCHS_PER_EVAL"] ==0:
            p,r=test_one_epoch(model=model,dataloader_test=dataloader_test,epoch=epoch)
            f1_score = 2 * p * r / (p + r + 1e-6)
            output_dict["test"]=dict(epoch=epoch,precision=p,recall=r,f1_score=f1_score)
        if config["WANDB"] == True:
            wandb.log(output_dict)
        scheduler.step()

        train_states["start_epoch"] += 1
        if multi_checkpoint is True:
            pass
        else:
            if (epoch + 1) % config["EPOCHS_SPACE"] == 0:
                save_checkpoint(
                    model=model,
                    path=os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth"),
                    states=train_states,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
    if config["WANDB"] == True:
        wandb.finish()
    return

def get_param_groups(config: dict, model: nn.Module) -> Tuple[List[Dict], List[str]]:
    """
    用于针对不同部分的参数使用不同的 lr 等设置
    Args:
        config: 实验的配置信息
        model: 需要训练的模型

    Returns:
        params_group: a list of params groups.
        lr_names: a list of params groups' lr name, like "lr_backbone".
    """
    def match_keywords(name: str, keywords: List[str]):
        matched = False
        for keyword in keywords:
            if keyword in name:
                matched = True
                break
        return matched
    # keywords
    backbone_keywords = ["swinv2_model","bert_model"]
    process_keywords = ["reprocess_image","text_linear"]
    fusion_keywords = ["fusion_text_local", "fusion_text_global","full_fusion_layer","repeat_text_layer"]  # 在 transformer 中用于选取参考点和采样点的网络参数关键字
    param_groups = [
        {   # backbone 学习率设置
            "params": [p for n, p in model.named_parameters() if match_keywords(n, backbone_keywords) and p.requires_grad],
            "lr": config["LR_BACKBONE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, fusion_keywords)
                       and p.requires_grad],
            "lr": config["LR_POINTS"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, process_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        },
        {
            "params": [p for n, p in model.named_parameters() if not match_keywords(n, backbone_keywords)
                       and not match_keywords(n, fusion_keywords)
                       and not match_keywords(n, process_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        }
    ]
    return param_groups, ["lr_backbone", "lr_fusion", "lr_middle_fusion", "lr"]


def train_one_epoch(model: MEX, train_states: dict, max_norm: float,
                    dataloader: DataLoader, criterion: ModuleCriterion, optimizer: torch.optim,
                    epoch: int, logger: Logger,
                    accumulation_steps: int = 1, 
                    multi_checkpoint: bool = False):
    """
    Args:
        model: Model.
        train_states:
        max_norm: clip max norm.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Training optimizer.
        epoch: Current epoch.
        # metric_log: Metric Log.
        logger: unified logger.
        accumulation_steps:
        use_dab:
        multi_checkpoint:
        no_grad_frames:

    Returns:
        Logs
    """
    model.train()
    device = next(get_model(model).parameters()).device

    dataloader_len=len(dataloader)
    metric_log = MetricLog()
    epoch_start_timestamp = time.time()
    output_dict=dict()
    for i, data in enumerate(dataloader):
 
        expression = data['target_expressions']
        expression_ids = data['expression_id'].to(device)
        targets = data['target_labels'].view(-1).to(device)

        # forward
        inputs = dict(
            local_images=data['cropped_images'].to(device),
            global_image=data['global_images'].to(device),
            labels=targets,
            sentences=expression,
        )
        iter_start_timestamp = time.time()

        model_outputs= model(inputs)

        scores = model_outputs['scores']
        loss =sim_loss(scores, targets)
        # Metrics log
        metric_log.update(name="total_loss", value=loss.item())
        loss = loss / accumulation_steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output_dict["train"]=dict(epoch=epoch,loss=loss.item())
     
        iter_end_timestamp = time.time()
        metric_log.update(name="time per iter", value=iter_end_timestamp-iter_start_timestamp)
        # Outputs logs
        if i % 10 == 0:
            metric_log.sync()
            max_memory = max([torch.cuda.max_memory_allocated(torch.device('cuda', i))
                            for i in range(distributed_world_size())]) // (1024**2)
            second_per_iter = metric_log.metrics["time per iter"].avg
            
            logger.show(head=f"[Epoch={epoch}, Iter={i}, "
                            f"{second_per_iter:.2f}s/iter, "
                            f"{i}/{dataloader_len} iters, "
                            f"rest time: {int(second_per_iter * (dataloader_len - i) // 60)} min, "
                            f"Max Memory={max_memory}MB]",
                        log=metric_log)
            logger.write(head=f"[Epoch={epoch}, Iter={i}/{dataloader_len}]",
                        log=metric_log, filename="log.txt", mode="a")
            logger.tb_add_metric_log(log=metric_log, steps=train_states["global_iters"], mode="iters")

        if multi_checkpoint:
            if i % 10 == 0 and is_main_process():
                save_checkpoint(
                    model=model,
                    path=os.path.join(logger.logdir[:-5], f"checkpoint_{int(i // 10)}.pth")
                )

        train_states["global_iters"] += 1

    # Epoch end
    metric_log.sync()
    epoch_end_timestamp = time.time()
    epoch_minutes = int((epoch_end_timestamp - epoch_start_timestamp) // 60)
    logger.show(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                log=metric_log)
    logger.write(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                 log=metric_log, filename="log.txt", mode="a")
    logger.tb_add_metric_log(log=metric_log, steps=epoch, mode="epochs")
    output_dict["metric_log"]=dict(loss=metric_log.get_avg())
    return output_dict

def test_one_epoch(model:MEX,dataloader_test: DataLoader,epoch):
    torch.cuda.empty_cache()
    # if (epoch + 1) % 1 == 0:
    p, r = test_accuracy(model, dataloader_test)
    log_info = 'precision: {:.2f}% / recall: {:.2f}%'.format(p, r)

    print(log_info)

    torch.cuda.empty_cache()
    return p,r
