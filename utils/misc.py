import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor
import yaml
from torch.utils.tensorboard import SummaryWriter
import wandb

def make_wandb(model_dir, cfg):
    if is_main_process():
        wandb.login(key='provide your key here')
        run = wandb.init(project='TwoStreamSLT', config=cfg, reinit=True)
        wandb.run.name = '/'.join(model_dir.split('/')[-2:])
        wandb.run.save()
        return run
    else:
        return None

def neq_load_customized(model, pretrained_dict, verbose=False):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        # print("print info misc.py line36 ",list(model_dict.keys()))
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape==v.shape:
            tmp[k] = v
        else:
            if verbose:
                continue
                # print(k)
    if verbose:
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
            elif model_dict[k].shape != pretrained_dict[k].shape:
                print(k, 'shape mis-matched, not loaded')
        print('===================================\n')

    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model



def move_to_device(batch, device):
    for k, v in batch.items():
        if type(v)==dict:
            batch[k] = move_to_device(v, device)
        elif type(v)==torch.Tensor:
            batch[k] = v.to(device)
        elif type(v)==list and type(v[0])==torch.Tensor:
            batch[k] = [e.to(device) for e in v]
    return batch

def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.
    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if is_main_process():
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        elif overwrite:
            shutil.rmtree(model_dir)
            os.makedirs(model_dir)
    synchronize()
    return model_dir

def get_logger():
    return logger
    
def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.
    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    global logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux":
            sh = logging.StreamHandler()
            if not is_main_process():
                sh.setLevel(logging.ERROR)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        return logger

def make_writer(model_dir):
    if is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(model_dir + "/tensorboard/"))
    else:
        writer = None
    return writer

def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.
    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))

def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.
    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)


    if 'RecognitionNetwork' in cfg['model']:
        if 'keypoint' in cfg['Dataprocessing'].get('input_streams', ['rgb']):
            assert 'keypoint_s3d' in cfg['model']['RecognitionNetwork']
            from dataset.Dataset import get_keypoints_num
            keypoints_num = get_keypoints_num(
                keypoint_file=cfg['Dataprocessing']['keypoint_file'], use_keypoints=cfg['Dataprocessing']['use_keypoints'])
            if 'keypoint_s3d' in cfg['model']['RecognitionNetwork']:
                cfg['model']['RecognitionNetwork']['keypoint_s3d']['in_channel'] = keypoints_num
                print(f'Overwrite cfg.model.RecognitionNetwork.keypoint_s3d.in_channel -> {keypoints_num}')
    return cfg

def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None
    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(path: str, map_location: str='cpu') -> dict:
    """
    Load model from saved checkpoint.
    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint

def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training
    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    os.system('cp {} {}'.format(target, link_name))
    # try:
    #     os.symlink(target, link_name)
    # except FileExistsError as e:
    #     if e.errno == errno.EEXIST:
    #         os.remove(link_name)
    #         os.symlink(target, link_name)
    #     else:
    #         raise e

def is_main_process():
    return 'WORLD_SIZE' not in os.environ or os.environ['WORLD_SIZE']=='1' or os.environ['LOCAL_RANK']=='0'



def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def init_DDP():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{}'.format(local_rank))
    torch.distributed.init_process_group(backend="nccl", init_method="env://")    
    return local_rank, int(os.environ['WORLD_SIZE']), device

def synchronize():
    torch.distributed.barrier()

def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training
    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import os
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# def is_main_process():
#     return get_rank() == 0

def is_main_process():
    return 'WORLD_SIZE' not in os.environ or os.environ['WORLD_SIZE']=='1' or os.environ['LOCAL_RANK']=='0'


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)