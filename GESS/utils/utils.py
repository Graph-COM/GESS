r"""
Here consists of other tools used in this project.
"""
import os
import sys
import random
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
from joblib import Parallel, delayed
import pandas
import torch
import numpy as np
import time
from .logger import log


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def set_seed(seed):
    r"""
    Set seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_cpu(tensor):
    return tensor.detach().cpu() if tensor is not None else None


def get_random_idx_split(dataset_len, split, seed, restrict_training=0):
    np.random.seed(seed)

    log('[INFO] Randomly split dataset!')
    idx = np.arange(dataset_len)
    np.random.shuffle(idx)

    n_train, n_valid = int(split['train'] * len(idx)), int(split['val'] * len(idx))
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]
    if restrict_training > 0:
        train_idx = train_idx[:restrict_training]
    return {'train': train_idx, 'val': valid_idx, 'test': test_idx}


def get_ood_split(dataset_len, split, seed):
    np.random.seed(seed)

    log('[INFO] Randomly split dataset!')
    idx = np.arange(dataset_len)
    np.random.shuffle(idx)

    n_train, n_iid_val, n_iid_test, n_ood_val = int(split['train'] * len(idx)), int(split['iid_val'] * len(idx)), int(
        split['iid_test'] * len(idx)), int(split['ood_val'] * len(idx))
    train_idx = idx[:n_train]
    iid_val_idx = idx[n_train:n_train + n_iid_val]
    iid_test_idx = idx[n_train + n_iid_val:n_train + n_iid_val + n_iid_test]
    ood_val_idx = idx[n_train + n_iid_val + n_iid_test:n_train + n_iid_val + n_iid_test + n_ood_val]
    ood_test_idx = idx[n_train + n_iid_val + n_iid_test + n_ood_val:]
    return {'train': train_idx, 'iid_val': iid_val_idx, 'iid_test': iid_test_idx, 'ood_val': ood_val_idx,
            'ood_test': ood_test_idx}


