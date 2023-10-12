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

init_metric_dict_ood_auc = {'metric/best_clf_epoch': 0,
                            'metric/clf_train_loss': 0,
                            'metric/best_clf_valid_loss': 0,
                            'metric/best_clf_train_auc': 0,
                            'metric/best_clf_valid_auc': 0,
                            'metric/best_clf_test_auc': 0,
                            }

init_metric_dict_ood_acc = {'metric/best_clf_epoch': 0,
                            'metric/clf_train_loss': 0,
                            'metric/best_clf_valid_loss': 0,
                            'metric/best_clf_train_acc': 0,
                            'metric/best_clf_valid_acc': 0,
                            'metric/best_clf_test_acc': 0,
                            }

init_metric_dict_ood_mae = {'metric/best_regrs_epoch': 0,
                            'metric/regrs_train_loss': 0,
                            'metric/best_regrs_valid_loss': 0,
                            'metric/best_regrs_train_mae': 0,
                            'metric/best_regrs_valid_mae': 1e6,
                            'metric/best_regrs_test_mae': 0,
                            }


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


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


class DiscriminationCollater():
    def __init__(self, follow_batch=[], exclude_keys=[]):
        self.collater = Collater(follow_batch, exclude_keys)
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, data_list):

        if isinstance(data_list[0], Data):
            return self.collater(data_list)
        elif len(data_list[0]) == 2:
            data_orig_list, corrupt_data_list = zip(*data_list)
        else:
            raise NotImplementedError
        mix_list = list(data_orig_list)
        mix_list.extend(corrupt_data_list)

        if len(data_list[0]) == 2:
            batch = self.collater(mix_list)
            return batch


def set_seed(seed):
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


def load_model(seed, clf, log_dir, mode='auc'):
    file_name = f'model_{mode}_{seed}.pt'
    file_path = log_dir / file_name
    model_load = torch.load(file_path, map_location='cpu')
    model_dict = dict()
    for item in model_load['model_state_dict'].keys():
        item1 = item.replace("clf.", '')
        model_dict[item1] = model_load['model_state_dict'][item]
    clf.load_state_dict(model_dict)
    return clf

