from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from ..ood_dataset import HEP_Pileup_Shift, HEP_Signal_Shift, QMOF, Drug3d
from ..utils import utils


def get_data_loaders(dataset_name, config, shift_config, seed):
    root_dir = (Path(config['dir_config']['dataset_dir']) / f'{dataset_name}').as_posix()
    batch_size = config['optimizer']['batch_size']
    data_config = config['data']
    shift_name = shift_config['shift_name']
    setting = data_config['setting']
    if dataset_name == 'Track':
        assert shift_name in ["pileup", "signal"]
        Dataset = HEP_Pileup_Shift if shift_name == 'pileup' else HEP_Signal_Shift
    elif dataset_name == 'QMOF':
        assert shift_name == 'fidelity'
        Dataset = QMOF
    elif dataset_name == 'DrugOOD-3D':
        assert shift_name in ["size", "scaffold", "assay"]
        Dataset = Drug3d
    else:
        raise NotImplementedError
    dataset = Dataset(root_dir, data_config, shift_config, seed)

    def process(dataset):
        if dataset.dataset_name == "DrugOOD-3D":
            dataset.data.y = dataset.data.y.view(-1, 1)
            dataset.data.x[dataset.data.x == -1] = 13
        if dataset.dataset_name == "QMOF":
            dataset.data.y = dataset.data.y.view(-1, 1)
        return dataset

    dataset = process(dataset)  # necessary for DrugOOD-3D
    loaders = get_ood_data_loader(batch_size, dataset=dataset, idx_split=dataset.idx_split, setting=setting)

    if data_config['setting'] == "O-Feature":
        loaders['train_source'] = utils.ForeverDataIterator(loaders['train_source'])
        loaders['train_target'] = utils.ForeverDataIterator(loaders['train_target'])
    return loaders, dataset


def get_ood_data_loader(batch_size, dataset, idx_split, setting):
    data_loader = dict()
    for item in idx_split.keys():
        shuffling = True if item.split('_')[0] == 'train' else False
        drop_last = True if (item.split('_')[0] == 'train' and setting == "O-Feature") else False
        batch_size = 32 if item in ['ood_val', 'ood_test'] and dataset.dataset_name == 'Track' else batch_size
        loader = DataLoader(dataset[idx_split[item]], batch_size=batch_size, shuffle=shuffling, follow_batch=None,
                            drop_last=drop_last)
        data_loader[item] = loader
    return data_loader
