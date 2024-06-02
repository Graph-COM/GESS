import torch
from torch_geometric.data import InMemoryDataset
from typing import Dict
from torch.utils.data import DataLoader
from GESS import register
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.algorithms.baselines.base_algo import BaseAlgo
from torch_geometric.loader import DataLoader
from GESS.utils.utils import send_to_device
from GESS import register


@register.dataloader_register
class BaseDataloader:
    r"""
    Dataloader Setup, which can be used to create dataloaders of No-Info, O-Feature, Par-Label levels in this project.
    You can inherit the `BaseDataloader` class for specific needs.
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        self.setting = config.dataset.setting
        self.train_bs = config.train.train_bs
        self.id_val_bs = config.train.id_val_bs
        self.id_test_bs = config.train.id_test_bs
        self.ood_val_bs = config.train.ood_val_bs
        self.ood_test_bs = config.train.ood_test_bs
        self.name2loader = {
            'No-Info': self.no_info_loader,
            'O-Feature': self.o_feature_loader,
            'Par-Label': self.par_label_loader
        }
    
    def setup(self, dataset: InMemoryDataset):
        return self.name2loader[self.setting](dataset)
    
    def no_info_loader(self, dataset: InMemoryDataset):
        r"""
        Setup dataloader specifically used in No-Info level.
        """
        idx_split = dataset.idx_split
        loader = {
        'train': DataLoader(dataset[idx_split['train']], batch_size=self.train_bs, shuffle=True, follow_batch=None, drop_last=False),
        'iid_val': DataLoader(dataset[idx_split['iid_val']], batch_size=self.id_val_bs, shuffle=False, follow_batch=None, drop_last=False),
        'iid_test': DataLoader(dataset[idx_split['iid_test']], batch_size=self.id_test_bs, shuffle=False, follow_batch=None, drop_last=False),
        'ood_val': DataLoader(dataset[idx_split['ood_val']], batch_size=self.ood_val_bs, shuffle=False, follow_batch=None, drop_last=False),
        'ood_test': DataLoader(dataset[idx_split['ood_test']], batch_size=self.ood_test_bs, shuffle=False, follow_batch=None, drop_last=False)
        }
        return loader
    
    def o_feature_loader(self, dataset: InMemoryDataset):
        r"""
        Setup dataloader specifically used in O-Feature level.
        """
        idx_split = dataset.idx_split
        loader = {
        'train_source': DataLoader(dataset[idx_split['train_source']], batch_size=self.train_bs, shuffle=True, follow_batch=None, drop_last=True),
        'train_target': DataLoader(dataset[idx_split['train_target']], batch_size=self.train_bs, shuffle=True, follow_batch=None, drop_last=True),        
        'iid_val': DataLoader(dataset[idx_split['iid_val']], batch_size=self.id_val_bs, shuffle=False, follow_batch=None, drop_last=False),
        'iid_test': DataLoader(dataset[idx_split['iid_test']], batch_size=self.id_test_bs, shuffle=False, follow_batch=None, drop_last=False),
        'ood_val': DataLoader(dataset[idx_split['ood_val']], batch_size=self.ood_val_bs, shuffle=False, follow_batch=None, drop_last=False),
        'ood_test': DataLoader(dataset[idx_split['ood_test']], batch_size=self.ood_test_bs, shuffle=False, follow_batch=None, drop_last=False)
        }
        loader['train_source'] = ForeverDataIterator(loader['train_source'])
        loader['train_target'] = ForeverDataIterator(loader['train_target'])       
        
        return loader
    
    def par_label_loader(self, dataset: InMemoryDataset):
        r"""
        Setup dataloader specifically used in Par-Label level.
        """
        idx_split = dataset.idx_split
        loader = {
        'train': DataLoader(dataset[idx_split['train']], batch_size=self.train_bs, shuffle=True, follow_batch=None, drop_last=False),
        'val': DataLoader(dataset[idx_split['val']], batch_size=self.id_val_bs, shuffle=False, follow_batch=None, drop_last=False),
        'test': DataLoader(dataset[idx_split['test']], batch_size=self.id_test_bs, shuffle=False, follow_batch=None, drop_last=False),
        }
        return loader
        

class ForeverDataIterator:
    r"""
    A data iterator that will never stop producing data
    """
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