from .basic_pipeline import BasePipeline
import torch
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from GESS.utils import logger, results_logger, to_cpu
from GESS import register
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.algorithms.baselines.base_algo import BaseAlgo


@register.pipeline_register
class O_Feature_Pipeline(BasePipeline):
    r"""
    A pipeline version for O-Feature setting.
    The only difference is the training process. Others keep the same.
    """
    def __init__(self, config: Union[CommonArgs, Munch], baseline: BaseAlgo, loaders: Union[DataLoader, Dict[str, DataLoader]]):
        super().__init__(config, baseline, loaders)
        self.iters_per_epoch = config.train.iters_per_epoch

    def train_one_epoch(self, ):
        """
        Model training in one epoch specific to the O-Feature setting.
        """
        all_loss_dict = {}
        log_dict = {'model_out': [], 'labels': []}
        train_source_iter, train_target_iter = self.loaders['train_source'], self.loaders['train_target']
        pbar = tqdm(range(self.iters_per_epoch))
        for idx in pbar:
            data_s = next(train_source_iter).to(self.config.device)
            data_t = next(train_target_iter).to(self.config.device)
            loss_dict, model_out = self.train_one_batch((data_s, data_t))
            if len(data_s.y.shape) == 1:
                data_s.y = data_s.y.unsqueeze(1)
            labels = to_cpu(data_s.y)
            for key in log_dict.keys():
                log_dict[key].append(eval(key))

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            if idx == self.iters_per_epoch - 1:
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / self.iters_per_epoch
                model_out = torch.cat(log_dict['model_out'])
                labels = torch.cat(log_dict['labels'])
                metric_score = self.metrics.cal_metrics_score(labels, model_out)
        return metric_score, all_loss_dict, all_loss_dict['pred']