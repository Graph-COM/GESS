import torch
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from GESS.utils import logger, results_logger, to_cpu, load_checkpoint
import os
from GESS import register
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.algorithms.baselines.base_algo import BaseAlgo
from .basic_pipeline import BasePipeline


@register.pipeline_register
class Par_Label_Pipeline(BasePipeline):
    r"""
    A pipeline version for Par-Label setting.
    Load pretrained checkpoint of No-Info ERM is needed in this setting.
    Note: The original in-distribution val and test data are not evaluated in this setting. 
    """
    def __init__(self, config: Union[CommonArgs, Munch], baseline: BaseAlgo, loaders: Union[DataLoader, Dict[str, DataLoader]]):
        super().__init__(config, baseline, loaders)
        assert os.path.exists(config.path.load_pretrain_ckpt), \
            "A pretrained model of No-Info ERM is needed in the Par-Label level.\n Please firstly run `gess-run --config_path [dataset]/[shift]/[target]/No-Info/ERM.yaml --gdl [GDL]`,\n where [dataset], [shift], [target], [GDL] are args you specify here."
        load_checkpoint(baseline.model, config.path.load_pretrain_ckpt)
    
    def start_pipeline(self):
        r"""
        Note: Here 'val' and 'test' refer to OOD val and test data. 
        The original in-distribution val and test data are not evaluated in this setting. 
        """
        for epoch in range(self.config.train.epochs):
            self.train_res = self.train_one_epoch()
            logger(f'epoch {epoch} ' + f"{' '.join([f'{k} {v}' for k, v in self.train_res[1].items()])}",
                  log_file=self.config.path.loss_file)
            self.valid_res = self.eval_one_epoch('val')
            self.test_res = self.eval_one_epoch('test')
            self.metrics.update_id_metrics(self.train_res, self.valid_res, self.test_res, epoch, self.config, self.baseline.model)
        results_logger(self.config.path.result_path, self.config.seed, self.metrics.metrics_id)
