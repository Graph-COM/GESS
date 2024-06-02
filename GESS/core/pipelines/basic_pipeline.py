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
class BasePipeline:
    """
    start a basic pipeline: run for predefined epochs, log metric scores during the pipeline, and obtain final results.
    This is used in No-Info level in this project.
    You can inherit the `BasePipeline` class for specific needs.
    """
    def __init__(self, config: Union[CommonArgs, Munch], baseline: BaseAlgo, loaders: Union[DataLoader, Dict[str, DataLoader]]):
        self.config = config
        self.baseline = baseline
        self.loaders = loaders
        self.metrics = config.metrics
        self.epoch = 0
        self.train_res, self.valid_res, self.test_res, self.ood_valid_res, self.ood_test_res = None, None, None, None, None

    def start_pipeline(self):
        r"""
        run for predefined epochs, log losses and ID/OOD metric scores during the pipeline, and obtain final results.
        """
        for epoch in range(self.config.train.epochs):
            self.epoch = epoch
            self.train_res = self.train_one_epoch()
            logger(f'epoch {epoch} ' + f"{' '.join([f'{k} {v}' for k, v in self.train_res[1].items()])}",
                  log_file=self.config.path.loss_file)
            self.valid_res = self.eval_one_epoch('iid_val')
            self.test_res = self.eval_one_epoch('iid_test')
            self.metrics.update_id_metrics(self.train_res, self.valid_res, self.test_res, epoch, self.config, self.baseline.model)

            self.ood_valid_res = self.eval_one_epoch('ood_val')
            self.ood_test_res = self.eval_one_epoch('ood_test')
            self.metrics.update_ood_metrics(self.train_res, self.ood_valid_res, self.ood_test_res, epoch, self.config, self.baseline.model)
        results_logger(self.config.path.result_path, self.config.seed, self.metrics.metrics_id)
        results_logger(self.config.path.result_ood_path, self.config.seed, self.metrics.metrics_ood)

    def train_one_epoch(self):
        r"""
        Model training in one epoch.
        """
        results = self.run_one_epoch("train")
        return results

    def eval_one_epoch(self, phase: str):
        """
        Evaluate the metrics in one epoch.
        """
        results = self.run_one_epoch(phase)
        return results

    def run_one_epoch(self, phase: str):
        r"""
        Train or evaluate data in one epoch.
        Returns:
            - metric_score: evaluation metrics;
            - all_loss_dict: A dictionary with losses (average in one epoch, including prediction loss and other losses specific to an algorithm) and other auxiliary information;
            - all_loss_dict['pred']: average prediction loss in one epoch.
        """
        data_loader = self.loaders[phase]
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        log_dict = {'model_out': [], 'labels': []}
        all_loss_dict = {}
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            loss_dict, model_out = run_one_batch(data.to(self.config.device))
            labels = to_cpu(data.y)
            for key in log_dict.keys():
                log_dict[key].append(eval(key))

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            if idx == loader_len - 1:
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                model_out = torch.cat(log_dict['model_out'])
                labels = torch.cat(log_dict['labels'])
                metric_score = self.metrics.cal_metrics_score(labels, model_out)
        return metric_score, all_loss_dict, all_loss_dict['pred']      
    
    def eval_one_batch(self, data: Batch):
        """
        Evaluate the metrics in one batch.
        """
        self.baseline.model.eval()
        _, loss_dict, org_clf_logits = self.baseline.forward_pass(data, self.epoch, "eval")
        return loss_dict, to_cpu(org_clf_logits)

    def train_one_batch(self, data: Batch):
        """
        Model training in one batch.
        """
        self.baseline.model.train()
        loss, loss_dict, org_clf_logits = self.baseline.forward_pass(data, self.epoch, "train")
        self.baseline.loss_backward(loss)
        return loss_dict, to_cpu(org_clf_logits)


