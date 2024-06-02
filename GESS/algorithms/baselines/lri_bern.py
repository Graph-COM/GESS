import torch
from typing import Optional, List
from abc import ABC
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.models.models.base_model import BaseModel
from torch_geometric.data import Batch
from torch import Tensor
from .base_algo import BaseAlgo
from GESS import register


@register.algorithm_register
class LRIBern(BaseAlgo):
    r"""
    Original Paper:
    @article{miao2022interpretable,
    title={Interpretable geometric deep learning via learnable randomness injection},
    author={Miao, Siqi and Luo, Yunan and Liu, Mia and Li, Pan},
    journal={arXiv preprint arXiv:2210.16966},
    year={2022}
    }
    https://github.com/Graph-COM/LRI
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(LRIBern, self).__init__(config)
        self.info_loss_coef = config.algo.coeff
        self.temperature = config.algo.extra.temperature
        self.decay_interval = config.algo.extra.decay_interval
        self.decay_r = config.algo.extra.decay_r
        self.init_r = config.algo.extra.init_r
        self.final_r = config.algo.extra.final_r
        self.attn_constraint = config.algo.extra.attn_constraint

    def __loss__(self, attn: Tensor, clf_logits: Tensor, clf_labels: Tensor, epoch: int):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        r = self.get_r(epoch)
        info_loss = (attn * torch.log(attn/r + 1e-6) + (1 - attn) * torch.log((1 - attn)/(1 - r + 1e-6) + 1e-6)).mean()
        info_loss = self.info_loss_coef * info_loss
        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item(), 'r': r}
        return loss, loss_dict

    def forward_pass(self, data: Batch, epoch: int, phase: Optional[str]):
        emb, edge_index = self.model.get_emb(data)
        node_attn_log_logits = self.model.extractor(emb)
        node_attn = self.sampling(node_attn_log_logits)
        edge_attn = self.node_attn_to_edge_attn(node_attn, edge_index)
        masked_clf_logits = self.model(data=data, edge_attn=edge_attn)

        loss, loss_dict = self.__loss__(node_attn_log_logits.sigmoid(), masked_clf_logits, data.y, epoch)
        return loss, loss_dict, masked_clf_logits

    def get_r(self, current_epoch: int):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r
    
    def sampling(self, attn_log_logits: Tensor, do_sampling=True):
        if do_sampling:
            random_noise = torch.empty_like(attn_log_logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            attn_bern = ((attn_log_logits + random_noise) / self.temperature).sigmoid()
        else:
            attn_bern = (attn_log_logits).sigmoid()
        return attn_bern

    @staticmethod
    def node_attn_to_edge_attn(node_attn: Tensor, edge_index: Tensor):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        edge_attn = src_attn * dst_attn
        return edge_attn
