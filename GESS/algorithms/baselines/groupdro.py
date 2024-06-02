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
class GroupDRO(BaseAlgo):
    r"""
    Original Paper:
    @inproceedings{sagawa2019distributionally,
      title={Distributionally Robust Neural Networks},
      author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
      booktitle={International Conference on Learning Representations},
      year={2019}
    }
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GroupDRO, self).__init__(config)
        self.exp_coeff = config.algo.coeff

    def setup_criterion(self,):
        self.criterion = self.config.metrics.name2criterion[self.metrics_name](reduction="none")

    def loss_postprocess(self, loss: List[Tensor], data: Batch):
        loss_list = []
        domain_ids = torch.unique(data.domain_id)
        for i in domain_ids:
            env_idx = data.domain_id == i
            if loss[env_idx].shape[0] > 0:
                loss_list.append(loss[env_idx].sum() / loss[env_idx].shape[0])
        losses = torch.stack(loss_list)
        group_weights = torch.ones(losses.shape[0], device=self.device)
        group_weights *= torch.exp(self.exp_coeff * losses.data)
        group_weights /= group_weights.sum()
        loss = losses @ group_weights
        return loss

    def forward_pass(self, data: Batch, epoch: Optional[int], phase: str):
        clf_logits = self.model(data=data)
        losses = self.criterion(clf_logits, data.y.float())  # NOTE loss is a tensor with shape [batch_size, 1]
        pred_loss = losses.mean()
        if phase != 'train':
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}, clf_logits
        dro_loss = self.loss_postprocess(losses, data)
        # 　for signal shift, we use dro_loss + pred_loss as the loss
        #   because there is no subgroup splits for positive samples in this case.
        return dro_loss, {'loss': dro_loss.item(), 'pred': pred_loss.item(), 'dro': dro_loss.item()}, clf_logits
