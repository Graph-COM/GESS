import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn as nn
from .erm import ERM


class GroupDRO(ERM):
    """
    Original Paper:
    @inproceedings{sagawa2019distributionally,
      title={Distributionally Robust Neural Networks},
      author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
      booktitle={International Conference on Learning Representations},
      year={2019}
    }
    """
    def __init__(self, clf, criterion, config):
        super(GroupDRO, self).__init__(clf, criterion)
        self.exp_coeff = config['coeff']

    def loss_postprocess(self, loss, data):
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

    def forward_pass(self, data, epoch, phase):
        clf_logits = self.clf(data)
        losses = self.criterion(clf_logits, data.y.float())  # NOTE loss is a tensor with shape [batch_size, 1]
        pred_loss = losses.mean()
        if phase != 'train':
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}, clf_logits
        dro_loss = self.loss_postprocess(losses, data)
        return dro_loss, {'loss': dro_loss.item(), 'pred': pred_loss.item(), 'dro': dro_loss.item()}, clf_logits
