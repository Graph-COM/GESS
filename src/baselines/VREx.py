import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn as nn
from .erm import ERM


class VREx(ERM):
    """
    Original Ppaer:
    @inproceedings{krueger2021out,
      title={Out-of-distribution generalization via risk extrapolation (rex)},
      author={Krueger, David and Caballero, Ethan and Jacobsen, Joern-Henrik and Zhang, Amy and Binas, Jonathan and Zhang, Dinghuai and Le Priol, Remi and Courville, Aaron},
      booktitle={International Conference on Machine Learning},
      pages={5815--5826},
      year={2021},
      organization={PMLR}
    }
    """
    def __init__(self, clf, criterion, config):
        super(VREx, self).__init__(clf, criterion)
        self.exp_coeff = config['coeff']

    def loss_postprocess(self, loss, data):
        loss_list = []
        domain_ids = torch.unique(data.domain_id)
        for i in domain_ids:
            env_idx = data.domain_id == i
            if loss[env_idx].shape[0] > 0:
                loss_list.append(loss[env_idx].sum() / loss[env_idx].shape[0])
        losses = torch.stack(loss_list)
        var_loss = self.exp_coeff * torch.var(losses)
        if torch.isnan(var_loss):
            var_loss = 0
        return var_loss

    def forward_pass(self, data, epoch, phase):
        clf_logits = self.clf(data)
        losses = self.criterion(clf_logits, data.y.float())  # NOTE loss is a tensor with shape [batch_size, 1]
        pred_loss = losses.mean()
        if phase != 'train':
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}, clf_logits
        var_loss = self.loss_postprocess(losses, data)
        loss = pred_loss + var_loss
        return loss, {'loss': loss.item(), 'pred': pred_loss.item(), 'var': var_loss.item()}, clf_logits
