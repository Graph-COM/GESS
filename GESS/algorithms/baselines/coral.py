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
class Coral(BaseAlgo):
    r"""
    Original Paper:
    @inproceedings{sun2016deep,
      title={Deep coral: Correlation alignment for deep domain adaptation},
      author={Sun, Baochen and Saenko, Kate},
      booktitle={Computer Vision--ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14},
      pages={443--450},
      year={2016},
      organization={Springer}
    }
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Coral, self).__init__(config)
        self.coeff = config.algo.coeff

    def feat_postprocess(self, src_feats: Tensor, trg_feats: Tensor):
        coral_loss_list = []

        src_cov_mat = self.compute_covariance(src_feats)
        trg_cov_mat = self.compute_covariance(trg_feats)

        dis = src_cov_mat - trg_cov_mat
        cov_loss = torch.mean(torch.mul(dis, dis)) / 4
        coral_loss_list.append(cov_loss)

        coral_loss = torch.tensor(0) if len(coral_loss_list) == 0 else torch.tensor(coral_loss_list).mean()
        coral_loss = coral_loss * self.coeff

        return coral_loss

    def forward_pass(self, data: Batch, epoch: Optional[int], phase: str):

        if phase == 'train':
            assert len(data) == 2
            data_s, data_t = data

            # concat data_s and data_t
            x = torch.cat([data_s.x, data_t.x], dim=0)
            pos = torch.cat([data_s.pos, data_t.pos], dim=0)
            batch = torch.cat([data_s.batch, (data_t.batch+data_s.batch.max()+1)])

            feats = self.model.geo_dat_repr(x=x, pos=pos, batch=batch)
            clf_logits = self.model.clf_out(feats)

            # split data_s and data_t
            src_logits, _ = torch.chunk(clf_logits, 2)
            src_feats, trg_feats = torch.chunk(feats, 2)
            pred_loss, loss_dict = self.__loss__(src_logits, data_s.y)

            coral_loss = self.feat_postprocess(src_feats, trg_feats)
            loss = pred_loss + coral_loss
            return loss, {'loss': loss.item(), 'pred': pred_loss.item(), 'coral': coral_loss.item()}, src_logits

        if phase != 'train':
            feats = self.model.geo_dat_repr(data=data)
            clf_logits = self.model.clf_out(feats)
            pred_loss, loss_dict = self.__loss__(clf_logits, data.y)
            return pred_loss, loss_dict, clf_logits

    def compute_covariance(self, feats: Tensor):
        n = feats.shape[0]
        all_ones = torch.ones((1, n)).to(self.device)
        tmp = all_ones @ feats
        covariance = (feats.t() @ feats - (tmp.t() @ tmp) / n) / (n - 1)
        return covariance
