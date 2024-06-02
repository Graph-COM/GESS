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
class DANN(BaseAlgo):
    r"""
    Original Paper:
    @article{ganin2016domain,
      title={Domain-adversarial training of neural networks},
      author={Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Fran{\c{c}}ois and Marchand, Mario and Lempitsky, Victor},
      journal={The journal of machine learning research},
      volume={17},
      number={1},
      pages={2096--2030},
      year={2016},
      publisher={JMLR. org}
    }
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DANN, self).__init__(config)
        self.coeff = config.algo.coeff
        self.domain_discriminator_accuracy = None

    def feat_postprocess(self, src_feats: Tensor, trg_feats: Tensor):
        return self.domain_adv(src_feats, trg_feats) * self.coeff

    def domain_adv(self, f_s: Tensor, f_t: Tensor):
        f = self.model.grl(torch.cat((f_s, f_t), dim=0))  # torch.Size([256, 64])
        d = self.model.domain_discriminator(f)  # torch.Size([256, 1])

        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones(d_s.shape).to(f_s.device)
        d_label_t = torch.zeros(d_t.shape).to(f_t.device)
        d_label = torch.cat((d_label_s, d_label_t), dim=0)
        self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
        dann_loss = self.criterion(d, d_label)
        return dann_loss
    
    def forward_pass(self, data: Batch, epoch: Optional[int], phase: str):

        if phase == 'train':
            assert len(data) == 2
            data_s, data_t = data

            # concat data_s and data_t
            x = torch.cat([data_s.x, data_t.x], dim=0)
            pos = torch.cat([data_s.pos, data_t.pos], dim=0)
            batch = torch.cat([data_s.batch, (data_t.batch + data_s.batch.max() + 1)])

            feats = self.model.geo_dat_repr(x=x, pos=pos, batch=batch)
            clf_logits = self.model.clf_out(feats)

            # split data_s and data_t
            src_logits, _ = torch.chunk(clf_logits, 2)
            src_feats, trg_feats = torch.chunk(feats, 2)

            pred_loss, loss_dict = self.__loss__(src_logits, data_s.y)  # classification loss
            dann_loss = self.feat_postprocess(src_feats, trg_feats)  # DANN loss
            loss_dict['dann'] = dann_loss.item()
            loss_dict['disc_acc'] = self.domain_discriminator_accuracy
            loss = pred_loss + dann_loss
            return loss, loss_dict, src_logits
        else:
            feats = self.model.geo_dat_repr(data=data)
            clf_logits = self.model.clf_out(feats)
            pred_loss, loss_dict = self.__loss__(clf_logits, data.y)
            return pred_loss, loss_dict, clf_logits


def binary_accuracy(output: Tensor, target: Tensor):
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (torch.sigmoid(output) >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


