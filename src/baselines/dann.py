import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn as nn
from .erm import ERM
import torch.nn.functional as F
from typing import Optional, Any, Tuple
from torch.autograd import Function


class DANN(ERM):
    """
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
    def __init__(self, clf, DALoss, criterion, config, **kwargs):
        super(DANN, self).__init__(clf, criterion)
        self.coeff = config['coeff']
        self.domain_adv = DALoss

    def loss_postprocess(self, src_feats, trg_feats):
        return self.domain_adv(src_feats, trg_feats) * self.coeff

    def forward_pass(self, data, epoch, phase):

        if phase == 'train':
            assert len(data) == 2
            data_s, data_t = data

            # concat data_s and data_t
            x = torch.cat([data_s.x, data_t.x], dim=0)
            pos = torch.cat([data_s.pos, data_t.pos], dim=0)
            batch = torch.cat([data_s.batch, (data_t.batch + data_s.batch.max() + 1)])

            feats = self.clf.forward_passing(x, pos, batch)
            clf_logits = self.clf.clf_out(feats)

            # split data_s and data_t
            src_logits, _ = torch.chunk(clf_logits, 2)
            src_feats, trg_feats = torch.chunk(feats, 2)

            pred_loss, loss_dict = self.__loss__(src_logits, data_s.y)  # classification loss
            dann_loss = self.loss_postprocess(src_feats, trg_feats)  # DANN loss
            loss_dict['dann'] = dann_loss.item()
            loss_dict['disc_acc'] = self.domain_adv.domain_discriminator_accuracy
            loss = pred_loss + dann_loss
            return loss, loss_dict, src_logits
        else:
            feats = self.clf.forward_pass_(data)
            clf_logits = self.clf.clf_out(feats)
            pred_loss, loss_dict = self.__loss__(clf_logits, data.y)
            return pred_loss, loss_dict, clf_logits


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator, criterion, grl=None, max_iters=1000):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=max_iters,
                                                 auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.criterion = criterion
        self.domain_discriminator_accuracy = None

    def forward(self, f_s, f_t):
        f = self.grl(torch.cat((f_s, f_t), dim=0))  # torch.Size([256, 64])
        d = self.domain_discriminator(f)  # torch.Size([256, 1])

        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones(d_s.shape).to(f_s.device)
        d_label_t = torch.zeros(d_t.shape).to(f_t.device)
        d_label = torch.cat((d_label_s, d_label_t), dim=0)
        self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
        # if w_s is None:
        #     w_s = torch.ones_like(d_label_s)
        # if w_t is None:
        #     w_t = torch.ones_like(d_label_t)
        dann_loss = self.criterion(d, d_label)
        return dann_loss


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (torch.sigmoid(output) >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        # print(coeff)
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)
