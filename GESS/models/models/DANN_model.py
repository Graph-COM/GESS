from .base_model import BaseModel
from GESS.models.models.tools import CoorsNorm, MLP
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Any, Tuple
from torch.autograd import Function
from GESS import register
from GESS.utils.config_process import Union, CommonArgs, Munch
from torch_geometric.data import InMemoryDataset


@register.model_register
class DANNModel(BaseModel):
    r"""
    DANN model. 
    New modules in this algorithm: 
        - Gradient Reverse Layer (self.grl)
        - Domain Discriminator (self.domain_discriminator)
    """
    def __init__(self, config: Union[CommonArgs, Munch], dataset: InMemoryDataset, gdlencoder: torch.nn.Module):
        super().__init__(config, dataset, gdlencoder)
        hidden_size = config.algo.extra.disc_hidden
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=config.algo.extra.max_iters,
                                                 auto_step=True)
        self.domain_discriminator = MLP([self.hidden_size, hidden_size, hidden_size, self.out_dim], 
                                        self.dropout_p, self.norm_type, self.act_type)


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
