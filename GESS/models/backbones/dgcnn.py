from torch import Tensor
from typing import Callable, Union, Dict
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor
from .base_backbone import BaseGDLEncoder
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn.conv import MessagePassing
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.models.models.tools import FeatEncoder, MLP
from GESS import register


@register.gdlbackbone_register
class DGCNN(BaseGDLEncoder):
    r"""
    DGCNN. Original paper:
    @article{wang2019dynamic,
    title={Dynamic graph cnn for learning on point clouds},
    author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E and Bronstein, Michael M and Solomon, Justin M},
    journal={ACM Transactions on Graphics (tog)},
    volume={38},
    number={5},
    pages={1--12},
    year={2019},
    publisher={Acm New York, NY, USA}
    }
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_classification.py
    """
    def __init__(self, x_dim: int, pos_dim: int, model_config: Union[CommonArgs, Munch], 
                 feat_info: Dict[str, int], n_categorical_feat_to_use: int = -1, n_scalar_feat_to_use: int = -1, **kwargs):
        super(DGCNN, self).__init__(x_dim, pos_dim, model_config, feat_info, n_categorical_feat_to_use, n_scalar_feat_to_use, **kwargs)

        for _ in range(self.n_layers):
            mlp = MLP([self.hidden_size*3, self.hidden_size*2, self.hidden_size], 0.0, self.norm_type, self.act_type)
            self.convs.append(EdgeConv(mlp, self.hidden_size, self.norm_type, self.act_type, aggr='mean'))


class EdgeConv(MessagePassing):
    def __init__(self, nn: Callable, hidden_size, norm_type, act_type, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        self.nn = nn
        self.post_nn = Linear(hidden_size, hidden_size)
        self.act_fn = MLP.get_act(act_type)()
        self.norm = MLP.get_norm(norm_type)(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor], edge_index, pos, 
            batch, edge_attr=None, edge_attn=None) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")
        out = self.propagate(edge_index, x=x, size=None, edge_attr=edge_attr, edge_attn=edge_attn)
        out = self.post_nn(out)
        out = self.act_fn(self.norm(out))
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr, edge_attn) -> Tensor:
        msg = self.nn(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))
        if edge_attn is not None:
            return msg * edge_attn
        else:
            return msg

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'
