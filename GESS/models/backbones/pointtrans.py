from torch import Tensor
from typing import Callable, Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor, Dict
from GESS.utils.config_process import Union, CommonArgs, Munch
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from .base_backbone import BaseGDLEncoder
from torch_geometric.nn.inits import reset
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from GESS.models.models.tools import FeatEncoder, MLP
from GESS import register


@register.gdlbackbone_register
class PointTransformer(BaseGDLEncoder):
    r"""
    Point Transformer. Original paper:
    @inproceedings{zhao2021point,
    title={Point transformer},
    author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
    booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
    pages={16259--16268},
    year={2021}
    }
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py
    """
    def __init__(self, x_dim: int, pos_dim: int, model_config: Union[CommonArgs, Munch], 
                 feat_info: Dict[str, int], n_categorical_feat_to_use: int = -1, n_scalar_feat_to_use: int = -1, **kwargs):
        super(PointTransformer, self).__init__(x_dim, pos_dim, model_config, feat_info, n_categorical_feat_to_use, n_scalar_feat_to_use, **kwargs)

        self.raw_pos_dim = kwargs['aux_info']['raw_pos_dim']
        for _ in range(self.n_layers):
            self.convs.append(TransformerBlock(self.hidden_size, self.hidden_size, pos_dim=self.raw_pos_dim))



class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pos_dim):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)
        self.transformer = PointTransformerConv(in_channels, out_channels, pos_dim=pos_dim)

    def forward(self, x, edge_index, pos, batch=None, edge_attr=None, edge_attn=None):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index, edge_attr=edge_attr, edge_attn=edge_attn)
        x = self.lin_out(x).relu()
        return x


class PointTransformerConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, pos_nn: Optional[Callable] = None,
                 attn_nn: Optional[Callable] = None,
                 add_self_loops: bool = False, pos_dim=3, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.pos_nn = pos_nn
        if self.pos_nn is None:
            self.pos_nn = Linear(pos_dim, out_channels)

        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels, bias=False)
        self.lin_src = Linear(in_channels[0], out_channels, bias=False)
        self.lin_dst = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pos_nn)
        if self.attn_nn is not None:
            reset(self.attn_nn)
        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj, edge_attr=None, edge_attn=None
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x: PairTensor = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)


        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha, size=None, edge_attr=edge_attr, edge_attn=edge_attn)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int], edge_attr=None, edge_attn=None) -> Tensor:

        delta = self.pos_nn(pos_i - pos_j)
        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index, ptr, size_i)

        if edge_attr is not None:
            msg = alpha * (x_j + delta + edge_attr)
        else:
            msg = alpha * (x_j + delta)

        if edge_attn is not None:
            msg = msg * edge_attn
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
