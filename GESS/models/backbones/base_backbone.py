from torch import Tensor
from typing import Callable, Union, Dict
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn.conv import MessagePassing
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.models.models.tools import FeatEncoder, MLP


class BaseGDLEncoder(torch.nn.Module):
    """
    The base GDL encoder. The three GDL encoders inherit the `BaseGDLEncoder` class.
    If you would like to add more cutting-edge GDL encoder design, all you need is to specify your own GDL layer and add it to `self.convs`.
    (refer to GESS/models/backbones/egnn.py for example)
    Please ensure that the forward method of your custom GDL layer follows this order of parameters: `x`, `edge_index`, `pos`, `batch`, `edge_attr`, `edge_attn`.
    """
    def __init__(self, x_dim: int, pos_dim: int, model_config: Union[CommonArgs, Munch], 
                 feat_info: Dict[str, int], n_categorical_feat_to_use: int = -1, n_scalar_feat_to_use: int = -1, **kwargs):
        super().__init__()

        self.hidden_size = model_config.hidden_size
        self.n_layers = model_config.n_layers

        self.x_dim = x_dim
        self.pos_dim = pos_dim

        self.dropout_p = model_config.dropout_p
        self.norm_type = model_config.norm_type
        self.act_type = model_config.act_type

        self.node_encoder = FeatEncoder(self.hidden_size, feat_info['node_categorical_feat'], feat_info['node_scalar_feat'], n_categorical_feat_to_use, n_scalar_feat_to_use)
        self.edge_encoder = FeatEncoder(self.hidden_size, feat_info['edge_categorical_feat'], feat_info['edge_scalar_feat'])

        self.convs = torch.nn.ModuleList()

    def forward(self, x: Tensor, pos: Tensor, edge_attr: Tensor, edge_index: Tensor, batch: Tensor, 
                edge_attn: Tensor = None, node_attn: Tensor = None, with_enc: bool =True):
        if with_enc:
            if self.x_dim == 0 and self.pos_dim != 0:
                feats = pos
            elif self.x_dim != 0 and self.pos_dim == 0:
                feats = x
            elif self.x_dim == 0 and self.pos_dim == 0:
                feats = torch.ones(x.shape[0], 1, device=x.device)
            else:
                feats = torch.cat([x, pos], dim=1)
            x = self.node_encoder(feats)
        edge_attr = self.edge_encoder(edge_attr)
        if edge_attn is not None and edge_attn.dim() == 1:
            edge_attn = edge_attn.unsqueeze(1)
        for i in range(self.n_layers):
            identity = x
            x = self.convs[i](x, edge_index, pos, batch=batch, edge_attr=edge_attr, edge_attn=edge_attn)
            x = x + identity
            x = F.dropout(x, self.dropout_p, training=self.training)
        return x