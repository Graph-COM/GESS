r"""
Here consists of necessary modules used in GESS/models/backbones and GESS/models/models.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm


class MLP(nn.Sequential):
    r"""
    Multilayer Perceptron.
    """
    def __init__(self, channels, dropout_p, norm_type, act_type):
        norm = self.get_norm(norm_type)
        act = self.get_act(act_type)

        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i]))

            if i < len(channels) - 1:
                m.append(norm(channels[i]))
                m.append(act())
                m.append(nn.Dropout(dropout_p))

        super(MLP, self).__init__(*m)

    @staticmethod
    def get_norm(norm_type):
        if isinstance(norm_type, str) and 'batch' in norm_type:
            return BatchNorm
        elif norm_type == 'none' or norm_type is None:
            return nn.Identity
        else:
            raise ValueError('Invalid normalization type: {}'.format(norm_type))

    @staticmethod
    def get_act(act_type):
        if act_type == 'relu':
            return nn.ReLU
        elif act_type == 'silu':
            return nn.SiLU
        elif act_type == 'none':
            return nn.Identity
        else:
            raise ValueError('Invalid activation type: {}'.format(act_type))


class CoorsNorm(nn.Module):
    r"""
    3D coordinate normalization.
    """
    def __init__(self, eps=1e-6, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors  # * self.scale


class FeatEncoder(torch.nn.Module):
    r"""
    A tool used to construct `node_encoder` and `edge_encoder` in GDL backbones.
    (Refer to GESS/models/backbones/base_backbone.py for details.)
    """
    def __init__(self, hidden_size, categorical_feat, scalar_feat, n_categorical_feat_to_use=-1,
                 n_scalar_feat_to_use=-1):
        super().__init__()
        self.embedding_list = torch.nn.ModuleList()

        self.num_categorical_feat = len(categorical_feat)
        self.n_categorical_feat_to_use = self.num_categorical_feat if n_categorical_feat_to_use == -1 else n_categorical_feat_to_use
        self.num_scalar_feat_to_use = scalar_feat if n_scalar_feat_to_use == -1 else n_scalar_feat_to_use

        for i in range(self.n_categorical_feat_to_use):
            num_categories = categorical_feat[i]
            emb = torch.nn.Embedding(num_categories, hidden_size)
            self.embedding_list.append(emb)

        if self.num_scalar_feat_to_use > 0:
            assert n_scalar_feat_to_use == -1
            self.linear = torch.nn.Linear(self.num_scalar_feat_to_use, hidden_size)

        total_cate_dim = self.n_categorical_feat_to_use * hidden_size
        self.dim_mapping = torch.nn.Linear(total_cate_dim + hidden_size,
                                           hidden_size) if self.num_scalar_feat_to_use > 0 else torch.nn.Linear(
            total_cate_dim, hidden_size)

    def forward(self, x):
        x_embedding = []
        for i in range(self.n_categorical_feat_to_use):
            x_embedding.append(self.embedding_list[i](x[:, i].long()))

        if self.num_scalar_feat_to_use > 0:
            x_embedding.append(self.linear(x[:, self.num_categorical_feat:]))

        x_embedding = self.dim_mapping(torch.cat(x_embedding, dim=-1))
        return x_embedding