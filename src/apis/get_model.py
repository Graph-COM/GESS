import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from src.backbones import DGCNN, EGNN, PointTransformer
from src.utils import ExtractorMLP, MLP, CoorsNorm


class Model(nn.Module):
    """
    set up the GDL models.
    """
    def __init__(self, model_name, model_config, method_name, method_config, dataset):
        super().__init__()
        self.dataset = dataset
        self.model_name = model_name
        self.dataset_name = dataset.dataset_name
        self.method_name = method_name
        self.specify_out = True
        self.one_encoder = method_config.get('one_encoder', True)
        self.kr = method_config.get('kr', None)

        out_dim = 1
        hidden_size = model_config['hidden_size']
        dropout_p = model_config['dropout_p']
        norm_type = model_config['norm_type']
        act_type = model_config['act_type']

        if model_name == 'dgcnn':
            Model = DGCNN
        elif model_name == 'egnn':
            Model = EGNN
        elif model_name == 'pointtrans':
            Model = PointTransformer
        else:
            raise NotImplementedError

        if model_config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif model_config['pool'] == 'max':
            self.pool = global_max_pool
        elif model_config['pool'] == 'add':
            self.pool = global_add_pool
        else:
            raise NotImplementedError
        dataset.feat_info['edge_categorical_feat'], dataset.feat_info['edge_scalar_feat'] = [], dataset.pos_dim + 1

        raw_pos_dim = dataset.pos_dim
        if dataset.feature_type == 'only_pos':
            dataset.x_dim = 0
        elif dataset.feature_type == 'only_x':
            dataset.pos_dim = 0
        elif dataset.feature_type == 'only_ones':
            dataset.x_dim = 0
            dataset.pos_dim = 0
        else:
            assert dataset.feature_type == 'both_x_pos'
        aux_info = {'raw_pos_dim': raw_pos_dim, 'dataset_name': dataset.dataset_name}
        self.coors_norm = CoorsNorm()
        # self.mlp_out for causal clf; self.mlp_out_conf fot non-causal clf (specifically used for DIR).
        self.mlp_out = MLP([hidden_size, hidden_size * 2, hidden_size, out_dim], dropout_p, norm_type, act_type)
        self.mlp_out_conf = MLP([hidden_size, hidden_size * 2, hidden_size, out_dim], dropout_p, norm_type,
                                act_type) if self.specify_out else None
        self.model = Model(dataset.x_dim, dataset.pos_dim, model_config, dataset.feat_info, aux_info=aux_info)
        self.emb_model = Model(dataset.x_dim, dataset.pos_dim, model_config, dataset.feat_info,
                               aux_info=aux_info) if not self.one_encoder else None

    def forward(self, data, node_attn=None, edge_attn=None, node_noise=None):
        x, pos, edge_index, edge_attr = self.calc_geo_feat(data.x, data.pos, data.batch, node_noise)

        # 1. add location noise
        # 2. knn graph
        # 3. edge_attr: [unit vector, distance] \in \R^4
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        emb = self.model(x, pos, edge_attr, edge_index, data.batch, edge_attn=edge_attn, node_attn=node_attn)
        pool_out = self.pool(emb, batch=data.batch)  # graph representation
        out = self.mlp_out(pool_out)
        return out

    def forward_pass(self, x, pos, edge_attr, edge_index, batch, edge_weight, with_enc=True):
        """The same as forward_pass_ but different input format"""
        emb = self.model(x, pos, edge_attr, edge_index, batch, edge_weight, with_enc=with_enc)
        pool_out = self.pool(emb, batch)
        return pool_out

    def forward_pass_(self, data):
        """without mlp specified, just encoder --> pooling"""
        x, pos, edge_index, edge_attr = self.calc_geo_feat(data.x, data.pos, data.batch, None)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        emb = self.model(x, pos, edge_attr, edge_index, data.batch)
        pool_out = self.pool(emb, batch=data.batch)
        return pool_out

    def forward_passing(self, x, pos, batch):
        x, pos, edge_index, edge_attr = self.calc_geo_feat(x, pos, batch, None)
        emb = self.model(x, pos, edge_attr, edge_index, batch)
        pool_out = self.pool(emb, batch)
        return pool_out

    def clf_out(self, pool_out):
        return self.mlp_out(pool_out)

    def conf_out(self, pool_out):
        return self.mlp_out_conf(pool_out)

    def get_emb(self, data):
        x, pos, edge_index, edge_attr = self.calc_geo_feat(data.x, data.pos, data.batch, None)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.pos = pos

        model = self.model if self.one_encoder else self.emb_model
        emb = model(x, pos, edge_attr, edge_index, data.batch)

        return emb, edge_index

    def calc_geo_feat(self, x, pos, batch, node_noise):

        if self.dataset_name == 'Track':
            pos = self.coors_norm(pos)
        edge_index = knn_graph(pos, k=5 if self.kr is None else int(self.kr), batch=batch, loop=True)
        edge_attr = self.calc_edge_attr(pos, edge_index)
        return x, pos, edge_index, edge_attr

    def calc_edge_attr(self, pos, edge_index):
        row, col = edge_index
        rel_dist = torch.norm(pos[row] - pos[col], dim=1, p=2, keepdim=True)
        coord_diff = pos[row] - pos[col]
        edge_dir = coord_diff / (rel_dist + 1e-6)
        edge_attr = torch.cat([rel_dist, edge_dir], dim=1)
        return edge_attr

    def add_geo_feature(self, data):
        x, pos, edge_index, edge_attr = self.calc_geo_feat(data.x, data.pos, data.batch, None)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.pos = pos
