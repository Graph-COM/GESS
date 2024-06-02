import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from GESS.models.models.tools import CoorsNorm, MLP
from GESS import register
from GESS.utils.config_process import Union, CommonArgs, Munch
from torch_geometric.data import InMemoryDataset


@register.model_register
class BaseModel(nn.Module):
    r"""
    The basic ML model, which is composed of a GDL encoder and a MLP.
    If you need more modules in your added algorithms, please specify a new model.
    Please refer to GESS/models/models/DIR_model.py for an example.
    """
    def __init__(self, config: Union[CommonArgs, Munch], dataset: InMemoryDataset, gdlencoder: torch.nn.Module):
        super().__init__()

        self.dataset_name = config.dataset.data_name
        self.kr = config.backbone.kr
        self.hidden_size = config.backbone.hidden_size
        self.dropout_p = config.backbone.dropout_p
        self.norm_type = config.backbone.norm_type
        self.act_type = config.backbone.act_type
        self.out_dim = config.dataset.num_task

        if config.backbone.pool == 'mean':
            self.pool = global_mean_pool
        elif config.backbone.pool == 'max':
            self.pool = global_max_pool
        elif config.backbone.pool == 'add':
            self.pool = global_add_pool
        else:
            raise NotImplementedError
        # Get basic dataset information.
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
        # if you would like to add any auxiliary information to the GDL encoder
        self.aux_info = {'raw_pos_dim': raw_pos_dim, }
        self.GDLencoder = gdlencoder(dataset.x_dim, dataset.pos_dim, config.backbone, dataset.feat_info, aux_info=self.aux_info)
        self.coors_norm = CoorsNorm() if 'Track' in config.dataset.data_name else nn.Identity()
        self.classifier = MLP([self.hidden_size, self.hidden_size * 2, self.hidden_size, self.out_dim], self.dropout_p, self.norm_type, self.act_type)

    
    def forward(self, *args, **kwargs):
        r"""
        Data -> GDLencoder -> MLP
        This method supports different forms of args or kwargs.
        """
        pool_out = self.geo_dat_repr(*args, **kwargs)
        out = self.clf_out(pool_out)
        return out

    def geo_dat_repr(self, *args, **kwargs):
        r"""
        Get the representation of the geometric data. 
        Data -> create geometric graph -> GDLencoder -> pooling
        This method supports different forms of args or kwargs.
        """
        node_attn = kwargs.get("node_attn", None)
        edge_attn = kwargs.get("edge_attn", None)
        data = kwargs.get("data", None)
        if not data:
            if not args:
                assert 'x' in kwargs and 'pos' in kwargs and 'batch' in kwargs
                x, pos, batch = kwargs['x'], kwargs['pos'], kwargs['batch']
            elif len(args) == 3:
                x, pos, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"geo_dat_repr should take 3 arguments but got {len(args)}")
        else:
            x, pos, batch = data.x, data.pos, data.batch
        x, pos, edge_index, edge_attr = self.calc_geo_feat(x, pos, batch)
        emb = self.GDLencoder(x, pos, edge_attr, edge_index, batch, edge_attn=edge_attn, node_attn=node_attn)
        pool_out = self.pool(emb, batch=batch)
        return pool_out

    def clf_out(self, pool_out):
        return self.classifier(pool_out)

    def calc_geo_feat(self, x, pos, batch):
        r"""
        create geometric graph according to its 3d coordinates.
        """
        pos = self.coors_norm(pos)
        edge_index = knn_graph(pos, k=self.kr, batch=batch, loop=True)
        edge_attr = self.calc_edge_attr(pos, edge_index)
        return x, pos, edge_index, edge_attr

    def calc_edge_attr(self, pos, edge_index):
        row, col = edge_index
        rel_dist = torch.norm(pos[row] - pos[col], dim=1, p=2, keepdim=True)
        coord_diff = pos[row] - pos[col]
        edge_dir = coord_diff / (rel_dist + 1e-6)
        edge_attr = torch.cat([rel_dist, edge_dir], dim=1)
        return edge_attr

