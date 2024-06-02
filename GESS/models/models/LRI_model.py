from .base_model import BaseModel
from GESS.models.models.tools import CoorsNorm, MLP
from GESS import register
from GESS.utils.config_process import Union, CommonArgs, Munch
from torch_geometric.data import InMemoryDataset
import torch


@register.model_register
class LRIModel(BaseModel):
    r"""
    LRI Model.
    New modules in this algorithm:
        - Extractor: self.extractor 
    """
    def __init__(self, config: Union[CommonArgs, Munch], dataset: InMemoryDataset, gdlencoder: torch.nn.Module):
        super().__init__(config, dataset, gdlencoder)
        self.extractor = MLP([self.hidden_size, self.hidden_size * 2, self.hidden_size, self.out_dim], self.dropout_p, 
                             self.norm_type, self.act_type).to(config.device)
    
    def get_emb(self, data):
        x, pos, edge_index, edge_attr = self.calc_geo_feat(data.x, data.pos, data.batch)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.pos = pos
        emb = self.GDLencoder(x, pos, edge_attr, edge_index, data.batch)
        return emb, edge_index