from .base_model import BaseModel
from GESS.models.models.tools import CoorsNorm, MLP
from GESS import register
from GESS.utils.config_process import Union, CommonArgs, Munch
from torch_geometric.data import InMemoryDataset
import torch


@register.model_register
class DIRModel(BaseModel):
    r"""
    DIR Model.
    New modules in this algorithm: 
        - Spurious classifier (self.classifier_conf)
        - Rationale generator (self.emb_model and self.extractor)
    """
    def __init__(self, config: Union[CommonArgs, Munch], dataset: InMemoryDataset, gdlencoder: torch.nn.Module):
        super().__init__(config, dataset, gdlencoder)
        hidden_size = self.hidden_size * 2
        self.classifier_conf = MLP([self.hidden_size, self.hidden_size * 2, self.hidden_size, self.out_dim], 
                                   self.dropout_p, self.norm_type, self.act_type)
        self.extractor = MLP([hidden_size, hidden_size * 2, hidden_size, self.out_dim], self.dropout_p, 
                             self.norm_type, self.act_type).to(config.device)
        self.emb_model = gdlencoder(dataset.x_dim, dataset.pos_dim, config.backbone, dataset.feat_info, aux_info=self.aux_info)
    
    def forward_pass(self, x, pos, edge_attr, edge_index, batch, edge_weight):
        emb = self.GDLencoder(x, pos, edge_attr, edge_index, batch, edge_weight, with_enc=False)
        pool_out = self.pool(emb, batch)
        return pool_out
    
    def conf_out(self, pool_out):
        return self.classifier_conf(pool_out)

    def get_emb(self, data):
        x, pos, edge_index, edge_attr = self.calc_geo_feat(data.x, data.pos, data.batch)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.pos = pos
        emb = self.emb_model(x, pos, edge_attr, edge_index, data.batch)
        return emb, edge_index