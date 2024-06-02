import torch
from typing import Optional, List
from abc import ABC
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.models.models.base_model import BaseModel
from torch_geometric.data import Batch
from torch import Tensor
from .base_algo import BaseAlgo
from GESS import register


@register.algorithm_register
class ERM(BaseAlgo):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(ERM, self).__init__(config)
