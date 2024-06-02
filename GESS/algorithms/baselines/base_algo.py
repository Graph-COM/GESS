import torch
from typing import Optional, List
from abc import ABC
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.models.models.base_model import BaseModel
from torch_geometric.data import Batch
from torch import Tensor


class BaseAlgo(ABC):
    r"""
    Base class for the learning algorithms. The algorithm is able to manipulate various 
    components of the model, calculate loss, and perform optimization.
    You can inherit the `BaseAlgo` class to add new learning algorithms.

    Args:
        config config (Union[CommonArgs, Munch]): munchified dictionary of args.

    """
    def __init__(self, config: Union[CommonArgs, Munch]):

        super().__init__()
        self.criterion = None
        self.optimizer: torch.optim.Adam = None
        self.model: torch.nn.Module = None
        self.config = config
        self.device = config.device
        self.metrics_name = self.config.dataset.metrics_name

    def setup(self, model: BaseModel):
        r"""
        Setup the MLmodel, criterion, optimizer in the algorithm.
        """
        self.model = model
        self.setup_criterion()
        self.setup_optimizer()

    def setup_criterion(self,):
        r"""
        Setup criterion. Refer to `config.metrics.name2criterion` to see how
        the name of metrics (acc, auc, etc.) corresponds to the type of citerion.
        If you need add more criterions, override `self.setup_criterion`.
        """
        self.criterion = self.config.metrics.name2criterion[self.metrics_name]()
    
    def setup_optimizer(self,):
        r"""
        Default optimizer setup. If you need to design a specific optimizer for your algorithm, 
        override `self.setup_optimizer` (refer to GESS/algorithms/baselines/dir.py for an example).
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train.lr,
                                                        weight_decay=self.config.train.wd)
    
    def __loss__(self, clf_logits: Tensor, clf_labels: Tensor):
        r"""
        Loss calculation.
        Returns:
            - The calculated loss;
            - A dictionary with all losses or other auxiliary information;
        """
        pred_loss = self.criterion(clf_logits, clf_labels.view(clf_logits.shape).float())
        return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

    def loss_postprocess(self, loss: Union[Tensor, List[Tensor]], data: Batch):
        r"""
        Process loss. Refer to GESS/algorithms/baselines/VREx.py for an example.
        """
        return loss

    def forward_pass(self, data: Batch, epoch: Optional[int], phase: Optional[str]):
        r"""
        This method 
        Returns:
            - loss: The calculated loss;
            - loss_dict: A dictionary with losses (including prediction loss and other losses specific to an algorithm) and other auxiliary information;
            - clf_logits: The model output.
        """
        clf_logits = self.model(data=data)
        loss, loss_dict = self.__loss__(clf_logits, data.y)
        return loss, loss_dict, clf_logits
    
    def loss_backward(self, loss: Tensor):
        r"""
        Gradient backward and parameter update.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()