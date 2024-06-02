import numpy as np
import torch
from typing import Optional, List
from abc import ABC
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.models.models.base_model import BaseModel
from torch_geometric.data import Batch
from torch import Tensor
from abc import ABCMeta, abstractmethod
import torch.nn as nn
import random
from .base_algo import BaseAlgo
from GESS import register


@register.algorithm_register
class Mixup(BaseAlgo):
    r"""
    Original Paper:
    @inproceedings{zhang2018mixup,
      title={mixup: Beyond Empirical Risk Minimization},
      author={Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David},
      booktitle={International Conference on Learning Representations},
      year={2018}
    }
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Mixup, self).__init__(config)
        self.alpha = config.algo.extra.alpha
        self.num_classes = config.algo.extra.num_classes
        self.prob = config.algo.extra.prob
        self.augment = Augments(prob=self.prob, alpha=self.alpha, num_classes=self.num_classes)

    def mix_criterion(self, output: Tensor, y_a: Tensor, y_b: Tensor, lam: float):
        """
        Args:
            output: model logits
            y_a: labels before permutation
            y_b: labels after permutation

        """
        return lam * self.criterion(output, y_a) + (1 - lam) * self.criterion(output, y_b)

    def forward_pass(self, data: Batch, epoch: Optional[int], phase: str):
        if phase != 'train':
            clf_logits = self.model(data=data)
            loss, loss_dict = self.__loss__(clf_logits, data.y)
            return loss, loss_dict, clf_logits

        feats = self.model.geo_dat_repr(data=data)
        (mix_feats, labels, labels_perm, lam), sign = self.augment(feats, data.y)
        output = self.model.clf_out(mix_feats)
        pred_loss = self.mix_criterion(output, labels, labels_perm, lam)

        return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item(), 'select_freq': sign}, output
    
    
class Identity(object):
    def __init__(self, num_classes: int):
        super(Identity, self).__init__()

        assert isinstance(num_classes, int)
        self.num_classes = num_classes
        self.sign = 0

    def __call__(self, img, gt_label):
        return img, gt_label, gt_label, 1


class BaseMixupLayer(object, metaclass=ABCMeta):
    """Base class for MixupLayer.

    Args:
        alpha (float): Parameters for Beta distribution.
        num_classes (int): The number of classes.
    """

    def __init__(self, alpha: float, num_classes: int):
        super(BaseMixupLayer, self).__init__()

        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(num_classes, int)

        self.alpha = alpha
        self.num_classes = num_classes
        self.sign = 1

    @abstractmethod
    def mixup(self, imgs, gt_label):
        pass


class BatchMixupLayer(BaseMixupLayer):
    """Mixup layer for batch mixup."""

    def __init__(self, *args, **kwargs):
        super(BatchMixupLayer, self).__init__(*args, **kwargs)

    def mixup(self, img: Tensor, gt_label: Tensor):
        # one_hot_gt_label = F.one_hot(gt_label, num_classes=self.num_classes)
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        # mixed_gt_label = lam * one_hot_gt_label + (1 - lam) * one_hot_gt_label[index, :]
        gt_label_perm = gt_label[index, :]

        return mixed_img, gt_label, gt_label_perm, lam

    def __call__(self, img: Tensor, gt_label: Tensor):
        return self.mixup(img, gt_label)


class Augments(object):
    """Data augments.

    We just implement mixup augment and identity.

    """

    def __init__(self, prob: float, alpha: float, num_classes: int):
        super(Augments, self).__init__()

        self.augments = [BatchMixupLayer(alpha=alpha, num_classes=num_classes), Identity(num_classes=num_classes)]
        self.augment_probs = [prob, 1-prob]

    def __call__(self, img: Tensor, gt_label: Tensor):
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        aug = random_state.choice(self.augments, p=self.augment_probs)
        return aug(img, gt_label), aug.sign




