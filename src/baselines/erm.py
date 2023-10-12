import torch
import torch.nn as nn
import numpy as np


class ERM(nn.Module):
    def __init__(self, clf, criterion):
        super().__init__()
        self.clf = clf
        self.criterion = criterion
        self.device = next(self.parameters()).device

    def __loss__(self, clf_logits, clf_labels):
        if len(clf_logits.shape) != len(clf_labels.shape):
            clf_labels = clf_labels.reshape(clf_logits.shape)
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

    def forward_pass(self, data, epoch, phase):
        clf_logits = self.clf(data)
        loss, loss_dict = self.__loss__(clf_logits, data.y)
        return loss, loss_dict, clf_logits
