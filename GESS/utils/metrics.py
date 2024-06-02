from sklearn.metrics import roc_auc_score, mean_absolute_error
from GESS.utils.logger import logger, save_checkpoint
import torch

class Metrics(object):
    r"""
    Metrics module with various evaluation metrics, criterion, and metric-updating tools.
    In this benchmark, `acc`, `auc`, `mae` is provided. You could extend this module to more diverse metrics.
    """
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.name2score = {
            'acc': self.acc,
            'auc': self.auc,
            'mae': self.mae
        }  # calculations for evaluation metrics
        self.name2update_func = {
            'acc': higher_better,
            'auc': higher_better,
            'mae': lower_better
        }  # metric-updating functions
        self.name2criterion = {
            'acc': torch.nn.BCEWithLogitsLoss,
            'auc': torch.nn.BCEWithLogitsLoss,
            'mae': torch.nn.modules.loss.MSELoss           
        }  # for loss calculation.
        self.metrics_id = {
            'metric/best_epoch': 0,
            'metric/train_loss': 0,
            'metric/best_valid_loss': 0,
            'metric/best_train_metric': 0,
            'metric/best_valid_metric': 0 if self.metric_name != 'mae' else 1e6,
            'metric/best_test_metric': 0,
        }  # metrics related to in-distribution performance
        self.metrics_ood = {
            'metric/best_epoch': 0,
            'metric/train_loss': 0,
            'metric/best_valid_loss': 0,
            'metric/best_train_metric': 0,
            'metric/best_valid_metric': 0 if self.metric_name != 'mae' else 1e6,
            'metric/best_test_metric': 0,
        }  # metrics related to out-of-distribution performance
    
    def acc(self, labels, model_out):
        preds = get_preds_from_logits(model_out)
        preds = preds.reshape(labels.shape)
        return (preds == labels).sum().item() / labels.shape[0]

    def auc(self, labels, model_out):
        return roc_auc_score(labels, model_out.sigmoid())

    def mae(self, labels, model_out):
        return mean_absolute_error(labels, model_out)

    def cal_metrics_score(self, model_out, labels):
        r"""
        calculate evaluation metrics.
        """
        return self.name2score[self.metric_name](model_out, labels)
    
    def update_metrics(self, metric_dict, train_res, valid_res, test_res, epoch):
        better_val_metric = self.name2update_func[self.metric_name](valid_res[0], metric_dict['metric/best_valid_metric'])

        same_val_but_better_val_loss = (valid_res[0] == metric_dict['metric/best_valid_metric']) and (
                valid_res[-1] < metric_dict['metric/best_valid_loss'])
        
        metric_dict['metric/train_loss'] = train_res[-1]
        metric_dict['metric/best_train_metric'] = train_res[0]

        if better_val_metric or same_val_but_better_val_loss:
            metric_dict['metric/best_epoch'] = epoch
            metric_dict['metric/best_valid_loss'] = valid_res[-1]
            metric_dict['metric/best_valid_metric'] = valid_res[0]
            metric_dict['metric/best_test_metric'] = test_res[0]
            return True
        return False
        
    
    def update_id_metrics(self, train_res, valid_res, test_res, epoch, config, model):
        r"""
        update in-distribution evaluation metrics and save checkpoints after one epoch.
        """
        logger(
                f'#In-Domain INFO#epoch {epoch} val_loss {valid_res[-1]} test_loss {test_res[-1]} train_metric {train_res[0]} val_metric {valid_res[0]} test_metric {test_res[0]}',
                log_file=config.path.logging_id_metrics)
        if self.update_metrics(self.metrics_id, train_res, valid_res, test_res, epoch):
            logger(f'#INFO#Update in epoch {epoch}!', log_file=config.path.logging_id_metrics)
            save_checkpoint(model, config.path.logging_checkpoints, f'id_best.ckpt')
            logger('#INFO#Saved a new best In-Domain checkpoint.', log_file=config.path.logging_id_metrics)



    def update_ood_metrics(self, train_res, valid_res, test_res, epoch, config, model):
        r"""
        update out-of-distribution evaluation metrics and save checkpoints after one epoch.
        """
        logger(
                f'#Out-of-Domain INFO#epoch {epoch} val_loss {valid_res[-1]} test_loss {test_res[-1]} train_metric {train_res[0]} val_metric {valid_res[0]} test_metric {test_res[0]}',
                log_file=config.path.logging_ood_metrics)
        if self.update_metrics(self.metrics_ood, train_res, valid_res, test_res, epoch):
            logger(f'#INFO#Update in epoch {epoch}!', log_file=config.path.logging_ood_metrics)
            save_checkpoint(model, config.path.logging_checkpoints, f'ood_best.ckpt')
            logger('#INFO#Saved a new best Out-of-Domain checkpoint.', log_file=config.path.logging_ood_metrics)




def higher_better(new_val, given_val):
    return new_val > given_val

def lower_better(new_val, given_val):
    return new_val < given_val


def get_preds_from_logits(logits):
    if logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds