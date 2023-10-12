import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, mean_absolute_error
import torch.nn.functional as F
import time


def print_time_info(string, end='\n', dash_top=False, dash_bot=False, file=None):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-', file=file)
    print(string, end=end, file=file)
    if dash_bot:
        print(len(string) * '-', file=file)


def write_log(print_str, log_file, print_=True):
    if print_:
        print_time_info(print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)


def to_item(tensor):
    if tensor is None:
        return None
    elif isinstance(tensor, torch.Tensor):
        return tensor.item()
    else:
        return tensor


def get_preds_from_logits(logits):
    if logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds


def log(*args):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', *args)


def load_checkpoint(model, model_dir, model_name, map_location=None):
    checkpoint = torch.load(model_dir / (model_name + '.pt'), map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_checkpoint(model, model_dir, model_name):
    torch.save({'model_state_dict': model.state_dict()}, f'{model_dir}/{model_name}.pt')


def update_and_save_best_epoch_clf(metrics, seed, baseline, train_res, valid_res, test_res, metric_dict, epoch,
                                   file_path,
                                   model_dir, is_ood=False):
    better_val_metric = valid_res[0] > metric_dict[f'metric/best_clf_valid_{metrics}']
    same_val_metric_but_better_val_loss = (valid_res[0] == metric_dict[f'metric/best_clf_valid_{metrics}']) and (
            valid_res[-1] < metric_dict['metric/best_clf_valid_loss'])

    metric_dict['metric/clf_train_loss'] = train_res[-1]
    metric_dict[f'metric/best_clf_train_{metrics}'] = train_res[0]
    if better_val_metric or same_val_metric_but_better_val_loss:
        metric_dict['metric/best_clf_epoch'] = epoch
        metric_dict['metric/best_clf_valid_loss'] = valid_res[-1]

        metric_dict[f'metric/best_clf_valid_{metrics}'] = valid_res[0]
        metric_dict[f'metric/best_clf_test_{metrics}'] = test_res[0]
        write_log(f'***Update in epoch {epoch}!***', log_file=file_path)
        if model_dir is not None:
            save_checkpoint(baseline, model_dir,
                            model_name=f'model_{metrics}_{seed}' if not is_ood else f'model_ood_{metrics}_{seed}')
    return metric_dict


def update_and_save_best_epoch_rgs(metrics, seed, baseline, train_res, valid_res, test_res, metric_dict, epoch,
                                   file_path,
                                   model_dir, is_ood=False):
    assert metrics == 'mae'
    better_val_mae = valid_res[0] < metric_dict['metric/best_regrs_valid_mae']
    same_val_mae_but_better_val_loss = (valid_res[0] == metric_dict['metric/best_regrs_valid_mae']) and (
            valid_res[-1] < metric_dict['metric/best_regrs_valid_loss'])

    metric_dict['metric/regrs_train_loss'] = train_res[-1]
    metric_dict['metric/best_regrs_train_mae'] = train_res[0]
    if better_val_mae or same_val_mae_but_better_val_loss:
        metric_dict['metric/best_regrs_epoch'] = epoch
        metric_dict['metric/best_regrs_valid_loss'] = valid_res[-1]
        metric_dict['metric/best_regrs_valid_mae'] = valid_res[0]
        metric_dict['metric/best_regrs_test_mae'] = test_res[0]
        write_log(f'***Update in epoch {epoch}!***', log_file=file_path)
        if model_dir is not None:
            save_checkpoint(baseline, model_dir,
                            model_name=f'model_mae_{seed}' if not is_ood else f'model_ood_mae_{seed}')
    return metric_dict
