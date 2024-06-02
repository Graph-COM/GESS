r"""
Here includes necessary tools to log evaluation metrics, experimental results, and checkpoints.
"""
import torch
import numpy as np
from os.path import join as opj
from datetime import datetime
from sklearn.metrics import roc_auc_score, mean_absolute_error
import torch.nn.functional as F
import time
import os


def print_time_info(string, end='\n', dash_top=False, dash_bot=False, file=None):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-', file=file)
    print(string, end=end, file=file)
    if dash_bot:
        print(len(string) * '-', file=file)


def logger(print_str, log_file, print_=True):
    if print_:
        print_time_info(print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)


def results_logger(res_path, seed, report):
    logger(f'seed {seed}', res_path)
    for item in report.keys():
        logger(f'{item}: {report[item]}', res_path)


def to_item(tensor):
    if tensor is None:
        return None
    elif isinstance(tensor, torch.Tensor):
        return tensor.item()
    else:
        return tensor


def log(*args):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', *args)


def load_checkpoint(model, path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_checkpoint(model, model_dir, model_name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'model_state_dict': model.state_dict()}, opj(model_dir, model_name))

