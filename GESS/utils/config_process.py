r"""
This is adapted from GOOD: 
https://github.com/divelab/GOOD/blob/GOODv1/GOOD/utils/config_reader.py
A project configuration module that reads config argument from a file; set automatic generated arguments; and
overwrite configuration arguments by command arguments.
"""

import copy
import warnings
from os.path import join as opj
from pathlib import Path
from typing import Union
from pathlib import Path
import torch
from munch import Munch
from munch import munchify
from ruamel.yaml import YAML
from tap import Tap
import os
from .args import STORAGE_DIR
from GESS.utils.args import CommonArgs
from GESS.utils.metrics import Metrics


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py
    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.
    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def load_config(path: str, previous_includes: list = [], skip_include=False) -> dict:
    r"""Config loader.
    Loading configs from a config file.

    Args:
        path (str): The path to your yaml configuration file.
        previous_includes (list): Included configurations. It is for the :obj:`include` configs used for recursion.
            Please leave it blank when call this function outside.

    Returns:
        config (dict): config dictionary loaded from the given yaml file.
    """
    path = Path(path)
    if path in previous_includes:
        raise ValueError(
            f"Cyclic config include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]

    yaml = YAML(typ='safe')
    direct_config = yaml.load(open(path, "r"))
    if skip_include:
        return direct_config, None, None
    # direct_config = yaml.safe_load(open(path, "r"))

    # Load config from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )

    config = {}
    duplicates_warning = []
    duplicates_error = []

    for include in includes:
        include = path.parent / include
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def search_tap_args(args: CommonArgs, query: str):
    r"""
    Search a key in command line arguments.

    Args:
        args (CommonArgs): Command line arguments.
        query (str): The query for the target argument.

    Returns:
        A found or not flag and the target value if found.
    """
    found = False
    value = None
    for key in args.class_variables.keys():
        if query == key:
            found = True
            value = getattr(args, key)
        elif issubclass(type(getattr(args, key)), Tap):
            found, value = search_tap_args(getattr(args, key), query)
        if found:
            break
    return found, value


def args2config(config: Union[CommonArgs, Munch], args: CommonArgs):
    r"""
    Overwrite config by assigned arguments.
    If an argument is not :obj:`None`, this argument has the highest priority; thus, it will overwrite the corresponding
    config.

    Args:
        config (Union[CommonArgs, Munch]): Loaded configs.
        args (CommonArgs): Command line arguments.

    Returns:
        Overwritten configs.
    """
    for key in config.keys():
        # Use `extra` if you would like to add new arguments.
        if type(config[key]) is dict and key != "extra":
            args2config(config[key], args)
        elif key == 'extra':
            continue
        else:
            found, value = search_tap_args(args, key)
            if found:
                if value is not None:
                    config[key] = value
            else:
                warnings.warn(f'Argument {key} in the chosen config yaml file are not defined in command arguments, '
                              f'which will lead to incomplete code detection and the lack of argument temporary '
                              f'modification by adding command arguments.')


def process_configs(config: Union[CommonArgs, Munch]):
    r"""
    Process loaded configs.
    This process includes setting storage places for datasets, tensorboard logs, logs, and checkpoints. In addition,
    we also set random seed for each experiment round, checkpoint saving gap, and gpu device. Finally, we connect the
    config with two components :class:`GOOD.utils.metric.Metric` and :class:`GOOD.utils.train.TrainHelper` for easy and
    unified accesses.

    Args:
        config (Union[CommonArgs, Munch]): Loaded configs.

    Returns:
        Configs after setting.
    """
    # --- Dataset setting ---
    if config.dataset.dataset_root is None:
        config.dataset.dataset_root = opj(STORAGE_DIR, 'datasets')
    if not os.path.exists(config.dataset.dataset_root):
        os.makedirs(config.dataset.dataset_root)
    dataset_dirname = config.dataset.data_name + '_' + config.dataset.shift_name + '_' + str(config.dataset.target)
    
    # --- log setting ---
    if config.path.logging_dir is None:
        config.path.logging_dir = opj(STORAGE_DIR, 'logging', dataset_dirname, config.dataset.setting, config.algo.alg_name, config.backbone.name)
    if not os.path.exists(config.path.logging_dir):
        os.makedirs(config.path.logging_dir)

    config.path.logging_id_metrics = opj(config.path.logging_dir, "logging_id.txt")
    config.path.logging_ood_metrics = opj(config.path.logging_dir, "logging_ood.txt")
    config.path.logging_checkpoints = opj(config.path.logging_dir, f"checkpoints_seed#{config.seed}")
    config.path.loss_file = opj(config.path.logging_dir, "logging_train_loss.txt")
    config.path.result_path = opj(config.path.logging_dir, "result.txt")
    config.path.result_ood_path = opj(config.path.logging_dir, "result_ood.txt")

    config.path.load_pretrain_ckpt = opj(STORAGE_DIR, 'logging', dataset_dirname, 'No-Info', 'ERM', config.backbone.name, f"checkpoints_seed#{config.seed}", "id_best.ckpt")
    config.device = torch.device(f'cuda:{config.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    # --- setup metrics module ---
    config.metrics = Metrics(metric_name=config.dataset.metrics_name)


def config_summoner(args: CommonArgs) -> Union[CommonArgs, Munch]:
    r"""
    A config loading and postprocessing function.

    Args:
        args (CommonArgs): Command line arguments.

    Returns:
        Processed configs.
    """
    config, _, _ = load_config(args.config_path)
    gdl_config, _, _ = load_config(args.gdl)
    args2config(config, args)
    args2config(gdl_config, args)
    config["backbone"].update(gdl_config["backbone"])

    config = munchify(config)
    process_configs(config)
    
    return config