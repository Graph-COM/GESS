r"""pipeline loader
"""

from typing import Dict
from torch.utils.data import DataLoader
from GESS.algorithms.baselines.base_algo import BaseAlgo
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.utils.utils import set_seed
from GESS import register


def load_pipeline(loader: Union[DataLoader, Dict[str, DataLoader]],
                  algorithm: BaseAlgo,
                  config: Union[CommonArgs, Munch]
                  ):
    r"""
    A pipeline loader.
    Args:
        name (str): Name of the chosen pipeline. Currently we have "BasePipeline", "O_Feature_Pipeline", and "Par_Label_Pipeline".
        config (Union[CommonArgs, Munch]).

    Returns:
        A instantiated pipeline.

    """
    try:
        set_seed(config.seed)
        pipeline = register.pipelines[config.pipeline](config, algorithm, loader)
    except KeyError as e:
        print(f'Pipeline {config.pipeline} does not exist.')
        raise e
    return pipeline
