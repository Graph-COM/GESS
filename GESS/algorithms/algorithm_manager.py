"""Algorithm loader.
"""

from GESS import register
from GESS.algorithms.baselines.base_algo import BaseAlgo
from GESS.utils.config_process import Union, CommonArgs, Munch


def load_algorithm(name, config: Union[CommonArgs, Munch]):
    r"""
    algorithm loader.
    Args:
        name: Name of the chosen learning algorithm.
        config: please refer to `config.algo` for more details.

    Returns:
        An algorithm object `BaseAlgo`.

    """
    try:
        ood_algorithm: BaseAlgo = register.algorithms[name](config)
    except KeyError as e:
        print(f'The Algorithm of given name {name} does not exist.')
        raise e
    return ood_algorithm
