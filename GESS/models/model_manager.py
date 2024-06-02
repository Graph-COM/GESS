r"""ML model loader.
"""
import torch
from GESS import register
from GESS.utils.utils import set_seed
from GESS.utils.config_process import Union, CommonArgs, Munch



def load_model(name: str, dataset, config: Union[CommonArgs, Munch], gdlencoder: torch.nn.Module):
    r"""
    A model loader.
    Args:
        name (str): Name of the chosen ML model. This project includes `BaseModel`, `LRIModel`, `DIRModel`, and `DANNModel`.
        config (Union[CommonArgs, Munch]).

    Returns:
        A instantiated GNN model.

    """
    try:
        set_seed(config.seed)
        model = register.models[name](config, dataset, gdlencoder)
    except KeyError as e:
        print(f'Model {name} does not exist.')
        raise e
    return model