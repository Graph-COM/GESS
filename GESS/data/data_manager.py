r"""A module that is consist of a dataset loading function and a PyTorch dataloader loading function.
"""
from os.path import join as opj
from GESS import register
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.utils.utils import set_seed
from torch_geometric.data import InMemoryDataset


def load_dataset(name: str, config: Union[CommonArgs, Munch]):
    r"""
    Load a dataset given the dataset name. Currently this project includes "DrugOOD_3D", "QMOF", "Track_Pileup", "Track_Signal".

    Args:
        name (str): Dataset name.
        config (Union[CommonArgs, Munch]): Refer to `config.dataset` for more details.

    Returns:
        A dataset object.

    """
    try:
        set_seed(config.seed)
        dataset = register.datasets[name](root=opj(config.dataset.dataset_root, config.dataset.data_name), dataset_config=config.dataset)
    except KeyError as e:
        print('Dataset not found.')
        raise e
    return dataset


def create_dataloader(dataset: InMemoryDataset, config: Union[CommonArgs, Munch]):
    r"""
    Create a PyG dataloader.

    Args:
        dataset.
        config (Union[CommonArgs, Munch]): Refer to `config.train` for more details.

    Returns:
        A PyG dataset loader.

    """
    loader_name = config.dataset.dataloader_name
    try:
        set_seed(config.seed)
        loader = register.dataloaders[loader_name](config).setup(dataset)
    except KeyError as e:
        print(f'DataLoader {loader_name} not found.')
        raise e

    return loader
