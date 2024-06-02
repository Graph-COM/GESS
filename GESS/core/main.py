from GESS import config_summoner, args_parser
from GESS.data.data_manager import create_dataloader, load_dataset
from GESS.models.model_manager import load_model
from GESS.algorithms.algorithm_manager import load_algorithm
from GESS.core.pipeline_manager import load_pipeline
from GESS import register
import torch
torch.set_num_threads(2)

def main():

    # setup arguments used in this project.
    args = args_parser()
    config = config_summoner(args)

    # load datasets and dataloader.
    dataset = load_dataset(config.dataset.data_name, config)
    dataloader = create_dataloader(dataset, config)

    # setup model.
    model = load_model(config.algo.model_name, dataset, config, register.gdlbackbones[config.backbone.name]).to(config.device)
    
    # setup algorithm.
    algorithm = load_algorithm(config.algo.alg_name, config)
    algorithm.setup(model)
   
    # setup training & evaluating pipeline.
    pipeline = load_pipeline(dataloader, algorithm, config)
    pipeline.start_pipeline()

if __name__ == '__main__':
    main()