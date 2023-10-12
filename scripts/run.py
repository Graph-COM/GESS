import sys
import os

current_script_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_directory)
sys.path.append(project_root)

from src.utils import set_seed
from src.apis import run_a_seed
from pathlib import Path
import yaml
import argparse
import torch
torch.set_num_threads(2)

def main():
    parser = argparse.ArgumentParser(description='GDL-DS Benchmark')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='Track')
    parser.add_argument('-m', '--method', type=str, help='method used', default='erm')
    parser.add_argument('-s', '--shift', type=str, help='shift type', default='pileup')
    parser.add_argument('-t', '--target', type=str, help='specific cases for the shift', default='50')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=1)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--setting', type=str, help='option: No-Info, O-Feature, Par-Label', default='No-Info')
    parser.add_argument('--finetune_num', type=int, help='the number of data entries used for model fine-tuning', default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    config_dir = Path("../src") / "configs"
    config_path = config_dir / args.dataset / f'{args.shift}_{args.target}.yml'
    config_model_path = config_dir / 'model.yml'
    config_path_path = config_dir / 'path.yml'
    config = yaml.safe_load(config_path.open('r'))
    model_config = yaml.safe_load(config_model_path.open('r'))
    path_config = yaml.safe_load(config_path_path.open('r'))
    if config[args.method].get(args.backbone, False):
        config[args.method].update(config[args.method][args.backbone])
    config.update(model_config)
    config.update(path_config)
    config['data']['setting'] = args.setting
    config['shift'][args.shift]['restrict_TL_train'] = args.finetune_num
    setting = f'{args.setting}_#{args.finetune_num}' if args.setting == 'Par-Label' else args.setting
    device = torch.device(f'cuda:{args.cuda}' if args.cuda >= 0 else 'cpu')

    log_dir = Path(
        config['dir_config'][
            'log_dir']) / args.dataset / args.method / f'{args.backbone}_{args.shift}_{args.target}_{setting}'
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("../logging") / args.dataset / 'erm' / f'{args.backbone}_{args.shift}_{args.target}_No-Info' # used for fine-tuning
    run_a_seed(config, args.method, args.backbone, args.seed, args.dataset, log_dir, device, args.shift, model_dir)


if __name__ == '__main__':
    main()
