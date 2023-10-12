from src.baselines import ERM, DIR, MixUp, GroupDRO, LRIBern, VREx, Coral, DANN, DomainAdversarialLoss
import torch
from src.utils import *


def get_baseline(setting, method_name, clf, config, seed, model_dir=None):
    """
    get baseline and optimizer for running
    Args:
        setting: No-Info, O-Feature, and Par-Label settings mentioned in our paper;
        method_name: We select ERM, VREx, GroupDRO, MixUp, DIR, LRI (No-Info Level); Coral, DANN (O-Feature); and TL (Par-Label) methods;
        clf: The used GDL model;
        config: config about models, algorithms ond optimizers for running;
        seed: random seed;
        model_dir: path of models to be loaded for model fine-tuning.

    Returns: baseline and optimizer

    """
    metrics = config['data']['metrics']
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean") if metrics != 'mae' else torch.nn.modules.loss.MSELoss()
    # Criterion of some algos are specified.
    optimizer = get_wp_optimizer(clf, config['optimizer'])
    # optimizer of DIR, LRI, DANN should be specified.
    if method_name == 'erm':
        if setting == "Par-Label":
            assert model_dir is not None
            clf = load_model(seed, deepcopy(clf), model_dir, metrics).to(next(clf.parameters()).device)
            optimizer = get_wp_optimizer(clf, config['optimizer'])
        baseline = ERM(clf, criterion)

    elif method_name == 'lri_bern':
        extractor = ExtractorMLP(config['model'][clf.model_name]['hidden_size'], config[method_name]).to(
            next(clf.parameters()).device)
        optimizer = get_optimizer(clf, extractor, config['optimizer'])
        baseline = LRIBern(clf, extractor, criterion, config['lri_bern'])

    elif method_name == 'mixup':
        baseline = MixUp(clf, criterion, config['mixup'])

    elif method_name == 'dir':
        extractor = ExtractorMLP(config['model'][clf.model_name]['hidden_size'] * 2, config[method_name]).to(
            next(clf.parameters()).device)
        baseline = DIR(clf, extractor, criterion, config['dir'])
        optimizer = get_dir_optimizer(clf, extractor, config['optimizer'], config['dir'])

    elif method_name == 'groupdro':
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none") if metrics != 'mae' else torch.nn.modules.loss.MSELoss(reduction="none")
        baseline = GroupDRO(clf, criterion, config['groupdro'])

    elif method_name == 'VREx':
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none") if metrics != 'mae' else torch.nn.modules.loss.MSELoss(reduction="none")
        baseline = VREx(clf, criterion, config['VREx'])

    elif method_name == 'coral':
        baseline = Coral(clf, criterion, config[method_name])

    elif method_name == 'DANN':
        disc = domain_disc(config['model'][clf.model_name]['hidden_size'], config[method_name]).to(
            next(clf.parameters()).device)
        domain_adv = DomainAdversarialLoss(disc, criterion).to(next(clf.parameters()).device)
        optimizer = get_dann_optimizer(clf, disc, config['optimizer'])
        baseline = DANN(clf, domain_adv, criterion, config[method_name])
    else:
        raise NotImplementedError
    return baseline, optimizer
