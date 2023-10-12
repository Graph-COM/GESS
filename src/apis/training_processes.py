from src.utils import *
import torch
from tqdm import tqdm
from ..apis.get_model import Model
from ..apis.get_baseline import get_baseline
from pathlib import Path


def write_res(res_path, seed, report):
    write_log(f'seed {seed}', res_path)
    for item in report.keys():
        write_log(f'{item}: {report[item]}', res_path)


def run_a_seed(config, method_name, model_name, seed, dataset_name, log_dir, device, shift_name, model_dir=None):
    setting = config['data']['setting']
    shift_config = config['shift'][shift_name]
    loaders, Dataset = get_data_loaders(dataset_name, config, shift_config, seed)
    clf = Model(model_name, config['model'][model_name], method_name, config[method_name], Dataset).to(device)
    model_dir = model_dir if setting == "Par-Label" else None
    baseline, optimizer = get_baseline(setting, method_name, clf, config, seed, model_dir)
    Runner = DARunner if setting == "O-Feature" else BaseRunner
    runner = Runner(dataset_name, method_name, log_dir, config, baseline, optimizer, loaders, seed)
    runner.start_pipeline()


class BaseRunner:
    """
    start a pipeline: run for predefined epochs, log metric scores during the pipeline, and obtain final results.
    """
    def __init__(self, dataset_name, method_name, log_dir, config, baseline, optimizer, loaders, seed):
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.setting = config['data']['setting']
        self.epochs = config[method_name]['epochs']
        self.baseline = baseline
        self.optimizer = optimizer
        self.loaders = loaders
        self.seed = seed
        self.metrics = config['data']['metrics']
        assert self.metrics in ['acc', 'auc', 'mae']
        self.run_ood = False if self.setting == 'Par-Label' else True

        self.log_dir = log_dir
        dir_config = config['dir_config']
        self.file_path = log_dir / dir_config['logging_file']
        self.file_path_ood = log_dir / dir_config['logging_ood_file']
        self.file_loss = log_dir / dir_config['loss_file']
        self.result_log_path = log_dir / dir_config['result_path']
        self.result_ood_log_path = log_dir / dir_config['result_ood_path']

        self.train_res, self.valid_res, self.test_res, self.ood_valid_res, self.ood_test_res = None, None, None, None, None

        if self.metrics == 'acc':
            self.metric_dict, self.metric_dict_ood = deepcopy(init_metric_dict_ood_acc), deepcopy(
                init_metric_dict_ood_acc)
        elif self.metrics == 'auc':
            self.metric_dict, self.metric_dict_ood = deepcopy(init_metric_dict_ood_auc), deepcopy(
                init_metric_dict_ood_auc)
        elif self.metrics == 'mae':
            self.metric_dict, self.metric_dict_ood = deepcopy(init_metric_dict_ood_mae), deepcopy(
                init_metric_dict_ood_mae)
        else:
            raise NotImplementedError

    def start_pipeline(self):
        self.run_and_log()
        self.log_results()

    def run_and_log(self):
        val = "val" if not self.run_ood else "iid_val"
        test = "test" if not self.run_ood else "iid_test"
        for epoch in range(self.epochs):
            self.train_res = self.run_one_epoch(self.optimizer, self.loaders['train'], epoch, 'train')
            self.valid_res = self.run_one_epoch(None, self.loaders[val], epoch, 'iid_valid')
            self.test_res = self.run_one_epoch(None, self.loaders[test], epoch, 'iid_test')
            if self.run_ood:
                self.ood_valid_res = self.run_one_epoch(None, self.loaders['ood_val'], epoch, 'ood_valid')
                self.ood_test_res = self.run_one_epoch(None, self.loaders['ood_test'], epoch, 'ood_test')
            self.log_and_update(epoch)

    def run_one_epoch(self, optimizer, data_loader, epoch, phase):
        loader_len = len(data_loader)
        run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
        log_dict = {'model_out': [], 'labels': []}
        all_loss_dict = {}
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            loss_dict, model_out = run_one_batch(self.baseline, optimizer, data.to(self.baseline.device), epoch)
            labels = to_cpu(data.y)
            for key in log_dict.keys():
                log_dict[key].append(eval(key))

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            if idx == loader_len - 1:
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                metric_score = self.get_metric_score(log_dict)
        return metric_score, all_loss_dict, all_loss_dict['pred']

    def get_metric_score(self, log_dict):
        model_out = torch.cat(log_dict['model_out'])
        labels = torch.cat(log_dict['labels'])
        metric_score = None
        if self.metrics == 'acc':
            clf_preds = get_preds_from_logits(model_out)
            clf_preds = clf_preds.reshape(labels.shape)
            metric_score = (clf_preds == labels).sum().item() / labels.shape[0]
        elif self.metrics == 'auc':
            metric_score = roc_auc_score(labels, model_out.sigmoid())
        elif self.metrics == 'mae':
            metric_score = mean_absolute_error(labels, model_out)
        return metric_score

    def log_and_update(self, epoch):
        # log performance per epoch
        # 0 for metric_score, 1 for log_dictionary, -1 for avg_loss
        update_and_save_best_epoch = update_and_save_best_epoch_rgs if self.metrics == 'mae' else update_and_save_best_epoch_clf
        write_log(f'epoch {epoch} ' + f"{' '.join([f'{k} {v}' for k, v in self.train_res[1].items()])}",
                  log_file=self.file_loss)
        write_log(
            f'epoch {epoch} val_loss {self.valid_res[-1]} test_loss {self.test_res[-1]} train_{self.metrics} {self.train_res[0]} val_{self.metrics} {self.valid_res[0]} test_{self.metrics} {self.test_res[0]}',
            log_file=self.file_path)
        self.metric_dict = update_and_save_best_epoch(self.metrics, self.seed, self.baseline, self.train_res,
                                                      self.valid_res, self.test_res,
                                                      self.metric_dict,
                                                      epoch, self.file_path, self.log_dir)

        if self.run_ood:
            write_log(
                f'epoch {epoch} ood_val_loss {self.ood_valid_res[-1]} ood_test_loss {self.ood_test_res[-1]} train_{self.metrics} {self.train_res[0]} ood_val_{self.metrics} {self.ood_valid_res[0]} ood_test_{self.metrics} {self.ood_test_res[0]}',
                log_file=self.file_path_ood)
            self.metric_dict_ood = update_and_save_best_epoch(self.metrics, self.seed, self.baseline,
                                                              self.train_res, self.ood_valid_res,
                                                              self.ood_test_res,
                                                              self.metric_dict_ood, epoch,
                                                              self.file_path_ood, self.log_dir, is_ood=True)

    def log_results(self):
        report_dict = {k.replace('metric/best_', ''): v for k, v in self.metric_dict.items()}
        report_dict_ood = {k.replace('metric/best_', ''): v for k, v in self.metric_dict_ood.items()}
        write_res(self.result_log_path, self.seed, report_dict)
        if self.run_ood:
            write_res(self.result_ood_log_path, self.seed, report_dict_ood)


class DARunner(BaseRunner):
    """
    a specified version for O-Feature setting.
    """
    def __init__(self, dataset_name, method_name, log_dir, config, baseline, optimizer, loaders, seed):
        super().__init__(dataset_name, method_name, log_dir, config, baseline, optimizer, loaders, seed)
        self.iters_per_epoch = config[method_name]['iters_per_epoch']

    def run_and_log(self):
        for epoch in range(self.epochs):
            self.train_res = self.train_one_epoch_DA(self.optimizer,
                                                     (self.loaders['train_source'], self.loaders['train_target']),
                                                     epoch, 'train', self.iters_per_epoch)
            self.valid_res = self.run_one_epoch(None, self.loaders['iid_val'], epoch, 'iid_valid')
            self.test_res = self.run_one_epoch(None, self.loaders['iid_test'], epoch, 'iid_test')
            self.ood_valid_res = self.run_one_epoch(None, self.loaders['ood_val'], epoch, 'ood_valid')
            self.ood_test_res = self.run_one_epoch(None, self.loaders['ood_test'], epoch, 'ood_test')
            self.log_and_update(epoch)

    def train_one_epoch_DA(self, optimizer, data_loader, epoch, phase, iters_per_epoch):
        all_loss_dict = {}
        # training process
        log_dict = {'model_out': [], 'labels': []}
        train_source_iter, train_target_iter = data_loader
        pbar = tqdm(range(iters_per_epoch))
        for i in pbar:
            data_s = next(train_source_iter).to(self.baseline.device)
            data_t = next(train_target_iter).to(self.baseline.device)
            loss_dict, model_out = train_one_batch(self.baseline, optimizer, (data_s, data_t), epoch)
            if len(data_s.y.shape) == 1:
                data_s.y = data_s.y.unsqueeze(1)
            labels = to_cpu(data_s.y)
            for key in log_dict.keys():
                log_dict[key].append(eval(key))

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            if i == iters_per_epoch - 1:
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / iters_per_epoch
                metric_score = self.get_metric_score(log_dict)
        return metric_score, all_loss_dict, all_loss_dict['pred']


def eval_one_batch(baseline, optimizer, data, epoch):
    assert optimizer is None
    baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
    baseline.domain_adv.eval() if hasattr(baseline, 'domain_adv') else None
    baseline.clf.eval()
    _, loss_dict, org_clf_logits = baseline.forward_pass(data, epoch, "not_train")
    return loss_dict, to_cpu(org_clf_logits)


def train_one_batch(baseline, optimizers, data, epoch):
    baseline.clf.train()
    if hasattr(baseline, 'extractor'):
        baseline.extractor.train()
    if hasattr(baseline, 'domain_adv'):
        baseline.domain_adv.train()

    loss, loss_dict, org_clf_logits = baseline.forward_pass(data, epoch, "train")
    loss = (loss, ) if not isinstance(loss, tuple) else loss
    optimizers = (optimizers, ) if not isinstance(optimizers, tuple) else optimizers
    for i in range(len(loss)):
        optimizers[i].zero_grad()
        loss[i].backward()
        optimizers[i].step()
    return loss_dict, to_cpu(org_clf_logits)
