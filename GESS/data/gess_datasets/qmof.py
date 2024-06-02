from torch_geometric.data import InMemoryDataset, Data
from GESS.utils.config_process import Union, CommonArgs, Munch
from GESS.utils.url import decide_download, download_url, extract_zip
from GESS.utils import get_random_idx_split
from GESS import register
import os
from pathlib import Path
import torch
import re
import pickle
import pandas as pd
import random



domain_mapping = {'pbe': 0, 'hle17': 1, 'hse06_10hf': 2, 'hse06': 3}

@register.dataset_register
class QMOF(InMemoryDataset):
    """
    Object for obtaining QMOF Dataset (Materials Science) with its distribution shifts.
        As for QMOF, our raw data are sourced from https://github.com/Andrew-S-Rosen/QMOF and
        https://doi.org/10.6084/m9.figshare.13147324.
    """

    def __init__(self, root: str, dataset_config: Union[CommonArgs, Munch]):
        self.url_processed = "https://zenodo.org/record/10070680/files/QMOF_processed.zip"
        self.split = dataset_config.extra.split  # Only used in processing raw files. So ignore it.
        self.targ_method = dataset_config.target  # hse06_10hf
        self.high_fi_list = ['hle17', 'hse06_10hf', 'hse06']
        assert self.targ_method in self.high_fi_list
        self.atom_types = dataset_config.extra.atom_types
        self.measure = dataset_config.extra.measure
        self.setting = dataset_config.setting
        if self.setting == 'Par-Label':
            self.restrict_TL_train = dataset_config.OOD_labels
            self.setting = f'Par-Label_#{self.restrict_TL_train}'

        super().__init__(root)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[1]
        self.pos_dim = self.data.pos.shape[1]
        self.feature_type = dataset_config.feature_type
        self.dataset_name = 'QMOF'

        node_categorical_feat = [self.atom_types]
        if self.feature_type == 'only_pos':
            node_scalar_feat = self.pos_dim
            node_categorical_feat = []
        elif self.feature_type == 'only_x':
            node_scalar_feat = self.x_dim - 1
        elif self.feature_type == 'only_ones':
            node_scalar_feat = 1
            node_categorical_feat = []
        else:
            assert self.feature_type == 'both_x_pos'
            node_scalar_feat = self.x_dim - 1 + self.pos_dim

        self.feat_info = {'node_categorical_feat': node_categorical_feat, 'node_scalar_feat': node_scalar_feat}
        self.post_processes()

    @property
    def processed_file_names(self):
        return [f'{self.setting}_{self.measure}_target_{self.targ_method}.pt']

    def post_processes(self):
        self.data.y = self.data.y.view(-1, 1)
    
    def process(self):
        print("our raw data are sourced from https://doi.org/10.6084/m9.figshare.13147324. Download the file "
              "`qmof_database` into the raw_dir ../dataset/QMOF/raw/ if you need (not necessary)")
        if decide_download(self.url_processed, is_raw=False):
            path = download_url(self.url_processed, self.root)
            extract_zip(path, self.root)
            os.unlink(path)
            return
        base_dir = Path(self.raw_dir) / 'qmof_database'
        structure_dir = base_dir / 'relaxed_structures'
        try:
            data = pd.read_csv(base_dir / 'qmof.csv', low_memory=False)
        except FileNotFoundError as e:
            print(e)
            print("raw files not found!")
            print("our raw data are sourced from https://doi.org/10.6084/m9.figshare.13147324. Download the file "
                  "`qmof_database` into the raw_dir ../dataset/QMOF/raw/ if you need.")
            exit(-1)
        files = [x for x in structure_dir.glob('**/*') if x.is_file()]
        # source_files & targ_files
        random.seed(0)
        self.high_fi_list.remove(self.targ_method)
        assert len(self.high_fi_list) == 2
        high_fi1 = self.high_fi_list[0]
        high_fi2 = self.high_fi_list[1]

        target_id = random.sample(self.get_qmof_id(data, method=self.targ_method), 6000)
        high_fi1_id = random.sample(self.get_qmof_id(data, method=high_fi1, del_list=target_id), 2000)
        high_fi2_id = random.sample(self.get_qmof_id(data, method=high_fi2, del_list=target_id + high_fi1_id), 2000)
        pbe_id = self.get_qmof_id(data, method='pbe', del_list=target_id + high_fi1_id + high_fi2_id)

        pbe_files = [x for x in files if x.stem in pbe_id]
        high_fi1_file = [x for x in files if x.stem in high_fi1_id]
        high_fi2_file = [x for x in files if x.stem in high_fi2_id]
        targ_file = [x for x in files if x.stem in target_id]

        target_data_list = []
        for item in targ_file:
            target_data_list.append(extract_data(data, item, method=self.targ_method, measure=self.measure))

        if self.setting.split('_')[0] == "Par-Label":
            targ_data, targ_slices = self.collate(target_data_list)
            targ_train_idx = random.sample(range(4000, 6000), self.restrict_TL_train)
            targ_idx_split = {'train': targ_train_idx, 'val': range(2000), 'test': range(2000, 4000)}
            torch.save((targ_data, targ_slices, targ_idx_split), self.processed_paths[0])

        # build source dataset
        else:
            idx_split = dict()
            Dataset = []
            dataset_dict_ = dict()
            dataset_dict = {'train': [], 'iid_val': [], 'iid_test': [], 'ood_val': [], 'ood_test': []}
            dataset_dict['ood_val'], dataset_dict['ood_test'] = target_data_list[:2000], target_data_list[2000:4000]

            pbe_idx_split = get_random_idx_split(len(pbe_id), self.split, 0)
            high_fi_idx_split = get_random_idx_split(2000, self.split, 0)

            for item in ['train', 'iid_val', 'iid_test']:
                item_ = item.split('_')[1] if len(item.split('_')) > 1 else item
                for idx in pbe_idx_split[item_]:
                    dataset_dict[item].append(
                        extract_data(data, pbe_files[idx], method='pbe', measure=self.measure))

                for idx in high_fi_idx_split[item_]:
                    dataset_dict[item].append(
                        extract_data(data, high_fi1_file[idx], method=high_fi1, measure=self.measure))
                    dataset_dict[item].append(
                        extract_data(data, high_fi2_file[idx], method=high_fi2, measure=self.measure))

            if self.setting.split('_')[0] == "No-Info":
                dataset_dict_ = dataset_dict
            elif self.setting.split('_')[0] == "O-Feature":
                dataset_dict_ = {'train_source': dataset_dict['train'],
                                 'train_target': target_data_list,
                                 'iid_val': dataset_dict['iid_val'],
                                 'iid_test': dataset_dict['iid_test'],
                                 'ood_val': dataset_dict['ood_val'],
                                 'ood_test': dataset_dict['ood_test']}

            for item in dataset_dict_.keys():
                idx_split[item] = [i + len(Dataset) for i in range(len(dataset_dict_[item]))]
                Dataset += dataset_dict_[item]
            data_, slices = self.collate(Dataset)
            torch.save((data_, slices, idx_split), self.processed_paths[0])

    def get_qmof_id(self, mat_data, method, del_list=None):
        # 'hle17', 'hse06_10hf', 'hse06'
        method_col = mat_data[f'outputs.{method}.{self.measure}']
        method_idx = method_col[method_col.notnull()].index.tolist()

        id_list = mat_data.iloc[method_idx, :]['qmof_id'].tolist()
        if del_list is None:
            return id_list
        else:
            return [x for x in id_list if x not in del_list]


def extract_all_num(line):
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', line.split('  ')[1])
    return int(matches[0][1]) + 1


def extract_data(mat_data, file_path, method, measure, allocated_num=79):
    file_name = file_path.stem
    lines = open(file_path).readlines()
    all_atoms = extract_all_num(lines[-1])
    node_emb_list, pos_list = [], []

    atom_type_embed = pickle.load(open(file_path.parent.parent / 'type_dict.pkl', 'rb'))

    for line in reversed(lines):
        if line[0] == ' ':
            break
        split_line = line.split('  ')
        node_emb_list.append(torch.tensor(atom_type_embed[split_line[0]]).reshape(1, -1))
        pos_list.append(
            torch.tensor([float(split_line[3]), float(split_line[4]), float(split_line[5])]).reshape(1, -1))

    node_emb = torch.concat(node_emb_list, dim=0)
    pos = torch.concat(pos_list, dim=0)
    assert node_emb.shape[0] == pos.shape[0] == all_atoms

    record = mat_data[mat_data['qmof_id'] == file_name]

    bandgap = record[f'outputs.{method}.{measure}'].item()
    assert bandgap is not None

    return Data(x=node_emb, pos=pos, y=bandgap, fidelity=method, measure=measure, domain_id=domain_mapping[method])
