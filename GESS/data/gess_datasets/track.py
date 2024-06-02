import pickle
import numpy as np
from GESS import register
from GESS.utils.utils import get_random_idx_split, get_ood_split
from GESS.utils.url import decide_download, download_url, extract_zip
import os
import os.path as osp
from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset
from GESS.utils.config_process import Union, CommonArgs, Munch


class HEP_OOD_Shift(InMemoryDataset):
    """
    Object for obtaining Track Dataset (HEP) with its distribution shifts.

    """

    def __init__(self, root: str, dataset_config: Union[CommonArgs, Munch], pileup, tesla='2T'):
        self.tesla = tesla
        self.dataset_name = dataset_config.data_name
        self.split = dataset_config.extra.split
        self.id_split = dataset_config.extra.iid_split
        self.dataset_dir = Path(root)
        self.bkg_dir = self.dataset_dir / 'raw' / 'background'
        self.sig_dir = self.dataset_dir / 'raw' / 'z2mu'
        self.pileup = pileup
        self.setting = dataset_config.setting
        if self.setting == 'Par-Label':
            self.restrict_TL_train = dataset_config.OOD_labels
            self.setting = f'Par-Label_#{self.restrict_TL_train}'
        self.shift_type = f"{dataset_config.shift_name}_shift"
        self.target_domain = dataset_config.target
        super().__init__(root)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[1]
        self.pos_dim = self.data.pos.shape[1]
        self.feature_type = dataset_config.feature_type
        if self.feature_type == 'only_pos':
            node_scalar_feat = self.pos_dim
        elif self.feature_type == 'only_x':
            node_scalar_feat = self.x_dim
        elif self.feature_type == 'only_ones':
            node_scalar_feat = 1
        else:
            assert self.feature_type == 'both_x_pos'
            node_scalar_feat = self.x_dim + self.pos_dim

        self.feat_info = {'node_categorical_feat': [], 'node_scalar_feat': node_scalar_feat}

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed')


@register.dataset_register
class Track_Pileup(HEP_OOD_Shift):
    def __init__(self, root: str, dataset_config: Union[CommonArgs, Munch]):
        super(Track_Pileup, self).__init__(root, dataset_config, dataset_config.extra.pileup_train)
        self.url_processed = None
        self.domain_splits = 5  # predefined parameter
        

    @property
    def processed_file_names(self):
        return [f'{self.shift_type}_{self.pileup}_{self.target_domain}_{self.setting}.pt']

    def process(self):
        target_zip_file = f"Track_processed_pileup_shift_{self.pileup}_{self.target_domain}_{self.setting}.zip".replace("#", "")
        self.url_processed = get_url(target_zip_file)
        if decide_download(self.url_processed, is_raw=False):
            path = download_url(self.url_processed, self.root)
            extract_zip(path, self.root)
            os.unlink(path)
            return

        def obtain_list(event_type, pileups):
            # event_type = 'bkg' or 'signal'
            base_dir = self.bkg_dir if event_type == 'bkg' else self.sig_dir
            file_path = base_dir / f'{event_type}_events_{self.tesla}_pileups_{pileups}.pkl'
            get_list = pickle.load(open(file_path, 'rb'))
            return get_list

        try:
            bkg_list = obtain_list('bkg', self.pileup)
        except:
            print("raw files not found!")
            exit(-1)
        sig_list = obtain_list('signal', self.pileup)
        bkg_list_ood = obtain_list('bkg', self.target_domain)
        sig_list_ood = obtain_list('signal', self.target_domain)

        bkg_split = get_random_idx_split(len(bkg_list), self.id_split, 0)
        sig_split = get_random_idx_split(len(sig_list), self.id_split, 0)

        idx_split, Dataset = {}, []
        if self.setting.split('_')[0] in ["No-Info", "O-Feature"]:
            dataset_dict = {'train': [], 'iid_val': [], 'iid_test': [], 'ood_val': [], 'ood_test': []}
            dataset_dict_ = dict()
            bkg_min, bkg_max = get_size_extremum(bkg_list)
            sig_min, sig_max = get_size_extremum(sig_list)
            bkg_bins = np.linspace(start=bkg_min, stop=bkg_max, num=self.domain_splits + 1)
            sig_bins = np.linspace(start=sig_min, stop=sig_max, num=self.domain_splits + 1)

            # in-distribution dataset
            for item in ['train', 'iid_val', 'iid_test']:
                item_ = item.split('_')[1] if len(item.split('_')) > 1 else item
                for idx in bkg_split[item_]:
                    data = bkg_list[idx]
                    domain_id = np.digitize(data[0].shape[0], bkg_bins) - 1
                    dataset_dict[item].append(
                        build_data_object(data, signal=False, event_type='bkg', domain_id=domain_id))
                for idx in sig_split[item_]:
                    data = sig_list[idx]
                    domain_id = np.digitize(data[0].shape[0], sig_bins) - 1
                    dataset_dict[item].append(
                        build_data_object(data, signal=True, event_type='z2mu', domain_id=domain_id))

            # out-of-distribution dataset, with 2500 `bkg` and `sig` data respectively
            train_target_list = []
            data_num = 10000 if self.target_domain == 50 else 7700
            for idx in range(data_num):
                bkg_data_obj = build_data_object(bkg_list_ood[idx], signal=False, event_type='bkg')
                sig_data_obj = build_data_object(sig_list_ood[idx], signal=True, event_type='z2mu')
                train_target_list.append(bkg_data_obj)
                train_target_list.append(sig_data_obj)
                if 0 <= idx < 2500:
                    dataset_dict['ood_val'].append(bkg_data_obj)
                    dataset_dict['ood_val'].append(sig_data_obj)
                elif 2500 <= idx < 5000:
                    dataset_dict['ood_test'].append(bkg_data_obj)
                    dataset_dict['ood_test'].append(sig_data_obj)
            if self.setting == 'No-Info':
                dataset_dict_ = dataset_dict
            elif self.setting == 'O-Feature':
                dataset_dict_ = {'train_source': dataset_dict['train'],
                                 'train_target': train_target_list,
                                 'iid_val': dataset_dict['iid_val'],
                                 'iid_test': dataset_dict['iid_test'],
                                 'ood_val': dataset_dict['ood_val'],
                                 'ood_test': dataset_dict['ood_test']}
            for item in dataset_dict_.keys():
                idx_split[item] = [i + len(Dataset) for i in range(len(dataset_dict_[item]))]
                Dataset += dataset_dict_[item]

        elif self.setting.split('_')[0] == "Par-Label":
            dataset_dict = {'train': [], 'val': [], 'test': []}
            for idx in range(2500):
                dataset_dict['val'].append(
                    build_data_object(bkg_list_ood[idx], signal=False, event_type='bkg'))
                dataset_dict['val'].append(
                    build_data_object(sig_list_ood[idx], signal=True,
                                      event_type='z2mu'))
            for idx in range(2500, 5000):
                dataset_dict['test'].append(
                    build_data_object(bkg_list_ood[idx], signal=False, event_type='bkg'))
                dataset_dict['test'].append(
                    build_data_object(sig_list_ood[idx], signal=True,
                                      event_type='z2mu'))
            for idx in range(5000, 5000 + int(self.restrict_TL_train / 2)):
                dataset_dict['train'].append(
                    build_data_object(bkg_list_ood[idx], signal=False, event_type='bkg'))
                dataset_dict['train'].append(
                    build_data_object(sig_list_ood[idx], signal=True,
                                      event_type='z2mu'))
            for item in dataset_dict.keys():
                idx_split[item] = [i + len(Dataset) for i in range(len(dataset_dict[item]))]
                Dataset += dataset_dict[item]

        data, slices = self.collate(Dataset)
        torch.save((data, slices, idx_split), self.processed_paths[0])


@register.dataset_register
class Track_Signal(HEP_OOD_Shift):
    def __init__(self, root: str, dataset_config: Union[CommonArgs, Munch]):
        super(Track_Signal, self).__init__(root, dataset_config, dataset_config.extra.pileup)
        self.url_processed = None
        self.dataset_dir = Path(root)
        self.z_dir = self.dataset_dir / 'raw' / 'z2mu'
        self.tau_dir = self.dataset_dir / 'raw' / 'tau3mu'
        self.z_prime_dir = self.dataset_dir / 'raw' / 'z\'2mu'
        

    @property
    def processed_file_names(self):
        return [f'{self.shift_type}_pileup_{self.pileup}_{self.setting}_target_{self.target_domain}.pt']

    def process(self):
        target_zip_file = f"Track_processed_signal_shift_pileup_10_{self.setting}_target_{self.target_domain}.zip".replace(
            "#", "")
        self.url_processed = get_url(target_zip_file)
        if decide_download(self.url_processed, is_raw=False):
            path = download_url(self.url_processed, self.root)
            extract_zip(path, self.root)
            os.unlink(path)
            return
        # for source domain, we have z, z'80/70/60/50, 5 domain in total
        # used for source domain
        signal_z_path = self.z_dir / f'signal_events_{self.tesla}_pileups_{self.pileup}.pkl'
        signal_z_p_80 = self.z_prime_dir / f'signal_events_m0_80_pileups_{self.pileup}.pkl'
        signal_z_p_70 = self.z_prime_dir / f'signal_events_m0_70_pileups_{self.pileup}.pkl'
        signal_z_p_60 = self.z_prime_dir / f'signal_events_m0_60_pileups_{self.pileup}.pkl'
        signal_z_p_50 = self.z_prime_dir / f'signal_events_m0_50_pileups_{self.pileup}.pkl'

        background_path = self.bkg_dir / f'bkg_events_{self.tesla}_pileups_{self.pileup}.pkl'
        bkg_path_for_train = self.bkg_dir / f'bkg_events_{self.tesla}_pileups_{self.pileup}_for_train.pkl'
        # used for target domain
        target_info = self.target_domain.split("_")
        target_signal = self.tau_dir / f'signal_events_{self.tesla}_pileups_{self.pileup}.pkl' if len(
            target_info) == 1 else \
            self.z_prime_dir / f'signal_events_m0_{target_info[1]}_pileups_{self.pileup}.pkl'
        try:
            bkg_list = pickle.load(open(background_path, 'rb'))
        except:
            print("raw files not found!")
            exit(-1)
        z_list = pickle.load(open(signal_z_path, 'rb'))
        z_p_80_list = pickle.load(open(signal_z_p_80, 'rb'))
        z_p_70_list = pickle.load(open(signal_z_p_70, 'rb'))
        z_p_60_list = pickle.load(open(signal_z_p_60, 'rb'))
        z_p_50_list = pickle.load(open(signal_z_p_50, 'rb'))
        target_list = pickle.load(open(target_signal, 'rb'))
        bkg_list_for_train = pickle.load(open(bkg_path_for_train, 'rb'))

        bkg_split = get_ood_split(len(bkg_list), self.split, 0)
        source_split = get_random_idx_split(4000, self.id_split, 0)  # shared within 5 domains: `0`, `1`,...,`4`
        target_split = get_random_idx_split(20000, self.id_split,
                                            0)  # `4k` and `20k` are specially designed to avoid label shift.
        idx_split = dict()
        Dataset = []

        if self.setting.split('_')[0] in ["No-Info", "O-Feature"]:
            # train, id_valid, id_test --> just z2mu
            # ood_valid, ood_test --> just tau3mu
            # (of course background events are included.)
            # load background events
            dataset_dict = {'train': [], 'iid_val': [], 'iid_test': [], 'ood_val': [], 'ood_test': []}
            dataset_dict_ = dict()
            for item in dataset_dict.keys():
                for idx in bkg_split[item]:
                    event_bkg = bkg_list[idx]
                    dataset_dict[item].append(build_data_object(event_bkg, signal=False, event_type='bkg'))
            # load z2mu events
            for item in ['train', 'iid_val', 'iid_test']:
                item_ = item.split('_')[1] if len(item.split('_')) > 1 else item
                for idx in source_split[item_]:
                    dataset_dict[item].append(
                        build_data_object(z_list[idx], signal=True, event_type='z2mu', domain_id=0))
                    dataset_dict[item].append(
                        build_data_object(z_p_80_list[idx], signal=True, event_type='z\'2mu', domain_id=1))
                    dataset_dict[item].append(
                        build_data_object(z_p_70_list[idx], signal=True, event_type='z\'2mu', domain_id=2))
                    dataset_dict[item].append(
                        build_data_object(z_p_60_list[idx], signal=True, event_type='z\'2mu', domain_id=3))
                    dataset_dict[item].append(
                        build_data_object(z_p_50_list[idx], signal=True, event_type='z\'2mu', domain_id=4))

            # load tau3mu events? or z' particles
            for item in ['ood_val', 'ood_test']:
                item_ = item.split('_')[1]
                for idx in target_split[item_]:
                    tau_sig_event = target_list[idx]
                    dataset_dict[item].append(build_data_object(tau_sig_event, signal=True, event_type='tau3mu'))
            if self.setting == 'No-Info':
                dataset_dict_ = dataset_dict
            elif self.setting == 'O-Feature':
                dataset_dict_ = {'train_source': dataset_dict['train'],
                                 'train_target': [],
                                 'iid_val': dataset_dict['iid_val'],
                                 'iid_test': dataset_dict['iid_test'],
                                 'ood_val': dataset_dict['ood_val'],
                                 'ood_test': dataset_dict['ood_test']}
                for idx in range(12000):
                    dataset_dict_['train_target'].append(
                        build_data_object(bkg_list_for_train[idx], signal=True, event_type='tau3mu'))
                for idx in range(15000):
                    dataset_dict_['train_target'].append(
                        build_data_object(target_list[idx], signal=True, event_type='tau3mu'))

            for item in dataset_dict_.keys():
                idx_split[item] = [i + len(Dataset) for i in range(len(dataset_dict_[item]))]
                Dataset += dataset_dict_[item]

        elif self.setting.split('_')[0] == "Par-Label":
            # `train`, `valid` and `test` all from target domain
            # num of training samples is a factor
            idx_split = dict()
            Dataset = []
            target_list_ = [build_data_object(target, signal=True, event_type='tau3mu') for target in target_list]
            bkg_list_ = [build_data_object(bkg, signal=False, event_type='bkg') for bkg in bkg_list]
            bkg_list_for_train_ = [build_data_object(bkg, signal=False, event_type='bkg') for bkg in bkg_list_for_train]
            TL_split = get_random_idx_split(20000, self.id_split, 0, restrict_training=int(self.restrict_TL_train / 2))
            dataset_dict = {'train': [], 'val': [], 'test': []}
            for item in dataset_dict.keys():
                for idx in TL_split[item]:
                    dataset_dict[item].append(target_list_[idx])
            for item in ['val', 'test']:
                for idx in bkg_split[f'ood_{item}']:
                    dataset_dict[item].append(bkg_list_[idx])
            for idx in range(int(self.restrict_TL_train / 2)):
                dataset_dict['train'].append(bkg_list_for_train_[idx])

            for item in dataset_dict.keys():
                idx_split[item] = [i + len(Dataset) for i in range(len(dataset_dict[item]))]
                Dataset += dataset_dict[item]
        data, slices = self.collate(Dataset)
        torch.save((data, slices, idx_split), self.processed_paths[0])


def build_data_object(event, signal, event_type, domain_id=-1):
    other_features = ['tt', 'tpx', 'tpy', 'tpz', 'te', 'deltapx', 'deltapy', 'deltapz', 'deltae']
    hits = event[0]
    signal_im = event[1]
    domain_id = torch.tensor(domain_id)
    y = torch.tensor(1).float().view(-1, 1) if signal else torch.tensor(0).float().view(-1, 1)

    hits['node_id'] = range(len(hits))
    pos = torch.tensor(hits[['tx', 'ty', 'tz']].to_numpy()).float()
    x = torch.tensor(hits[other_features].to_numpy()).float()
    node_label = torch.tensor(hits['node_label'].to_numpy()).float().view(-1)
    node_dir = torch.tensor(hits[['tpx', 'tpy', 'tpz']].to_numpy()).float()
    track_ids = torch.full((len(hits),), -1)  # indices which track the node belongs to
    num_tracks = 0
    all_ptcls = hits['particle_id'].unique()
    for ptcl in all_ptcls:
        track = hits[hits['particle_id'] == ptcl]
        track_ids[track['node_id'].to_numpy()] = num_tracks
        num_tracks += 1
    assert -1 not in track_ids
    return Data(x=x, pos=pos, y=y, node_label=node_label,
                node_dir=node_dir, num_tracks=num_tracks, track_ids=track_ids, signal_im=signal_im,
                event_type=event_type, domain_id=domain_id)


def get_size_extremum(dataset):
    size_list = []
    for data in dataset:
        size_list.append(data[0].shape[0])
    size_list = np.array(size_list)
    return size_list.min(), size_list.max()


def get_url(target_zip):
    if target_zip in ["Track_processed_pileup_shift_10_50_Par-Label_1000.zip",
                      "Track_processed_pileup_shift_10_50_Par-Label_500.zip",
                      "Track_processed_signal_shift_pileup_10_Par-Label_500_target_tau.zip"]:
        return f"https://zenodo.org/record/10012747/files/{target_zip}"
    else:
        return f"https://zenodo.org/record/10070680/files/{target_zip}"
