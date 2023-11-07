# GDL-DS

This repository contains the official implementation of GDL-DS as described in the paper: [GDL-DS: A Benchmark for Geometric Deep Learning under Distribution Shifts](https://arxiv.org/abs/2310.08677) by Deyu Zou, Shikun Liu, Siqi Miao, Victor Fung, Shiyu Chang, and Pan Li.

## Introduction

We propose **GDL-DS**, a comprehensive benchmark designed for evaluating the performance of geometric deep learning (GDL) models in scenarios where scientific applications encounter distribution shift challenges. Our evaluation datasets cover diverse scientific domains from particle physics and materials science to biochemistry, and encapsulate a broad spectrum of distribution shifts including conditional, covariate, and concept shifts. Furthermore, we study three levels of information access from the out-of-distribution (OOD) testing data, including no OOD information (No-Info), only OOD features without labels (O-Feature), and OOD features with a few labels (Par-Label).   

## Datasets

Figure 1 provides illustrations of some distribution shifts mentioned in this paper. Dataset statistics could be found in our paper. **All processed datasets are available for manual download from** [Zenodo](https://zenodo.org/records/10070680) (Here is another [upload](https://zenodo.org/records/10012747) due to the space limit). For the HEP dataset, we highly recommend using the processed files directly because the raw files would consume a significant amount of disk space and require a longer time for processing. Regarding DrugOOD-3D, in this paper, we utilized three cases of distribution shifts, including `lbap_core_ic50_assay`, `lbap_core_ic50_scaffold`, `lbap_core_ic50_size`,  and we recommend readers to find more details in https://github.com/tencent-ailab/DrugOOD. As for QMOF, our data is sourced from https://github.com/Andrew-S-Rosen/QMOF.

<p align="center"><img src="./dataset/fig1.png" width=85% height=85%></p>
<p align="center"><em>Figure 2.</em> Illustrations of the four scientific datasets in this work to study interpretable GDL models. </p>

## Installation

We have tested our code on `Python 3.9` with `PyTorch 1.12.1`, `PyG 2.2.0` and `CUDA 11.3`. Please follow the following steps to create a virtual environment and install the required packages.

Step 1: Clone the repository

```
git clone https://github.com/Graph-COM/GDL_DS.git
cd GDL_DS
```

Step 2: Create a virtual environment

```
conda create --name GDL_DS python=3.9 -y
conda activate GDL_DS
```

Step 3: Install dependencies

```
conda install -y pytorch==1.12.1 torchvision cudatoolkit=11.3 -c pytorch
pip install torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install -r requirements.txt
```

## Reproducing Results

We train a model by `run.sh` file:

```
cd ./scripts
sh run.sh
```

Specifically, use the following command in this file:

```
python run.py --dataset [dataset_name] --method [method_name] --shift [shift_name] --target [target] --setting [setting_name] --backbone [backbone_name]
```

`dataset_name` can be chosen from `Track`, `DrugOOD-3D`, and `QMOF`, and the dataset specified will be downloaded automatically.

`method_name` can be chosen from `erm`, `lri_bern`, `mixup`, `dir`, `groupdro`, `VREx`, `coral`, `DANN`. 

`shift_name` can be chosen from `pileup` (corresponding to `target` of `50` or `90`), `signal` (corresponding to `target` of `tau`, `zp_10` or `zp_20`), `assay` (corresponding to `target` of `lbap_core_ic50_assay`), `scaffold` (corresponding to `target` of `lbap_core_ic50_scaffold`), `size` (corresponding to `target` of `lbap_core_ic50_size`),  `fidelity` (corresponding to `target` of `hse06` or `hse06_10hf`).

`setting_name` can be chosen from `No-Info`, `O-Feature`, and `Par-Label`.

`backbone_model` can be chosen from `dgcnn`, `pointtrans` and `egnn`.

The tuned hyperparameters in the `egnn` backbone for all distribution shifts and cases can be found in `./src/configs`.

## Reference

If you find our paper and repo useful, please cite our paper:

```
@misc{zou2023gdlds,
      title={GDL-DS: A Benchmark for Geometric Deep Learning under Distribution Shifts}, 
      author={Deyu Zou and Shikun Liu and Siqi Miao and Victor Fung and Shiyu Chang and Pan Li},
      year={2023},
      eprint={2310.08677},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

