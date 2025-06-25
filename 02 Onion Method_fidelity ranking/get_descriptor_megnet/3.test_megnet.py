from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from matgl.layers import BondExpansion

import my_megnet as megnet
from load_datasets import get_val_test_dataset
from my_megnet import MEGNet

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1,0"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_my_model(name):
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    model = MEGNet(
        dim_node_embedding=64,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 32, 32),
        nlayers_set2set=1,
        niters_set2set=3,
        hidden_layer_sizes_output=(32, 16),
        is_classification=False,
        activation_type="softplus",
        bond_expansion=bond_expansion,
        cutoff=5.0,
        gauss_width=0.4,
    )

    megnet_model = model.load(path=f"./saved_models/{name}")

    return megnet_model


def get_test_results(name):
    # 加载训练好的模型
    megnet_model = load_my_model(name)

    megnet.now_descriptor = []

    _,_,test_structures, test_targets = get_val_test_dataset()
    pre_targets = []
    for s in test_structures:
        pre_target = megnet_model.predict_structure(s)
        pre_targets.append(pre_target)

    pre_targets_np = np.array(pre_targets)
    test_targets_np = np.array(test_targets)

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    print(f"{name}  MAE: {mae}, MSE: {mse}")

    return 0

# , 'hse', 'gllb-sc', 'pbe'
all_datasets_name = ['hse', 'gllb-sc', 'pbe']

for name in all_datasets_name:
    get_test_results(name)
