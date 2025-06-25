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

    targets_4_list.append(pre_targets)

    pre_targets_np = np.array(pre_targets)
    test_targets_np = np.array(test_targets)

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    print(f"{name}  MAE: {mae}, MSE: {mse}")

    return 0


# G H列 非加权融合值
def get_mae_mse_integrated():
    print(f"================================= G H 列 start  ======================================")
    # 计算综合mae，mse值
    temp_targets_list = targets_4_list.copy()  # 4*1352预测值
    mae_all_no_weighted = [] # 非加权
    mse_all_no_weighted = []
    for j in range(len(temp_targets_list[0])):  # 1352
        ae_4_no_weighted_temp=0
        for i in range(len(temp_targets_list)):  # 4
            ae_4_no_weighted_temp += temp_targets_list[i][j]*(1/len(temp_targets_list))
        ae_4_no_weighted = abs(ae_4_no_weighted_temp - test_y_real[j])
        se_4_no_weighted = (ae_4_no_weighted_temp - test_y_real[j]) ** 2
        mae_all_no_weighted.append(ae_4_no_weighted)
        mse_all_no_weighted.append(se_4_no_weighted)
    mae_no_weighted_ave = np.mean(mae_all_no_weighted)
    mse_no_weighted_ave = np.mean(mse_all_no_weighted)

    print(f"第G列->mae: {mae_no_weighted_ave:.3f}")
    print(f"第H列->mse: {mse_no_weighted_ave:.3f}")

    print(f"================================= G H 列  END  ======================================")

    return 0


_,_,_, test_y_real = get_val_test_dataset()  # test_y_real:真实值 1352个
targets_4_list = []  # 存放4个组的预测值 4*1352个


# , 'hse', 'gllb-sc', 'pbe'
all_datasets_name = ['scan','hse', 'gllb-sc', 'pbe']

for name in all_datasets_name:
    get_test_results(name)


get_mae_mse_integrated()