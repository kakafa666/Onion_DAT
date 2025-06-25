from __future__ import annotations

import os
import warnings

import matgl
import numpy as np
import torch
from matgl.layers import BondExpansion

from load_datasets import get_val_test_dataset
from my_megnet import MEGNet

warnings.simplefilter("ignore")

# 使用本机上的第二个GPU运行
os.environ["CUDA_VISIBLE_DEVICES"] = "-1,1"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def delete_cache():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json","train_dgl_graph.bin", "train_lattice.pt", "train_dgl_line_graph.bin", "train_state_attr.pt", "train_labels.json","val_dgl_graph.bin", "val_lattice.pt", "val_dgl_line_graph.bin", "val_state_attr.pt", "val_labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    print("删除缓存完成")


def get_mae_mse(model, test_x, test_y):
    uncertainties = []
    targets = []

    for s in test_x:
        u,gap_pred = model.predict_structure(s)
        uncertainties.append(u)
        targets.append(gap_pred)

    uncertainties = torch.tensor(uncertainties)
    targets = torch.tensor(targets)
    test_y = torch.tensor(test_y)

    # mae mse
    mae = torch.mean(torch.abs(targets - test_y))  # 将 test_y 转换为张量
    mse = torch.mean((targets - test_y) ** 2)

    return mae, mse, targets, uncertainties


def predict_p_and_u(name):
    _,_,test_x, test_y = get_val_test_dataset()
    # test_y = torch.tensor(test_y)

    # bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    # model_t = MEGNet(
    #     dim_node_embedding=64,
    #     dim_edge_embedding=100,
    #     dim_state_embedding=2,
    #     nblocks=3,
    #     hidden_layer_sizes_input=(64, 32),
    #     hidden_layer_sizes_conv=(64, 32, 32),
    #     nlayers_set2set=1,
    #     niters_set2set=3,
    #     hidden_layer_sizes_output=(32, 16),
    #     is_classification=False,
    #     activation_type="softplus",
    #     bond_expansion=bond_expansion,
    #     cutoff=5.0,
    #     gauss_width=0.4,
    # )

    print(f"{name}:")

    model = matgl.load_model(path=f"./saved_models/{name}")
    mae, mse, y_hat, u = get_mae_mse(model, test_x, test_y)

    uncertainty_4_list.append(u)
    targets_4_list.append(y_hat)

    print(f"第C列->MAE:{mae.item():.3f} 第D列->MSE:{mse.item():.3f}")


# G H I J 列 加权/非加权融合值
def get_both_mae_mse_integrated():
    print(f"================================= G H I J 列 start  ======================================")
    # 计算1352组归一化权重
    temp_list = uncertainty_4_list.copy()  # 4*1352 不确定度
    weighted_list = []  # 存放1352组权重(1352个list）->内部list(4个权重)
    for j in range(len(temp_list[0])):  # 1352
        sub_list = []  # 存放31个对应u的倒数
        for i in range(len(temp_list)):  # 4
            sub_list.append(1 / temp_list[i][j])
        sum_reciprocals = sum(sub_list)
        weights = [r / sum_reciprocals for r in sub_list]  # 归一化处理
        weighted_list.append(weights)
    # 计算mae，mse
    temp_targets_list = targets_4_list.copy()  # 4*1352预测值
    mae_all = []  # 存放1352个绝对误差
    mse_all = []  # 存放1352个均方差
    mae_all_no_weighted = [] # 非加权
    mse_all_no_weighted = []
    for j in range(len(temp_targets_list[0])):  # 1352
        ae_4_temp = 0
        ae_4_no_weighted_temp=0
        for i in range(len(temp_targets_list)):  # 4
            ae_4_temp += temp_targets_list[i][j] * weighted_list[j][i]
            ae_4_no_weighted_temp += temp_targets_list[i][j]*(1/len(temp_targets_list))
        ae_4 = abs(ae_4_temp - test_y_real[j])
        se_4 = (ae_4_temp - test_y_real[j]) ** 2
        ae_4_no_weighted = abs(ae_4_no_weighted_temp - test_y_real[j])
        se_4_no_weighted = (ae_4_no_weighted_temp - test_y_real[j]) ** 2
        mae_all.append(ae_4)
        mse_all.append(se_4)
        mae_all_no_weighted.append(ae_4_no_weighted)
        mse_all_no_weighted.append(se_4_no_weighted)
    mae_weighted_ave = np.mean(mae_all)
    mse_weighted_ave = np.mean(mse_all)
    mae_no_weighted_ave = np.mean(mae_all_no_weighted)
    mse_no_weighted_ave = np.mean(mse_all_no_weighted)

    print(f"第G列->mae: {mae_no_weighted_ave:.3f}")
    print(f"第H列->mse: {mse_no_weighted_ave:.3f}")
    print(f"第I列->加权mae: {mae_weighted_ave:.3f}")
    print(f"第J列->加权mse: {mse_weighted_ave:.3f}")

    print(f"================================= G H I J 列  END  ======================================")

    return 0


all_datasets_name = ['scan','hse', 'pbe','gllb-sc']

_,_,_, test_y_real = get_val_test_dataset()  # test_y_real:真实值 1352个（4组中对应序号的真实值都相同）
targets_4_list = []  # 存放4个组的预测值 4*1352个
uncertainty_4_list = []  # 存放4个组的测试数据的不确定度;  4*1352个


for name in all_datasets_name:
    delete_cache()
    predict_p_and_u(name)


get_both_mae_mse_integrated()