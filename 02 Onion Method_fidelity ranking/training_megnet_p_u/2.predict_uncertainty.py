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
    print(f"第C列->MAE:{mae.item():.3f} 第D列->MSE:{mse.item():.3f}")

all_datasets_name = ['scan','hse', 'gllb-sc', 'pbe']

for name in all_datasets_name:
    delete_cache()
    predict_p_and_u(name)

