from __future__ import annotations

import os
import warnings

import matgl
import numpy as np
import torch
from matgl.layers import BondExpansion

from load_datasets import get_train_val_test_dataset
from my_megnet import MEGNet

warnings.simplefilter("ignore")

# 使用本机上的第二个GPU运行
os.environ["CUDA_VISIBLE_DEVICES"] = "-1,0"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def delete_cache():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    print("删除缓存完成")


def load_my_model():
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    model_t = MEGNet(
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

    # 构建模型目录路径
    model_dir = f"./checkpoints/model_5/"
    # 获取该目录下所有的 .ckpt 文件
    ckpt_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]

    # 检查是否只有一个 .ckpt 文件
    if len(ckpt_files) == 1:
        checkpoint_path = os.path.join(model_dir, ckpt_files[0])
    else:
        raise ValueError(f"目录 {model_dir} 下应该只有一个 .ckpt 文件，但找到了 {len(ckpt_files)} 个。")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            new_key = key[6:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    results = model_t.load_state_dict(new_state_dict)

    # # 输出模型的所有参数和细节
    # print("加载的模型参数和细节：")
    # for name, param in model_t.named_parameters():
    #     print(f"{name}: {param.data}")

    print("模型加载成功！")

    return model_t

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


def predict_p_and_u():
    _,_,_,_,test_x, test_y = get_train_val_test_dataset()

    # model_t = matgl.load_model(path="./saved_models/model_5_p_u")

    model_t=load_my_model()
    mae, mse, y_hat, u = get_mae_mse(model_t, test_x, test_y)
    print(f"第C列->MAE:{mae.item():.3f}  第D列->MSE:{mse.item():.3f}")

    return 0


delete_cache()
predict_p_and_u()
delete_cache()


