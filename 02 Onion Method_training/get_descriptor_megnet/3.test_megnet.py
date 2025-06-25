from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from matgl.layers import BondExpansion

import my_megnet_1 as megnet
from load_datasets import get_train_val_test_dataset
from my_megnet import MEGNet

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1,0"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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


def get_test_results():
    # 加载训练好的模型
    megnet_model = load_my_model()

    megnet.now_descriptor = []

    _,_,_,_,test_structures, test_targets = get_train_val_test_dataset()
    pre_targets = []
    for s in test_structures:
        pre_target = megnet_model.predict_structure(s)
        pre_targets.append(pre_target)

    pre_targets_np = np.array(pre_targets)
    test_targets_np = np.array(test_targets)

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    print(f" 第C列->MAE: {mae:.3f}, 第D列->MSE: {mse:.3f}")
    return 0
    #return mae, mse



get_test_results()

