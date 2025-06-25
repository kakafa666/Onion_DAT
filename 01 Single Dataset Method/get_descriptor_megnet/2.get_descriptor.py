from __future__ import annotations

import os
import warnings

import torch
from matgl.layers import BondExpansion

import my_megnet as megnet
from my_megnet import MEGNet

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1,0"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from load_datasets import get_sampled_low_fidelity_dataset, get_low_fidelity_dataset, get_val_test_dataset


def delete_cache():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json","train_dgl_graph.bin", "train_lattice.pt", "train_dgl_line_graph.bin", "train_state_attr.pt", "train_labels.json","val_dgl_graph.bin", "val_lattice.pt", "val_dgl_line_graph.bin", "val_state_attr.pt", "val_labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    print("删除缓存完成")

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


def get_train_datasets_descriptors(name):
    delete_cache()
    megnet_model = load_my_model(name)

    megnet.now_descriptor = []

    train_structures, train_targets = get_low_fidelity_dataset(name)

    # 通过预测方法得到数据集所对应的描述子
    for s in train_structures:
        train_target = megnet_model.predict_structure(s)

    train_descriptors = megnet.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}/train_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {
        'descriptors': train_descriptors,
        'targets': train_targets
    }
    torch.save(train_data_dict, file_path)

    print(f"train_data_dict:{name}保存完成！", len(train_descriptors), len(train_targets))
    delete_cache()

    return 0


def get_val_datasets_descriptors(name):
    delete_cache()
    # 加载训练好的模型
    megnet_model = load_my_model(name)

    megnet.now_descriptor = []

    val_structures, val_targets,_,_ = get_val_test_dataset()

    for s in val_structures:
        target = megnet_model.predict_structure(s)

    val_descriptors = megnet.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}/val_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    val_data_dict = {
        'descriptors': val_descriptors,
        'targets': val_targets
    }
    torch.save(val_data_dict, file_path)

    print(f"test_data_dict:{name}保存完成！", len(val_descriptors), len(val_targets))
    delete_cache()
    return 0


def get_test_datasets_descriptors(name):
    delete_cache()
    # 加载训练好的模型
    megnet_model = load_my_model(name)

    megnet.now_descriptor = []

    _,_,test_structures, test_targets = get_val_test_dataset()

    for s in test_structures:
        target = megnet_model.predict_structure(s)

    test_descriptors = megnet.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}/test_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    test_data_dict = {
        'descriptors': test_descriptors,
        'targets': test_targets
    }
    torch.save(test_data_dict, file_path)

    print(f"test_data_dict:{name}保存完成！", len(test_descriptors), len(test_targets))
    delete_cache()

    return 0


datasets_name = ['pbe']

# for name in datasets_name:
#     get_train_datasets_descriptors(name)
#     get_val_datasets_descriptors(name)
#     get_test_datasets_descriptors(name)


get_val_datasets_descriptors('pbe')
get_test_datasets_descriptors('pbe')

