from __future__ import annotations

import os
import warnings
import gc
import numpy as np
import torch
from matgl.config import DEFAULT_ELEMENTS
from matgl.layers import BondExpansion
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from models_megnet import FeatureExtractor
from load_datasets import get_low_fidelity_dataset, get_val_test_dataset
from pymatgen.core import Element

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

init_seed = 42
torch.manual_seed(init_seed)
# torch.cuda.manual_seed(init_seed)
# torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_my_model(name):

    # converter_inputs = structures
    # elem_list = get_element_list(converter_inputs)
    # converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=5.0)
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    # 实例化 FeatureExtractor 类
    feature_extractor = FeatureExtractor(
        dim_node_embedding=64,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 32, 32),
        nlayers_set2set=1,
        niters_set2set=3,
        activation_type="softplus",
        include_state=True,  # 默认值
        dropout=0.0,  # 默认值
        element_types=DEFAULT_ELEMENTS,  # 使用相同的元素列表
        bond_expansion=bond_expansion,
        cutoff=5.0,
        gauss_width=0.4
    )


    # # 加载检查点,记得根据实际的文件名加载
    # checkpoint_path = f"./checkpoints/{name}/epoch=382.ckpt"
    # checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))

    # 构建检查点目录的路径
    checkpoint_dir = f"./checkpoints/{name}"
    # 获取该目录下所有的文件
    files = os.listdir(checkpoint_dir)
    # 筛选出扩展名为.ckpt的文件
    ckpt_files = [file for file in files if file.endswith('.ckpt')]
    # 检查是否只有一个.ckpt文件
    if len(ckpt_files) == 1:
        # 获取唯一的.ckpt文件的文件名
        ckpt_filename = ckpt_files[0]
        # 构建完整的检查点文件路径
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_filename)
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print("检查点加载成功！")
    else:
        print("该目录下没有.ckpt文件或有多个.ckpt文件，请检查。")

    # 提取完整的 state_dict
    state_dict = checkpoint['state_dict']

    # # 打印 state_dict 的所有键
    # print(checkpoint['state_dict'].keys())

    # 提取与 feature_extractor 相关的参数，跳过 embedding.layer_node_embedding.weight
    feature_extractor_state_dict = {
        k.replace('feature_extractor.', ''): v
        for k, v in state_dict.items()
        if k.startswith('feature_extractor.') and k != 'feature_extractor.embedding.layer_node_embedding.weight'
    }

    # 加载 feature_extractor 的参数
    feature_extractor.load_state_dict(feature_extractor_state_dict, strict=False)

    print("FeatureExtractor模型参数加载成功！")

    return feature_extractor



def get_train_datasets_descriptors(name):
    train_structures, train_targets = get_low_fidelity_dataset(name)
    # 加载训练好的模型
    model_f = load_my_model(name)

    file_path = f"./saved_descriptors/{name}/train_data.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {'targets': train_targets}
    train_descriptors = []
    # train_targets_batch = []
    for s in train_structures:
        features = model_f.get_features(s)
        train_descriptors.append(features)

    train_data_dict['descriptors'] = train_descriptors
    torch.save(train_data_dict, file_path)
    print(f"train_data_dict:{name}保存完成！", len(train_descriptors), len(train_targets))

    return 0

def get_val_datasets_descriptors(name):
    val_structures, val_targets,_,_ = get_val_test_dataset()
    # 加载训练好的模型
    model_f = load_my_model(name)

    file_path = f"./saved_descriptors/{name}/val_data.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    val_data_dict = {'targets': val_targets}
    val_descriptors = []

    for s in val_structures:
        features = model_f.get_features(s)
        val_descriptors.append(features)

    val_data_dict['descriptors'] = val_descriptors
    torch.save(val_data_dict, file_path)
    print(f"val_data_dict:{name}保存完成！", len(val_descriptors), len(val_targets))

    return 0

def get_test_datasets_descriptors(name):
    _, _,test_structures,test_targets = get_val_test_dataset()
    # 加载训练好的模型
    model_f = load_my_model(name)

    file_path = f"./saved_descriptors/{name}/test_data.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    test_data_dict = {'targets': test_targets}
    test_descriptors = []
    # train_targets_batch = []
    for s in test_structures:
        features = model_f.get_features(s)
        test_descriptors.append(features)

    test_data_dict['descriptors'] = test_descriptors
    torch.save(test_data_dict, file_path)
    print(f"val_data_dict:{name}保存完成！", len(test_descriptors), len(test_targets))

    return 0


all_datasets_name = ['scan','hse', 'pbe','gllb-sc']
for name in all_datasets_name:
    get_val_datasets_descriptors(name)
    get_test_datasets_descriptors(name)

datasets_name = ['scan','hse','gllb-sc']
for name in datasets_name:
    get_train_datasets_descriptors(name)







