from __future__ import annotations

import os
import warnings

import torch
from matgl.layers import BondExpansion
from torch.utils.data import DataLoader

import my_megnet as megnet
from my_megnet import MEGNet

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1,1"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from load_datasets import get_low_fidelity_dataset, get_train_val_test_dataset



def custom_collate_fn(batch):
    result = []
    for item in batch:
        result.append(item)
    return result


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

# def get_train_datasets_descriptors(name, seed):
#     megnet_model = load_my_model(name, seed)
#     megnet.now_descriptor = []
#     train_structures, train_targets = get_low_fidelity_dataset(name)
#     # 通过预测方法得到数据集所对应的描述子
#     for s in train_structures:
#         train_target = megnet_model.predict_structure(s)
#     train_descriptors = megnet.now_descriptor.copy()
#     file_path = f"./saved_descriptors/{name}_{seed}/train_data.pt"
#     # 创建保存路径的父目录
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     train_data_dict = {
#         'descriptors': train_descriptors,
#         'targets': train_targets
#     }
#     torch.save(train_data_dict, file_path)
#     print(f"train_data_dict:{name}_{seed}保存完成！", len(train_descriptors), len(train_targets))

def get_train_datasets_descriptors(name):
    megnet_model = load_my_model()
    megnet.now_descriptor = []
    if name == 'exp':
        train_structures, train_targets, _, _, _, _ = get_train_val_test_dataset()
    else:
        train_structures, train_targets = get_low_fidelity_dataset(name)

    train_loader = DataLoader(train_structures,
                              batch_size=32,
                              shuffle=False,
                              num_workers = 4,  # 使用多进程数据加载
                              collate_fn=custom_collate_fn)

    file_path = f"./saved_descriptors/{name}/train_data.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {'targets': train_targets}
    train_descriptors = []
    # train_targets_batch = []
    with torch.no_grad():  # 避免存储梯度信息
        for batch in train_loader:
            for s in batch:
                train_target = megnet_model.predict_structure(s)
            train_descriptors.extend(megnet.now_descriptor.copy())
            megnet.now_descriptor.clear()
        train_data_dict['descriptors'] = train_descriptors
        torch.save(train_data_dict, file_path)
        print(f"train_data_dict:{name}保存完成！", len(train_descriptors), len(train_targets))

def get_val_datasets_descriptors():
    # 加载训练好的模型
    megnet_model = load_my_model()

    megnet.now_descriptor = []

    _,_,val_structures, val_targets,_,_ = get_train_val_test_dataset()

    for s in val_structures:
        target = megnet_model.predict_structure(s)

    val_descriptors = megnet.now_descriptor.copy()

    file_path = f"./saved_descriptors/val/val_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    val_data_dict = {
        'descriptors': val_descriptors,
        'targets': val_targets
    }
    torch.save(val_data_dict, file_path)

    print(f"test_data_dict:验证描述子保存完成！", len(val_descriptors), len(val_targets))


def get_test_datasets_descriptors():
    # 加载训练好的模型
    megnet_model = load_my_model()
    megnet.now_descriptor = []
    _,_,_,_,test_structures, test_targets = get_train_val_test_dataset()

    for s in test_structures:
        target = megnet_model.predict_structure(s)

    test_descriptors = megnet.now_descriptor.copy()

    file_path = f"./saved_descriptors/test/test_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    test_data_dict = {
        'descriptors': test_descriptors,
        'targets': test_targets
    }
    torch.save(test_data_dict, file_path)

    print(f"test_data_dict:测试描述子保存完成！", len(test_descriptors), len(test_targets))



all_datasets_name=['exp','gllb-sc', 'hse', 'scan', 'pbe']

get_val_datasets_descriptors()
get_test_datasets_descriptors()

for name in all_datasets_name:
    get_train_datasets_descriptors(name)



