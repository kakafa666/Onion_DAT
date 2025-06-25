import gzip
import json
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")


init_seed = 42


# 以下是all_data.json中ordered_exp中对应的其中一个分子的结构数据以及带隙数据
# "ordered_exp":
# {
#         "icsd-44914": {
#             "structure": {
#                 "lattice": {
#                     "a": 5.533,
#                     "c": 5.533,
#                     "b": 5.533,
#                     "matrix": [
#                         [
#                             5.533,
#                             0.0,
#                             3.387985369841153e-16
#                         ],
#                         [
#                             8.89774262066083e-16,
#                             5.533,
#                             3.387985369841153e-16
#                         ],
#                         [
#                             0.0,
#                             0.0,
#                             5.533
#                         ]
#                     ],
#                     "volume": 169.38775443700004,
#                     "beta": 90.0,
#                     "gamma": 90.0,
#                     "alpha": 90.0
#                 },
#                 "charge": null,
#                 "sites": [
#                     {
#                         "abc": [
#                             0.0,
#                             0.0,
#                             0.0
#                         ],
#                         "xyz": [
#                             0.0,
#                             0.0,
#                             0.0
#                         ],
#                         "label": "Lu3+",
#                         "species": [
#                             {
#                                 "occu": 1.0,
#                                 "element": "Lu",
#                                 "oxidation_state": 3.0
#                             }
#                         ]
#                     },
#                     {
#                         "abc": [
#                             0.0,
#                             0.5,
#                             0.5
#                         ],
#                         "xyz": [
#                             4.448871310330415e-16,
#                             2.7665,
#                             2.7665
#                         ],
#                         "label": "Lu3+",
#                         "species": [
#                             {
#                                 "occu": 1.0,
#                                 "element": "Lu",
#                                 "oxidation_state": 3.0
#                             }
#                         ]
#                     },
#                     {
#                         "abc": [
#                             0.5,
#                             0.0,
#                             0.5
#                         ],
#                         "xyz": [
#                             2.7665,
#                             0.0,
#                             2.7665
#                         ],
#                         "label": "Lu3+",
#                         "species": [
#                             {
#                                 "occu": 1.0,
#                                 "element": "Lu",
#                                 "oxidation_state": 3.0
#                             }
#                         ]
#                     },
#                     {
#                         "abc": [
#                             0.5,
#                             0.5,
#                             0.0
#                         ],
#                         "xyz": [
#                             2.7665000000000006,
#                             2.7665,
#                             3.387985369841153e-16
#                         ],
#                         "label": "Lu3+",
#                         "species": [
#                             {
#                                 "occu": 1.0,
#                                 "element": "Lu",
#                                 "oxidation_state": 3.0
#                             }
#                         ]
#                     },
#                     {
#                         "abc": [
#                             0.5,
#                             0.5,
#                             0.5
#                         ],
#                         "xyz": [
#                             2.7665000000000006,
#                             2.7665,
#                             2.7665000000000006
#                         ],
#                         "label": "P3-",
#                         "species": [
#                             {
#                                 "occu": 1.0,
#                                 "element": "P",
#                                 "oxidation_state": -3.0
#                             }
#                         ]
#                     },
#                     {
#                         "abc": [
#                             0.5,
#                             0.0,
#                             0.0
#                         ],
#                         "xyz": [
#                             2.7665,
#                             0.0,
#                             1.6939926849205765e-16
#                         ],
#                         "label": "P3-",
#                         "species": [
#                             {
#                                 "occu": 1.0,
#                                 "element": "P",
#                                 "oxidation_state": -3.0
#                             }
#                         ]
#                     },
#                     {
#                         "abc": [
#                             0.0,
#                             0.5,
#                             0.0
#                         ],
#                         "xyz": [
#                             4.448871310330415e-16,
#                             2.7665,
#                             1.6939926849205765e-16
#                         ],
#                         "label": "P3-",
#                         "species": [
#                             {
#                                 "occu": 1.0,
#                                 "element": "P",
#                                 "oxidation_state": -3.0
#                             }
#                         ]
#                     },
#                     {
#                         "abc": [
#                             0.0,
#                             0.0,
#                             0.5
#                         ],
#                         "xyz": [
#                             0.0,
#                             0.0,
#                             2.7665
#                         ],
#                         "label": "P3-",
#                         "species": [
#                             {
#                                 "occu": 1.0,
#                                 "element": "P",
#                                 "oxidation_state": -3.0
#                             }
#                         ]
#                     }
#                 ],
#                 "@class": "Structure",
#                 "@module": "pymatgen.core.structure"
#             },
#             "icsd_id": 44914,
#             "is_ordered": true,
#             "band_gap": 1.3
#         },
#
#    "icsd-629000":{},
#     "xxx":{},
#     ........
#
# }


# 合并样本量少的层,解决 stratify 分层抽样的层内样本太少而报错的问题
# def update_layers(combined_features_o):
#     # 假设 combined_features_o 是一个二维的列表/数组，其中每个元素是一个包含多个特征的列表
#     # 将其转换为不可变的元组，便于进行统计
#     combined_features = [tuple(feature) for feature in combined_features_o]  # 转换为元组以便统计
#
#     # 统计每个类别的出现次数
#     counter = collections.Counter(combined_features)
#     print("类别出现次数:", counter)
#
#     # 设定一个阈值，假设我们要合并样本数小于2的类别
#     threshold = 2
#     merged_classes = {}
#
#     # 合并样本数小于阈值的类别
#     for feature, count in counter.items():
#         if count < threshold:
#             # 合并到 'other' 类别
#             merged_classes[feature] = 'other'
#         else:
#             merged_classes[feature] = feature
#
#     # 根据合并后的类别构造新的标签，生成一维的标签列表
#     new_labels = [merged_classes[feature] for feature in combined_features]
#
#     # 输出新的标签
#     # print("新标签:", new_labels)
#
#     # 返回新的标签
#     return new_labels


# 根据now_bandgap中的material_id加载对应的structure化学结构数据，最后返回structure和bandgap数据


def load_dataset(now_bandgap):
    with open("../datasets/mp.2019.04.01.json") as f:
        structure_data = {i["material_id"]: i["structure"] for i in json.load(f)}
    structures = []
    targets = []
    for mid in now_bandgap.keys():
        now_bandgap[mid] = torch.tensor(now_bandgap[mid])
        struct = Structure.from_str(structure_data[mid], fmt="cif")
        structures.append(struct)
        targets.append(now_bandgap[mid])
    return structures, targets

# 将实验数据五五开划分为验证集和测试集
def get_val_test_dataset():
    with open('../datasets/all_data.json', 'r') as fp:
        d = json.load(fp)

    # 高保真度数据集
    Exp_structures = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]  # 将文件中的原始结构数据转换为pymatgen结构
    Exp_targets = [torch.tensor(x['band_gap']) for x in d['ordered_exp'].values()]  # 带隙数据


    combined = list(zip(Exp_structures, Exp_targets))
    n_exp = len(Exp_targets)

    # 将带隙数据离散化为多个区间，以便进行分层
    bins = [0, 4.0, 8.0, 12.0, float('inf')]  # 设置分层区间
    targets_discretized = np.digitize([target.item() for target in Exp_targets], bins) - 1  # 将带隙离散化

    # 使用带隙离散化后的标签进行分层抽样
    exp_val_data, exp_test_data = train_test_split(combined,
                                                   test_size=0.5,
                                                   random_state=42,
                                                   stratify=targets_discretized)

    # 获取验证集和测试集的索引
    val_indices = np.array([i for i in range(n_exp) if combined[i] in exp_val_data])
    test_indices = np.array([i for i in range(n_exp) if combined[i] in exp_test_data])

    print("验证集索引集合:", val_indices, f"验证集索引长度：{len(val_indices)}")
    print("测试集索引集合:", test_indices, f"测试集索引长度：{len(test_indices)}")

    # 提取验证集的化学结构和目标带隙数据
    exp_val_structures = [Exp_structures[i] for i in val_indices]
    exp_val_targets = [Exp_targets[i] for i in val_indices]

    # 提取测试集的化学结构和目标带隙数据
    exp_test_structures = [Exp_structures[i] for i in test_indices]
    exp_test_targets = [Exp_targets[i] for i in test_indices]

    print(f"验证集大小：{len(exp_val_data)}")
    print(f"测试集大小：{len(exp_test_data)}")

    return exp_val_structures, exp_val_targets, exp_test_structures, exp_test_targets


def get_sampled_low_fidelity_dataset(name, random_seed):
    with gzip.open("../datasets/data_no_structs.json.gz", "rb") as f:
        bandgap = json.loads(f.read())  # 所有带隙数据

    now_bandgap = bandgap[name]  # 当前数据集name的所有带隙数据
    now_structures, now_targets = load_dataset(now_bandgap)  # 当前数据集name的所有structures和带隙数据
    combined = list(zip(now_structures, now_targets))

    sample_size=472
    # 设置随机种子
    random.seed(random_seed)

    # 将带隙数据离散化为多个区间
    # 假设我们将带隙分为4个区间：（<4.0）、（4.0~8.0）、（8.0~12.0）、（>12.0）
    bins = [0, 4.0, 8.0, 12.0, float('inf')]  # 设置分层区间
    targets_discretized = np.digitize([target.item() for target in now_targets], bins) - 1  # 将带隙离散化为0, 1, 2标签

    # 按离散化后的带隙值分层
    stratified_groups = defaultdict(list)
    for structure, target, label in zip(now_structures, now_targets, targets_discretized):
        stratified_groups[label].append((structure, target))

    # 计算每个类别应抽取的样本数
    total_samples = len(combined)
    sample_per_class = {}

    for label, group in stratified_groups.items():
        sample_per_class[label] = int(len(group) * (sample_size / total_samples))

    # 执行分层抽样
    sampled_data = []

    for label, group in stratified_groups.items():
        # 从每个类别中随机抽取样本
        num_samples = sample_per_class[label]
        sampled_data.extend(random.sample(group, num_samples))

    # 如果总的抽样数不准确，修正抽样数量
    if len(sampled_data) != sample_size:
        difference = sample_size - len(sampled_data)
        remaining_data = [item for group in stratified_groups.values() for item in group]
        sampled_data.extend(random.sample(remaining_data, difference))

    # 将抽样后的结构和目标带隙数据拆分开
    sampled_structures, sampled_targets = zip(*sampled_data)

    # print(f"{name}数据集按带隙范围分层抽样后大小：{len(sampled_structures)} 随机种子：{random_seed}")
    # print(f"前10条抽样目标值：{sampled_targets[:10]}")

    return sampled_structures, sampled_targets

# 加载scan数据集的所有数据
def get_low_fidelity_dataset(name):
    with gzip.open("../datasets/data_no_structs.json.gz", "rb") as f:
        bandgap = json.loads(f.read())
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    print(f"{name}数据集大小：{len(now_structures)}")
    return now_structures, now_targets


