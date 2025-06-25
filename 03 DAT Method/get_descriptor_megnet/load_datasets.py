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


def get_low_fidelity_dataset(name):
    with gzip.open("../datasets/data_no_structs.json.gz", "rb") as f:
        bandgap = json.loads(f.read())
    now_bandgap = bandgap[name]
    now_structures, now_targets = load_dataset(now_bandgap)
    print(f"{name}数据集大小：{len(now_structures)}")
    return now_structures, now_targets




