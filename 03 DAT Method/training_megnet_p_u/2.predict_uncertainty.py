from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from matgl.config import DEFAULT_ELEMENTS
from matgl.layers import BondExpansion
from matgl.ext.pymatgen import Structure2Graph, get_element_list

# import my_megnet_1 as megnet
from load_datasets import get_val_test_dataset,get_low_fidelity_dataset
from data_megnet import MEGNetDataset, MGLDataLoader, collate_fn_DAT
from megnet_lightning import ModelLightningModule
from models_megnet import FeatureExtractor,MegnetRegression,MegnetClassification

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1,0"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
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
    # 实例化 MegnetRegression 类
    megnet_regression = MegnetRegression(
        dim_blocks_out=32,
        hidden_layer_sizes_output=(32, 16),
        activation_type="softplus"
    )

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

    # # 提取与 feature_extractor 相关的参数
    # feature_extractor_state_dict = {k.replace('feature_extractor.', ''): v for k, v in state_dict.items() if
    #                                 k.startswith('feature_extractor.')}
    # # 加载 feature_extractor 的参数
    # feature_extractor.load_state_dict(feature_extractor_state_dict)

    # 提取与 regressor 相关的参数
    regressor_state_dict = {k.replace('regressor.', ''): v for k, v in state_dict.items() if k.startswith('regressor.')}
    # 加载 regressor 的参数
    megnet_regression.load_state_dict(regressor_state_dict)

    print("FeatureExtractor 和 MegnetRegression 模型参数加载成功！")

    return feature_extractor,megnet_regression


def get_test_results(name):
    _,_,test_structures, test_targets = get_val_test_dataset()
    # 加载训练好的模型
    model_f,model_r = load_my_model(name)

    pre_targets = []
    uncertainties =[]
    # 进行预测
    for s in test_structures:
        features = model_f.get_features(s)
        pre_target,u_target = model_r.predict_structure(features)
        pre_targets.append(pre_target)
        uncertainties.append(u_target)

    targets_4_list.append(pre_targets)
    uncertainty_4_list.append(uncertainties)

    pre_targets_np = np.array(pre_targets)
    test_targets_np = np.array(test_targets)

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    print(f"{name}  第C列->MAE: {mae:.3f}, 第D列->MSE: {mse:.3f}")
    return 0



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


# all_datasets_name = ['scan','hse', 'pbe','gllb-sc']

_,_,_, test_y_real = get_val_test_dataset()  # test_y_real:真实值 1352个（4个组中对应序号的真实值都相同）
targets_4_list = []  # 存放4个组的预测值 4*1352个
uncertainty_4_list = []  # 存放4个组的测试数据的不确定度; 4*1352个


# all_datasets_name = ['scan','hse', 'pbe','gllb-sc']
all_datasets_name = ['scan','hse', 'pbe','gllb-sc']

for name in all_datasets_name:
    get_test_results(name)

get_both_mae_mse_integrated()