from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import warnings
from random import sample

from matplotlib import ticker
from torch.utils.data import DataLoader

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

from data import StruData, collate_pool
# from model import CrystalGraphConvNet
# import model
from load_datasets import get_val_test_dataset,get_low_fidelity_dataset,get_sampled_low_fidelity_dataset
from model_conv import ConvNetDat

warnings.simplefilter("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--max_epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1000)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
# emb dim
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')

parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')

args = parser.parse_args(sys.argv[1:])

# 使用本机上的第二个GPU运行
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

# 回归器模型
class CrystalGraphFc(nn.Module):
    def __init__(self):
        super(CrystalGraphFc, self).__init__()
        self.conv_to_fc = nn.Linear(64, 128)
        self.conv_to_fc_softplus = nn.Softplus()
        # 新增两个预测全连接层 zdn
        self.fc_out_bandgap = nn.Linear(128, 1)
        self.fc_out_uncertainty = nn.Linear(128, 1)
        # self.fc_out = nn.Linear(128, 1)

    def forward(self, crys_fea_conv):
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea_conv))
        # print("after conv_to_fc atom_fea: ", crys_fea.shape)

        crys_fea = self.conv_to_fc_softplus(crys_fea)
        # print("after conv_to_fc_softplus atom_fea: ", crys_fea.shape)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        # 分别经过两个预测全连接层 zdn
        out_bandgap = self.fc_out_bandgap(crys_fea)
        out_uncertainty = self.fc_out_uncertainty(crys_fea)

        return out_bandgap, out_uncertainty



class Cgcnn_lightning(pl.LightningModule):
    def __init__(self, convNetDat, normalizer):
        super().__init__()
        self.normalizer = normalizer
        # 使用cgcnn的卷积层和池化层作为特征提取器
        self.feature_extractor = convNetDat
        # 实例化 回归器
        self.crystalGraph_regression = CrystalGraphFc()

    def forward(self, *inputs):
        return self.crystalGraph_regression(self.feature_extractor(*inputs))


    def validation_step(self, batch, batch_idx):
        regression_output = []
        target_var = []
        (x1, x2, x3, x4), (y, labels) = batch
        input_vars = (x1, x2, x3, x4)
        y=self.normalizer.norm(y)
        y_hat_bandgap, y_hat_uncertainty = self.forward(*input_vars)  # 批量调用
        y_hat = torch.stack([y_hat_uncertainty.squeeze(), y_hat_bandgap.squeeze()], dim=1)
        diff = torch.abs(y - y_hat[:, 1]) # |Ptrue - P|
        p_loss = nn.MSELoss()(y, y_hat[:, 1])  # |Ptrue - P|
        u_loss = nn.MSELoss()(diff, y_hat[:, 0])  # (|Ptrue - P| - u)²
        val_loss_regression = p_loss + u_loss # 计算总损失 (Ptrue -P)² + (|Ptrue - P| - u)²
        
        self.log('val_r_loss', val_loss_regression, on_epoch=True, prog_bar=True, batch_size=len(target_var))

        return val_loss_regression



def load_model_from_checkpoint(inputs,outputs,path):
    
    inputs = pd.Series(inputs)
    outputs = pd.Series(outputs)
    dataset = StruData(inputs, outputs)

    checkpoint = torch.load(path)
    normalizer = checkpoint['normalizer']  # 从保存的文件中加载标准化器，适应训练数据的normalizer,防止数据泄露

    structures, _, = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    # 创建一个新的feature_extractor实例
    feature_extractor = ConvNetDat(orig_atom_fea_len, nbr_fea_len)
    # 创建一个新的Cgcnn_lightning模型实例
    model_t = Cgcnn_lightning(feature_extractor, normalizer)
    # 提取需要加载的参数（特征提取器和回归器）
    state_dict_to_load = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('feature_extractor.') or key.startswith('crystalGraph_regression.'):
            state_dict_to_load[key] = value
    # 加载参数到Cgcnn_lightning模型
    model_t.load_state_dict(state_dict_to_load, strict=False)
    return model_t


def predict(structures, targets, model_t):
    test_inputs = pd.Series(structures)
    test_outputs = pd.Series(targets)
    dataset = StruData(test_inputs, test_outputs)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=len(test_inputs), # 也可以设置批处理大小为128，防止内存溢出
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_pool)

    model_t.eval()
    predictions = []
    uncertainties=[]

    with torch.no_grad():
        for input_t, target in test_loader:
            output,uncertainty = model_t(*input_t)
            # # 只反标准化维度为0的值，因为维度为0的值是预测的带隙值，而其他的维度可能额是材料的其他性质，在这里没用上，所以不用管
            # if output.dim() == 2:
            prediction= model_t.normalizer.denorm(output)
            # output[:, 0] = model_t.normalizer.denorm(output[:, 0])
            predictions.extend(prediction)
            uncertainties.extend(uncertainty)

    # mae_errors.avg,
    return predictions,uncertainties


def get_test_results(name):
    # model.now_descriptor = []
    _,_,test_structures, test_targets = get_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model_from_checkpoint(test_structures, test_targets, path=f"./saved_models/{name}_DAT_loss.ckpt")

    pre_targets, u_targets = predict(test_structures, test_targets, cgcnn_model)

    targets_4_list.append(pre_targets)
    uncertainty_4_list.append(u_targets)

    pre_targets_np = np.array(pre_targets)# 预测值
    test_targets_np = np.array(test_targets)# 真实值

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    # mae = round(mae, 3)
    # mse = round(mse, 3)

    print(f"{name}  第C列->MAE: {mae:.3f}, 第D列->MSE: {mse:.3f}")
    return 0



# G H I J 列 加权/非加权融合值
def get_both_mae_mse_integrated():
    print(f"================================= G H I J 列 start  ======================================")
    # 计算1352组归一化权重
    temp_list = uncertainty_4_list.copy()  # 4*1352 不确定度
    # print("行，列:", len(temp_list),len(temp_list[0]))
    weighted_list = []  # 存放1352组权重(1352个list）->内部list(4个权重)
    for j in range(len(temp_list[0])):  # 1352
        sub_list = []  # 存放4个对应u的倒数
        for i in range(len(temp_list)):  # 4
            sub_list.append(1/temp_list[i][j])
        sum_reciprocals = sum(sub_list)
        weights = [r/sum_reciprocals for r in sub_list]  # 归一化处理
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
# all_datasets_name = ['scan']
all_datasets_name = ['scan','hse', 'pbe','gllb-sc']
for name in all_datasets_name:
    get_test_results(name)


get_both_mae_mse_integrated()
