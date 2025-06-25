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
from model import CrystalGraphConvNet
import model
from load_datasets import get_val_test_dataset,get_low_fidelity_dataset,get_sampled_low_fidelity_dataset

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

# 在此处修改损失函数
class Cgcnn_lightning(pl.LightningModule):

    def __init__(self, crystalGraphConvNet, normalizer):
        super().__init__()
        self.crystalGraphConvNet = crystalGraphConvNet
        self.normalizer = normalizer

    def forward(self, *input):
        return self.crystalGraphConvNet(*input)

    def training_step(self, batch):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])
        y = self.normalizer.norm(y)
        # 模型输出带隙预测值和不确定度
        y_hat_bandgap, y_hat_uncertainty = self.forward(*input_var)
        y_hat = torch.stack([y_hat_uncertainty.squeeze(), y_hat_bandgap.squeeze()], dim=1)

        diff = torch.abs(y - y_hat[:, 1]) # |Ptrue - P|
        p_loss = nn.MSELoss()(y, y_hat[:, 1])  # |Ptrue - P|
        u_loss = nn.MSELoss()(diff, y_hat[:, 0])  # (|Ptrue - P| - u)²
        total_loss = p_loss + u_loss # 计算总损失 (Ptrue -P)² + (|Ptrue - P| - u)²

        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return total_loss


    def validation_step(self, batch):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])
        y = self.normalizer.norm(y)
        y_hat_bandgap, y_hat_uncertainty = self.forward(*input_var)
        y_hat = torch.stack([y_hat_uncertainty.squeeze(), y_hat_bandgap.squeeze()], dim=1)

        mae_uncertainty = nn.L1Loss()(y_hat[:, 0], torch.abs(y - y_hat[:, 1]))
        mae_bandgap = nn.L1Loss()(y, y_hat[:, 1])
        total_loss = mae_uncertainty + mae_bandgap

        self.log('val_MAE', total_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }



def load_model(train_inputs,train_outputs,path):
    train_inputs = pd.Series(train_inputs)
    train_outputs = pd.Series(train_outputs)
    dataset = StruData(train_inputs, train_outputs)

    # 获取目录下所有文件
    files = os.listdir(path)
    # 筛选出 .ckpt 文件
    ckpt_files = [f for f in files if f.endswith('.ckpt')]

    # 确保只有一个 .ckpt 文件
    if len(ckpt_files) != 1:
        raise ValueError(f"Expected exactly one .ckpt file in {path}, but found {len(ckpt_files)}.")

    # 构建完整的 .ckpt 文件路径
    ckpt_path = os.path.join(path, ckpt_files[0])

    checkpoint = torch.load(ckpt_path)
    normalizer = checkpoint['normalizer']  # 从保存的文件中加载标准化器，适应训练数据的normalizer,防止数据泄露

    structures, _, = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model_t = Cgcnn_lightning(CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                                atom_fea_len=args.atom_fea_len,
                                                n_conv=3,
                                                h_fea_len=128,
                                                n_h=1,
                                                classification=False), normalizer)
    # 加载模型权重
    model_t.load_state_dict(checkpoint['state_dict'])

    return model_t

def predict(test_loader, model_t):
    model_t.eval()
    predictions = []

    with torch.no_grad():
        for input_t, target in test_loader:
            output = model_t(*input_t)
            if isinstance(output, tuple) and len(output) == 2:
                bandgap_pred, u_pred = output
                denorm_bandgap = model_t.normalizer.denorm(bandgap_pred.squeeze()) # 只反标准化带隙预测值
                predictions.append([u_pred.item(), denorm_bandgap.item()]) # 这里把u和bandgap先后顺序改了
    # with torch.no_grad():
    #     for input_t, target in test_loader:
    #         output = model(*input_t)
    #         # 只反标准化维度为1的值，因为维度为1的值是预测值，而维度为0的值是不确定度u
    #         if output.dim() == 2:
    #             output[:, 1] = model.normalizer.denorm(output[:, 1])
    #         predictions.extend(output.tolist())

    return predictions


def get_mae_mse(model_t, test_loader, test_y):

    pre = predict(test_loader, model_t)
    target = np.array([result[1] for result in pre])
    u = np.array([result[0] for result in pre])

    targets = torch.tensor(target)
    test_y_tensor = torch.tensor(test_y)
    # mae mse
    mae = torch.mean(torch.abs(targets - test_y_tensor))
    mse = torch.mean((targets - test_y_tensor) ** 2)

    return mae, mse, target, u,


def predict_p_and_u(name):
    _,_,test_x, test_y = get_val_test_dataset()

    test_x = pd.Series(test_x)
    test_y = pd.Series(test_y)
    dataset = StruData(test_x, test_y)
    test_loader = DataLoader(dataset=dataset,
                             # batch_size=128,  # 设置批处理大小,默认是1
                             shuffle=False,  # 关闭数据洗牌
                             num_workers=0,  # 设置数据加载器的工作线程数量
                             collate_fn=collate_pool)

    structures, targets = get_low_fidelity_dataset(name)

    print(f"-----------{name}:---------------")
    model_pu = load_model(structures, targets, path=f"./saved_models/{name}")

    mae, mse, targets_pre, u = get_mae_mse(model_pu, test_loader, test_y)

    uncertainty_4_list.append(u)
    targets_4_list.append(targets_pre)

    print(f"{name} 第C列->MAE:{mae.item():.3f} {name} 第D列->MSE:{mse.item():.3f}")

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


all_datasets_name = ['scan', 'hse', 'pbe', 'gllb-sc']

_,_,_, test_y_real = get_val_test_dataset()  # test_y_real:真实值 1352个（4个组中对应序号的真实值都相同）
targets_4_list = []  # 存放4个组的预测值 4*1352个
uncertainty_4_list = []  # 存放34个组的测试数据的不确定度; 4*1352个


for name in all_datasets_name:
    predict_p_and_u(name)

get_both_mae_mse_integrated()
