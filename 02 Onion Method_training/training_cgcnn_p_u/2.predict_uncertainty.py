from __future__ import annotations

import argparse
import glob
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
from load_datasets import get_train_val_test_dataset,get_low_fidelity_dataset,get_sampled_low_fidelity_dataset

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



# def load_model_from_checkpoint(orig_atom_fea_len, nbr_fea_len, path):
#     # 首先创建一个新的 CrystalGraphConvNet 实例
#     crystal_graph_conv_net = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, atom_fea_len=args.atom_fea_len,
#                                                  n_conv=3, h_fea_len=128, n_h=1, classification=False)
#     # 创建一个新的 Cgcnn_lightning 实例
#     model_temp = Cgcnn_lightning(crystal_graph_conv_net, normalizer_train)  # 这里暂时将 normalizer 设置为 None，后续需正确设置

#     # 加载模型的状态字典（仅包含权重信息）
#     state_dict = torch.load(model_path)['state_dict']

#     # 过滤状态字典，只保留与 CrystalGraphConvNet 相关的键
#     # 通过去掉 'crystalGraphConvNet.' 前缀来调整键名
#     filtered_state_dict = {k.replace('crystalGraphConvNet.', ''): v for k, v in state_dict.items()}

#     # 加载过滤后的状态字典到 CrystalGraphConvNet 中
#     model_temp.crystalGraphConvNet.load_state_dict(filtered_state_dict)
#     return model_temp
#     random.seed(42)
#     combined = list(zip(train_inputs, train_outputs))
#     sampled_data = random.sample(combined, 472)
#     train_inputs, train_outputs = zip(*sampled_data)

#     train_inputs = pd.Series(train_inputs)
#     train_outputs = pd.Series(train_outputs)
#     dataset = StruData(train_inputs, train_outputs)

#     checkpoint = torch.load(path)
#     normalizer = checkpoint['normalizer']  # 从保存的文件中加载标准化器，适应训练数据的normalizer,防止数据泄露

#     # build model
#     structures, _, = dataset[0]
#     orig_atom_fea_len = structures[0].shape[-1]
#     nbr_fea_len = structures[1].shape[-1]

#     model_t = Cgcnn_lightning(CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
#                                                 atom_fea_len=args.atom_fea_len,
#                                                 n_conv=3,
#                                                 h_fea_len=128,
#                                                 n_h=1,
#                                                 classification=False), normalizer)
#     # 加载模型权重
#     model_t.load_state_dict(checkpoint['state_dict'])

#     return model_t

def load_model_from_checkpoint(inputs,outputs, model_dir):
    # 找到指定目录下所有的检查点文件
    checkpoint_files = glob.glob(os.path.join(model_dir, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

    # 由于 save_top_k=1，只有一个检查点文件，直接获取该文件路径
    model_path = checkpoint_files[0]

    checkpoint = torch.load(model_path)
    normalizer = checkpoint['normalizer']  # 从保存的文件中加载标准化器，适应训练数据的normalizer,防止数据泄露
    # 首先创建一个新的 CrystalGraphConvNet 实例

    inputs = pd.Series(inputs)
    outputs = pd.Series(outputs)
    dataset = StruData(inputs,outputs)

    structures, _, = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    crystal_graph_conv_net = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                                atom_fea_len=args.atom_fea_len,
                                                n_conv=3,
                                                h_fea_len=128,
                                                n_h=1,
                                                classification=False)

    # 创建一个新的 Cgcnn_lightning 实例
    model_temp = Cgcnn_lightning(crystal_graph_conv_net, normalizer)

    # 加载模型的状态字典（仅包含权重信息）
    state_dict = checkpoint['state_dict']

    # 过滤状态字典，只保留与 CrystalGraphConvNet 相关的键
    # 通过去掉 'crystalGraphConvNet.' 前缀来调整键名
    filtered_state_dict = {k.replace('crystalGraphConvNet.', ''): v for k, v in state_dict.items()}

    # 加载过滤后的状态字典到 CrystalGraphConvNet 中
    model_temp.crystalGraphConvNet.load_state_dict(filtered_state_dict)
    return model_temp


def predict(test_loader, model_t):
    model_t.eval()
    predictions = []

    with torch.no_grad():
        for input_t, target in test_loader:
            output = model_t(*input_t)
            if isinstance(output, tuple) and len(output) == 2:
                bandgap_pred, u_pred = output
                # denorm_bandgap = model_t.normalizer.denorm(bandgap_pred.squeeze()) # 只反标准化带隙预测值
                predictions.append([u_pred.item(), bandgap_pred.item()]) # 这里把u和bandgap先后顺序改了

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


def predict_p_and_u():
    _,_,_,_,test_x, test_y = get_train_val_test_dataset()
    test_x = pd.Series(test_x)
    test_y = pd.Series(test_y)
    dataset = StruData(test_x, test_y)
    test_loader = DataLoader(dataset=dataset,
                             # batch_size=128,  # 设置批处理大小,默认是1
                             shuffle=False,  # 关闭数据洗牌
                             num_workers=0,  # 设置数据加载器的工作线程数量
                             collate_fn=collate_pool)

    structures, targets,_,_,_,_ = get_train_val_test_dataset()

    print(f"--------------------------------------")
    model_pu = load_model_from_checkpoint(structures, targets, "./saved_models/model_5")
    mae, mse, targets_pre, u = get_mae_mse(model_pu, test_loader, test_y)

    print(f"第C列->MAE:{mae.item():.3f}  第D列->MSE:{mse.item():.3f}")

    return 0


predict_p_and_u()

