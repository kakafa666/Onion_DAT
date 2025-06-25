import argparse
import os
import sys
import warnings

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model
from data import StruData, collate_pool
from load_datasets import get_val_test_dataset
from model import CrystalGraphConvNet

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
        target_var = self.normalizer.norm(y)

        # y_hat = self.crystalGraphConvNet(*input_var)
        y_hat = self.forward(*input_var)  # 使用 forward 方法进行前向传播

        loss_fn = nn.MSELoss()

        loss = loss_fn(y_hat, target_var)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=128)
        return loss

    def validation_step(self, batch):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])
        target_var = self.normalizer.norm(y)

        # y_hat = self.crystalGraphConvNet(*input_var)
        y_hat = self.forward(*input_var)

        loss_fn = nn.L1Loss()  # mae
        val_loss = loss_fn(y_hat, target_var)

        self.log('val_MAE', val_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return val_loss

    def test_step(self, batch):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])
        target_var = y

        # y_hat = self.crystalGraphConvNet(*input_var)
        y_hat = self.forward(*input_var)
        # loss
        loss_fn = nn.L1Loss()
        test_loss = loss_fn(self.normalizer.denorm(y_hat), target_var)
        self.log('test_MAE', test_loss, on_epoch=True, prog_bar=True, batch_size=128)

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

    model = Cgcnn_lightning(CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                                atom_fea_len=args.atom_fea_len,
                                                n_conv=3,
                                                h_fea_len=128,
                                                n_h=1,
                                                classification=False), normalizer)
    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])

    return model

def predict(structures, targets, model):
    test_inputs = pd.Series(structures)
    test_outputs = pd.Series(targets)
    dataset = StruData(test_inputs, test_outputs)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=len(test_inputs), # 也可以设置批处理大小为128，防止内存溢出
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_pool)

    model.eval()
    predictions = []

    with torch.no_grad():
        for input_t, target in test_loader:
            output = model(*input_t)
            # 只反标准化维度为0的值，因为维度为0的值是预测的带隙值，而其他的维度可能额是材料的其他性质，在这里没用上，所以不用管
            if output.dim() == 2:
                output[:, 0] = model.normalizer.denorm(output[:, 0])
            predictions.extend(output.tolist())

    # mae_errors.avg,
    return predictions


def get_test_results(name):
    model.now_descriptor = []
    _,_,test_structures, test_targets = get_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model(test_structures, test_targets, path=f"./saved_models/{name}")

    pre_targets = predict(test_structures, test_targets, cgcnn_model)
    pre_targets = [target[0] for target in pre_targets]

    targets_4_list.append(pre_targets)

    pre_targets_np = np.array(pre_targets)# 预测值
    test_targets_np = np.array(test_targets)# 真实值

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    # mae = round(mae, 3)
    # mse = round(mse, 3)

    print(f"{name}  MAE: {mae}, MSE: {mse}")

    # return mae, mse
    return 0


# G H 列 非加权融合
def get_mae_mse_integrated():
    print(f"================================= G H 列 start  ======================================")
    # 计算综合mae，mse值
    temp_targets_list = targets_4_list.copy()  # 4*1352预测值
    mae_all_no_weighted = [] # 非加权
    mse_all_no_weighted = []
    for j in range(len(temp_targets_list[0])):  # 1352
        ae_4_no_weighted_temp=0
        for i in range(len(temp_targets_list)):  # 4
            ae_4_no_weighted_temp += temp_targets_list[i][j]*(1/len(temp_targets_list))
        ae_4_no_weighted = abs(ae_4_no_weighted_temp - test_y_real[j])
        se_4_no_weighted = (ae_4_no_weighted_temp - test_y_real[j]) ** 2
        mae_all_no_weighted.append(ae_4_no_weighted)
        mse_all_no_weighted.append(se_4_no_weighted)
    mae_no_weighted_ave = np.mean(mae_all_no_weighted)
    mse_no_weighted_ave = np.mean(mse_all_no_weighted)

    print(f"第G列->mae: {mae_no_weighted_ave:.3f}")
    print(f"第H列->mse: {mse_no_weighted_ave:.3f}")

    print(f"================================= G H 列 END  ======================================")

    return 0


all_datasets_name = ['scan','hse', 'pbe','gllb-sc']
_,_,_, test_y_real = get_val_test_dataset()  # test_y_real:真实值 1352个（40组中对应序号的真实值都相同）
targets_4_list = []  # 存放4个组的预测值 4*1352

for name in all_datasets_name:
    get_test_results(name)

get_mae_mse_integrated()


