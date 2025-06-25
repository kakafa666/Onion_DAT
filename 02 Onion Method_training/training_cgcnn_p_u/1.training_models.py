import argparse
import glob
import gzip
import json
import os
import random
import shutil
import sys
import warnings

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from random import sample, seed
from model import CrystalGraphConvNet # cgcnn模型
from data import StruData, get_train_loader, collate_pool
from load_datasets import get_low_fidelity_dataset,get_train_val_test_dataset


# 处理anaconda和torch重复文件
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


# def mae(prediction, target):
#     """
#     Computes the mean absolute error between prediction and target
#
#     Parameters
#     ----------
#
#     prediction: torch.Tensor (N, 1)
#     target: torch.Tensor (N, 1)
#     """
#     return torch.mean(torch.abs(target - prediction))

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


warnings.simplefilter("ignore")


# 使用本机上的第二个GPU运行
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(init_seed)


class SaveNormalizerCallback(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['normalizer'] = pl_module.normalizer  # 保存 normalizer
        return checkpoint

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'normalizer' in checkpoint:
            pl_module.normalizer = checkpoint['normalizer']
        return checkpoint


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveNormalizerCallback(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['normalizer'] = pl_module.normalizer  # 保存 normalizer
        return checkpoint

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'normalizer' in checkpoint:
            pl_module.normalizer = checkpoint['normalizer']
        return checkpoint


def load_model_from_checkpoint(orig_atom_fea_len, nbr_fea_len, model_dir,normalizer_train):
    # 找到指定目录下所有的检查点文件
    checkpoint_files = glob.glob(os.path.join(model_dir, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

    # 由于 save_top_k=1，只有一个检查点文件，直接获取该文件路径
    model_path = checkpoint_files[0]

    # 首先创建一个新的 CrystalGraphConvNet 实例
    crystal_graph_conv_net = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, atom_fea_len=args.atom_fea_len,
                                                 n_conv=3, h_fea_len=128, n_h=1, classification=False)
    # 创建一个新的 Cgcnn_lightning 实例
    model_temp = Cgcnn_lightning(crystal_graph_conv_net, normalizer_train)  # 这里暂时将 normalizer 设置为 None，后续需正确设置

    # 加载模型的状态字典（仅包含权重信息）
    state_dict = torch.load(model_path)['state_dict']

    # 过滤状态字典，只保留与 CrystalGraphConvNet 相关的键
    # 通过去掉 'crystalGraphConvNet.' 前缀来调整键名
    filtered_state_dict = {k.replace('crystalGraphConvNet.', ''): v for k, v in state_dict.items()}

    # 加载过滤后的状态字典到 CrystalGraphConvNet 中
    model_temp.crystalGraphConvNet.load_state_dict(filtered_state_dict)
    return model_temp

def training_model(train_inputs,train_outputs , idx, prev_model_path=None):
    train_inputs = pd.Series(train_inputs)
    train_outputs = pd.Series(train_outputs)
    # print("训练数据集大小: ", len(train_inputs), len(train_outputs))
    _,_,val_inputs, val_outputs,_,_ = get_train_val_test_dataset()
    val_inputs = pd.Series(val_inputs)
    val_outputs = pd.Series(val_outputs)

    dataset_train = StruData(train_inputs, train_outputs)  # 这个只用于后续生成当前的标准化器，而不用于生成数据加载器，因为target值还没有标准化
    # dataset_val = StruData(val_inputs, val_outputs)

    # obtain target value normalizer
    if len(dataset_train) < 500:
        # warnings.warn('Dataset has less than 500 data points. '
        #               'Lower accuracy is expected. ')
        sample_data_list = [dataset_train[i] for i in range(len(dataset_train))]
    else:
        sample_data_list = [dataset_train[i] for i in
                            sample(range(len(dataset_train)), 500)]
    _, sample_target = collate_pool(sample_data_list)
    normalizer_train = Normalizer(sample_target)


    # 构建模型
    structures, _, = dataset_train[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    if prev_model_path is not None:
        # 从检查点文件中加载整个模型
        model_tc = load_model_from_checkpoint(orig_atom_fea_len, nbr_fea_len,prev_model_path,normalizer_train)
    else:
        model_tc = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, atom_fea_len=args.atom_fea_len,
                                       n_conv=3, h_fea_len=128, n_h=1, classification=False)
    model_t = Cgcnn_lightning(model_tc, normalizer_train)

    # 早停回调
    early_stop_callback = EarlyStopping(
        monitor="val_MAE",
        min_delta=0.00,
        patience=200,
        verbose=True,
        mode="min"
    )

    # CSVLogger，用于记录训练过程中的日志
    logger = CSVLogger("./logs", name=f"{idx}")

    # 模型检查点回调,用于保存模型参数
    # 它保存的是 模型的状态字典（state_dict），而不是整个模型对象，具体来说，保存的文件包含了模型的 参数，即 权重 和 偏置（通过 state_dict）。
    checkpoint_callback = ModelCheckpoint(
        monitor='val_MAE',
        dirpath=f"./saved_models/model_{idx}",  # 修改保存路径
        filename='{epoch}-{val_MAE:.4f}',  # 修改文件名格式
        save_top_k=1,
        mode='min'
    )

    # 保存normalizer
    normalizer_callback = SaveNormalizerCallback()

    trainer = pl.Trainer(max_epochs=100000, 
                         callbacks=[early_stop_callback, checkpoint_callback,normalizer_callback], 
                         logger=logger,
                         accelerator="cuda",  # 使用 CUDA 设备
                         devices=2,  # 使用 2 个 GPU
                         strategy="ddp",  # 使用分布式数据并行 (DDP)
                         )
    
    # 将目标值与输入值重新组合，用于后续生成数据加载器
    dataset_train_norm = StruData(train_inputs, train_outputs)
    dataset_val_norm = StruData(val_inputs, val_outputs)
    # 创建训练和验证的 DataLoader
    train_loader = DataLoader(dataset=dataset_train_norm,
                              batch_size=64,
                              collate_fn=collate_pool, # 自定义的数据整理函数collate_pool
                              num_workers=4) # 启动 4 个子进程同时进行数据加载（数据加载通常在 CPU 上进行）
    val_loader = DataLoader(dataset=dataset_val_norm,
                            batch_size=64,
                            collate_fn=collate_pool,
                            num_workers=4)

    # 训练模型
    print("开始训练...")
    trainer.fit(model_t, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("训练结束")

    return model_t, normalizer_train

# (1, ('gllb-sc', 'hse', 'scan', 'pbe', 'exp')),
all_datasets_name = [(2, ('hse', 'scan', 'pbe', 'exp')),
                     (3, ('hse', 'pbe', 'exp')),
                     (4, ('hse', 'exp')),
                     (5, 'exp')]

prev_model_path = './saved_models/model_1'

for idx, name in all_datasets_name:
    print(idx, name)
    print(f"START===================================train {idx} for p&u=======================================")
    if idx == 5:
        structures, targets, _, _, _, _ = get_train_val_test_dataset()
        training_model(structures, targets, idx, prev_model_path)
    else:
        all_structures = []
        all_targets = []
        for n in name:
            if n == 'exp':
                structures, targets, _, _, _, _ = get_train_val_test_dataset()
                all_structures.extend(structures)
                all_targets.extend(targets)
            else:
                structures, targets = get_low_fidelity_dataset(n)
                all_structures.extend(structures)
                all_targets.extend(targets)
        training_model(all_structures, all_targets, idx, prev_model_path)
        # 获取上一个检查点保存的模型路径
        prev_model_path = f'./saved_models/model_{idx}'

    print(f"END===================================train {idx} for p&u=======================================")
