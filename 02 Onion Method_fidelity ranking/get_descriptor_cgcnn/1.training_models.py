import argparse
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
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from random import sample, seed
from model import CrystalGraphConvNet
from data import StruData, get_train_loader, collate_pool
from load_datasets import get_sampled_low_fidelity_dataset,get_low_fidelity_dataset,get_val_test_dataset

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback


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

        loss = loss_fn(y_hat, target_var)# 此时的y_hat, target_var都是标准化后的数值

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=128)
        return loss

    def validation_step(self, batch):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])
        target_var = self.normalizer.norm(y)

        # y_hat = self.crystalGraphConvNet(*input_var)
        y_hat = self.forward(*input_var)

        loss_fn = nn.L1Loss()  # mae
        val_loss = loss_fn(y_hat, target_var)# 此时的y_hat, target_var都是标准化后的数值，即val_loss并不是真实的loss

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
        test_loss = loss_fn(self.normalizer.denorm(y_hat), target_var)# 此时的y_hat时反标准化的数值, target_var是原数值真值
        self.log('test_MAE', test_loss, on_epoch=True, prog_bar=True, batch_size=128)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

warnings.simplefilter("ignore")

# if torch.cuda.is_available():
#     torch.cuda.set_device(0)

# 使用本机上的第二个GPU运行
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def training_model(train_inputs,train_outputs,saved_name):
    train_inputs = pd.Series(train_inputs)
    train_outputs = pd.Series(train_outputs)
    # print("训练数据集大小: ", len(train_inputs), len(train_outputs))
    val_inputs, val_outputs,_,_ = get_val_test_dataset()
    val_inputs = pd.Series(val_inputs)
    val_outputs = pd.Series(val_outputs)

    dataset_train = StruData(train_inputs, train_outputs) # 这个只用于后续生成当前的标准化器，而不用于生成数据加载器，因为target值还没有标准化
    # dataset_val = StruData(val_inputs, val_outputs)

    # 材料的带隙在 0 eV 到 5 eV 之间，则标准化可以帮助提高模型的稳定性和训练效率，避免模型在训练时因目标值的尺度差异产生偏差。
    # 如果带隙值的范围在 0.5 eV 到 2.5 eV，则可能不需要标准化，因为在这种情况下，带隙的尺度本身不会造成明显的训练问题。
    # 计算数据集中目标值（target）的标准化器（Normalizer）。标准化器用于将目标值进行标准化，以便训练过程中的损失函数和优化器能更好地工作。常用于回归问题中。

    # 计算训练数据集目标值（target）的标准化器
    sample_data_list = [dataset_train[i] for i in range(len(dataset_train))]
    _, sample_target = collate_pool(sample_data_list)
    normalizer_train = Normalizer(sample_target)  # 使用训练集的目标值计算标准化参数，normalizer_train这是当前训练数据生成的标准化器


    # 构建模型
    structures, _, = dataset_train[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = Cgcnn_lightning(CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                                atom_fea_len=args.atom_fea_len,
                                                n_conv=3,
                                                h_fea_len=128,
                                                n_h=1,
                                                classification=False), normalizer_train)

    # 早停回调
    early_stop_callback = EarlyStopping(
        monitor="val_MAE",
        min_delta=0.00,
        patience=200,
        verbose=True,
        mode="min"
    )

    # CSVLogger，用于记录训练过程中的日志
    logger = CSVLogger("./logs", name=f"{saved_name}")

    # 模型检查点回调,用于保存模型参数
    # 它保存的是 模型的状态字典（state_dict），而不是整个模型对象，具体来说，保存的文件包含了模型的 参数，即 权重 和 偏置（通过 state_dict）。
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_MAE',
    #     dirpath=f"./saved_models", # 临时保存最佳模型的路径,这里实际上保存的是检查点的数据，但是该数据也可以再加载模型阶段使用
    #     filename=f'{saved_name}_{now_seed}',
    #     save_top_k=1,
    #     mode='min'
    # )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_MAE',
        dirpath=f"./saved_models/{saved_name}",  # 保存模型的文件夹路径，保持不变
        # 使用占位符 {epoch} 和 {val_MAE:.3f} 来动态生成文件名
        filename=f'{{epoch}}_{{val_MAE:.3f}}',
        save_top_k=1,
        mode='min'
    )

    # 保存normalizer
    normalizer_callback = SaveNormalizerCallback()


    trainer = pl.Trainer(max_epochs=100000, callbacks=[early_stop_callback, checkpoint_callback,normalizer_callback], logger=logger)
    
    # 目标值与输入值重新组合，用于后续生成数据加载器
    dataset_train = StruData(train_inputs, train_outputs)
    dataset_val = StruData(val_inputs, val_outputs)
    # 创建训练和验证的 DataLoader
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=64,
                              collate_fn=collate_pool, # 自定义的数据整理函数collate_pool
                              num_workers=4) # 启动 4 个子进程同时进行数据加载（数据加载通常在 CPU 上进行）
    val_loader = DataLoader(dataset=dataset_val,
                            batch_size=64,
                            collate_fn=collate_pool,
                            num_workers=4)

    # 训练模型
    print("开始训练...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("训练结束")

    return model, normalizer_train



# 'scan', 'hse', 'gllb-sc', 'pbe'
all_datasets_name = [ 'hse', 'gllb-sc', 'pbe']

for name in all_datasets_name:
    print(f"START===================================train {name} for p&u=======================================")
    if name == 'scan':
        structures, targets = get_low_fidelity_dataset(name)
        model = training_model(structures, targets, name)
    else:
        structures, targets = get_sampled_low_fidelity_dataset(name)
        model = training_model(structures,targets, name)
    print(f"END===================================train {name} for p&u=======================================")
