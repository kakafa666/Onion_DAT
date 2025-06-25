import argparse
import gzip
import json
import os
import random
import shutil
import sys
import warnings
from torch.utils.data import Sampler
from torch.utils.data import BatchSampler

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
from model_conv import ConvNetDat
from data import StruData, StruDataDat, collate_pool,collate_pool_Dat
from load_datasets import get_low_fidelity_dataset,get_val_test_dataset
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

        # out = self.fc_out(crys_fea)
        # # print("带隙预测值 output shape:", out.shape)
        # return out


class DomainGraphClassifier(nn.Module):
    def __init__(self, input_size):
        super(DomainGraphClassifier, self).__init__()
        # 第一个全连接层，将输入维度 input_size 转换为 128 维
        self.fc1 = nn.Linear(input_size, 128)
        # 激活函数 ReLU
        self.relu = nn.ReLU()
        # 第二个全连接层，将 128 维转换为 1 维
        self.fc2 = nn.Linear(128, 1)
        # Sigmoid 激活函数，用于将输出转换为概率分布
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 前向传播函数
        # 输入 x 通过第一个全连接层
        x = self.fc1(x)
        # 对第一个全连接层的输出应用 ReLU 激活函数
        x = self.relu(x)
        # 再通过第二个全连接层
        x = self.fc2(x)
        # 最后使用 Sigmoid 激活函数得到概率分布
        x = self.sigmoid(x)
        return x


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


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input * -ctx.lambda_, None

class Cgcnn_lightning(pl.LightningModule):
    def __init__(self, convNetDat, normalizer, lambda_grad_reverse, lambda_loss):
        super().__init__()
        self.normalizer = normalizer
        # 使用cgcnn的卷积层和池化层作为特征提取器
        self.feature_extractor = convNetDat
        # 实例化 回归器
        self.crystalGraph_regression = CrystalGraphFc()
        # 实例化 SimpleClassifier 分类器
        self.domain_classifier = DomainGraphClassifier(64)
        self.lambda_grad_reverse = lambda_grad_reverse
        self.lambda_loss = lambda_loss
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def forward_conv(self, *inputs):
        return self.feature_extractor(*inputs)

    def forward_regression(self, features):
        return self.crystalGraph_regression(features)

    def forward_classification(self, features):
        return self.domain_classifier(features)

    def training_step(self, batch, batch_idx):
        regression_output=[]
        target_var=[]
        (x1, x2, x3, x4), (y, labels) = batch
        #print("batch的第一条数据：", batch[0])
        input_vars = (x1, x2, x3, x4)
        y=self.normalizer.norm(y)
        # 分类过程
        # 将批次数据传递给特征提取器（特征提取过程）
        features = self.forward_conv(*input_vars)  # 批量调用
        data_with_features = list(zip(features, y, labels))

        # 所有数据直接进行分类
        classification_data = [item for item in data_with_features]
        classification_features = torch.stack([item[0] for item in classification_data])
        classification_labels = torch.stack([item[2] for item in classification_data])
        # 应用梯度反转层
        reversed_features = GradientReversalLayer.apply(classification_features, self.lambda_grad_reverse)
        # 调用分类模型进行预测
        classification_output = self.forward_classification(reversed_features)
        # 损失函数类型定义
        loss_fn_classification = nn.BCELoss()
        # 计算分类损失
        train_loss_classification = loss_fn_classification(classification_output, classification_labels)

        # 回归任务
        # 初始化回归损失为 0
        train_loss_regression = torch.tensor(0.0)
        features_r = self.forward_conv(*input_vars)  # 这里的冗余 是为了 后续的反向传播与分类的反向传播不冲突
        data_with_features_r = list(zip(features_r, y, labels))
        regression_data = [item for item in data_with_features_r if item[2] == 0]
        # print("train_step里，batch中，exp数据个数：", len(regression_data))
        if len(regression_data)>0:
            # 筛选后的批量数据
            regression_features = torch.stack([item[0] for item in regression_data])
            target_var = torch.tensor([item[1] for item in regression_data])

            y_hat_bandgap, y_hat_uncertainty = self.forward_regression(regression_features)
            y_hat = torch.stack([y_hat_uncertainty.squeeze(), y_hat_bandgap.squeeze()], dim=1)
            
            diff = torch.abs(y - y_hat[:, 1]) # |Ptrue - P|
            p_loss = nn.MSELoss()(y, y_hat[:, 1])  # |Ptrue - P|
            u_loss = nn.MSELoss()(diff, y_hat[:, 0])  # (|Ptrue - P| - u)²
            train_loss_regression = p_loss + u_loss # 计算总损失 (Ptrue -P)² + (|Ptrue - P| - u)²


        # 总损失 = 回归损失 + 分类损失（加权）
        train_loss_total = train_loss_regression + self.lambda_loss * train_loss_classification

        # 获取优化器
        optimizer = self.optimizers()

        # 反向传播
        train_loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录损失
        self.log('train_total_MAE', train_loss_total, on_epoch=True, prog_bar=True, batch_size=len(target_var))
        self.log('train_r_loss', train_loss_regression, on_epoch=True, prog_bar=True, batch_size=len(target_var))
        self.log('train_c_loss', train_loss_classification, on_epoch=True, prog_bar=True, batch_size=len(labels))
        self.log('train_r_data_count', len(target_var), on_epoch=True, prog_bar=True)

        return train_loss_total

    def validation_step(self, batch, batch_idx):
        regression_output = []
        target_var = []
        (x1, x2, x3, x4), (y, labels) = batch
        input_vars = (x1, x2, x3, x4)
        y=self.normalizer.norm(y)
        # 将批次数据传递给特征提取器（特征提取过程）
        features = self.forward_conv(*input_vars)  # 批量调用

        # 创建一个新的元组 (features, y, labels)
        data_with_features = [(f, target, label) for f, target, label in zip(features, y, labels)]
        # 筛选出label == 1的数据 用于后续的回归任务
        regression_data = [item for item in data_with_features if item[2] == 1]
        # print("val_step里，batch中，exp数据个数：", len(regression_data))
        classification_output = self.forward_classification(features)
        # print("分类器输出类型：", type(classification_output))
        # 损失函数类型定义
        loss_fn_classification = nn.BCELoss()
        # 计算分类损失
        val_loss_classification = loss_fn_classification(classification_output, labels)

        # 回归任务
        if len(regression_data)>0:
            # 筛选后的批量数据
            regression_features = torch.stack([item[0] for item in regression_data])
            target_var = torch.tensor([item[1] for item in regression_data])
            #print("val-目标带隙值输出类型和维度：", type(target_var), target_var.shape)
            # 调用回归模型进行预测
            # regression_output = self.forward_regression(regression_features)
            y_hat_bandgap, y_hat_uncertainty = self.forward_regression(regression_features)
            y_hat = torch.stack([y_hat_uncertainty.squeeze(), y_hat_bandgap.squeeze()], dim=1)
            
            diff = torch.abs(y - y_hat[:, 1]) # |Ptrue - P|
            p_loss = nn.MSELoss()(y, y_hat[:, 1])  # |Ptrue - P|
            u_loss = nn.MSELoss()(diff, y_hat[:, 0])  # (|Ptrue - P| - u)²
            val_loss_regression = p_loss + u_loss # 计算总损失 (Ptrue -P)² + (|Ptrue - P| - u)²

        # mean总损失 = mean回归损失 + mean分类损失（加权）
        val_loss_total = val_loss_regression + self.lambda_loss * val_loss_classification

        # 记录损失和相关信息
        self.log('val_total_MAE', val_loss_total, on_epoch=True, prog_bar=True, batch_size=len(target_var))
        self.log('val_r_loss', val_loss_regression, on_epoch=True, prog_bar=True, batch_size=len(target_var))
        self.log('val_c_loss', val_loss_classification, on_epoch=True, prog_bar=True, batch_size=len(labels))
        self.log('val_r_data_count', len(target_var), on_epoch=True, prog_bar=True)

        return val_loss_total




warnings.simplefilter("ignore")


# 使用本机上的cpu运行
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



class ProportionalSampler(Sampler):
    def __init__(self, dataset, com_ratio, exp_ratio, batch_size):
        """
        自定义采样器，确保每个批次中按比例选择 com 和 exp 数据
        :param dataset: 数据集
        :param com_ratio: com 数据集占比
        :param exp_ratio: exp 数据集占比
        :param batch_size: 每个批次的大小
        """
        # super().__init__(None, batch_size, False)  # 这里的 sampler 参数可以设置为 None
        self.dataset = dataset
        self.com_ratio = com_ratio
        self.exp_ratio = exp_ratio
        self.batch_size = batch_size

        # 获取com和exp数据的索引 i:索引值
        self.com_indices = [i for i, (_, output) in enumerate(dataset) if output[1] == 0]  # com数据的索引
        self.exp_indices = [i for i, (_, output) in enumerate(dataset) if output[1] == 1]  # exp数据的索引

        # 计算每个batch中包含的com和exp数据数量
        self.com_batch_size = int(batch_size * com_ratio)
        self.exp_batch_size = batch_size - self.com_batch_size

    def __iter__(self):
        """
        每次迭代返回一个batch的数据，确保每个batch中com和exp数据的比例
        """
        # 随机打乱com和exp数据的索引
        random.shuffle(self.com_indices)
        random.shuffle(self.exp_indices)

        # 计算多少个批次
        num_com_batches = len(self.com_indices) // self.com_batch_size
        num_exp_batches = len(self.exp_indices) // self.exp_batch_size
        num_batches = min(num_com_batches, num_exp_batches)

        # 为每个batch选择合适数量的样本
        for i in range(num_batches):
            com_batch = self.com_indices[i * self.com_batch_size : (i + 1) * self.com_batch_size]
            exp_batch = self.exp_indices[i * self.exp_batch_size : (i + 1) * self.exp_batch_size]

            # 合并com和exp的索引
            batch_indices = com_batch + exp_batch
            random.shuffle(batch_indices)  # 打乱每个batch内的顺序
            # print("查看batch_indices的类型：",type(batch_indices))
            # print("查看batch_indices：", batch_indices)

            yield batch_indices

    def __len__(self):
        return min(len(self.com_indices) // self.com_batch_size, len(self.exp_indices) // self.exp_batch_size)



def training_model(train_inputs,train_outputs,saved_name,lambda_grad_reverse, lambda_loss):
    dataset_train_norm = StruData(pd.Series(train_inputs), pd.Series(train_outputs))#用于创建标准化器

    exp_inputs, exp_outputs, _, _ = get_val_test_dataset()

    # 计算 A（com）和 B（exp）数据集的比例
    com_ratio = len(train_inputs) / (len(train_inputs) + len(exp_inputs))
    exp_ratio = 1 - com_ratio
    # print("比例：",com_ratio,exp_ratio)

    train_outputs = [(value, 0) for value in train_outputs]# 给其他数据集的数据加上标签0
    exp_outputs = [(value, 1) for value in exp_outputs]# 给实验数据加上标签1

    com_outputs_train, com_outputs_val = train_test_split(train_outputs, test_size=0.2, random_state=42)
    exp_outputs_train, exp_outputs_val = train_test_split(exp_outputs, test_size=0.2, random_state=42)
    com_inputs_train, com_inputs_val = train_test_split(train_inputs, test_size=0.2, random_state=42)
    exp_inputs_train, exp_inputs_val = train_test_split(exp_inputs, test_size=0.2, random_state=42)

    new_inputs_train = com_inputs_train + exp_inputs_train
    new_outputs_train = com_outputs_train + exp_outputs_train
    new_inputs_val = com_inputs_val + exp_inputs_val
    new_outputs_val = com_outputs_val + exp_outputs_val

    new_inputs_train = pd.Series(new_inputs_train)
    new_outputs_train = pd.Series(new_outputs_train)
    new_inputs_val = pd.Series(new_inputs_val)
    new_outputs_val = pd.Series(new_outputs_val)

    # print("StruDataDat之前：")
    # print("第1条输入数据",new_inputs_train[0])
    # print("第1条输出数据", new_outputs_train[0])

    dataset_train = StruDataDat(new_inputs_train, new_outputs_train)
    dataset_val = StruDataDat(new_inputs_val, new_outputs_val)

    # print("StruDataDat之后：")
    # print("第1条数据的label", dataset_train[0][])

    # 创建自定义采样器
    train_sampler = ProportionalSampler(dataset_train, com_ratio, exp_ratio, batch_size=128)
    val_sampler = ProportionalSampler(dataset_val, com_ratio, exp_ratio, batch_size=128)

    # 创建训练和验证的 DataLoader
    train_loader = DataLoader(
        dataset=dataset_train,
        #batch_size=128,
        batch_sampler=train_sampler,  # 使用自定义的采样器
        collate_fn=collate_pool_Dat,  # 自定义的数据整理函数collate_pool
        num_workers=4  # 启动 4 个子进程同时进行数据加载
    )

    val_loader = DataLoader(
        dataset=dataset_val,
        #batch_size=128,
        batch_sampler=val_sampler,  # 使用自定义的采样器
        collate_fn=collate_pool_Dat,
        num_workers=4
    )

    # 定义normalizer
    if len(dataset_train_norm) < 500:
        sample_data_list = [dataset_train_norm[i] for i in range(len(dataset_train_norm))]
    else:
        sample_data_list = [dataset_train_norm[i] for i in
                            sample(range(len(dataset_train_norm)), 500)]
    _, sample_target = collate_pool(sample_data_list)
    normalizer_train = Normalizer(sample_target)

    # 构建模型
    structures, _, = dataset_train_norm[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    # model_t = Cgcnn_lightning(ConvNetDat(orig_atom_fea_len,nbr_fea_len), normalizer_train)

    print("------------------lambda_grad_reverse为：", lambda_grad_reverse)
    print("------------------lambda_loss为：", lambda_loss)

    model_t = Cgcnn_lightning(convNetDat=ConvNetDat(orig_atom_fea_len, nbr_fea_len),
                              normalizer=normalizer_train,
                              lambda_grad_reverse=lambda_grad_reverse,
                              lambda_loss=lambda_loss)

    # 早停回调
    early_stop_callback = EarlyStopping(
        monitor="val_total_MAE",
        min_delta=0.00,
        patience=200,
        verbose=True,
        mode="min"
    )

    # CSVLogger，用于记录训练过程中的日志
    logger = CSVLogger("./logs", name=f"{saved_name}_DAT")

    # 模型检查点回调,用于保存模型参数
    # 它保存的是 模型的状态字典（state_dict），而不是整个模型对象，具体来说，保存的文件包含了模型的 参数，即 权重 和 偏置（通过 state_dict）。
    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_MAE',
        dirpath=f"./saved_models", # 临时保存最佳模型的路径,这里实际上保存的是检查点的数据，但是该数据也可以再加载模型阶段使用
        filename=f'{saved_name}_DAT_loss',
        save_top_k=1,
        mode='min'
    )

    # 保存normalizer
    normalizer_callback = SaveNormalizerCallback()

    # trainer = pl.Trainer(max_epochs=100000, callbacks=[early_stop_callback, checkpoint_callback,normalizer_callback], logger=logger)
    trainer = pl.Trainer(max_epochs=100000, callbacks=[early_stop_callback, checkpoint_callback, normalizer_callback],
                         logger=logger, use_distributed_sampler=False)

    # 训练模型
    print("开始训练...")
    trainer.fit(model_t, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("训练结束")

    return 0



all_datasets_name = [ 'scan','hse', 'gllb-sc','pbe']



for name in all_datasets_name:
    print(f"START===================================train {name} for p&u=======================================")

    # 从文件中加载最佳的 lambda_grad_reverse 和 lambda_loss，文件名包含数据集名称
    filename = f"./best_lambda/best_lambda_{name}.txt"
    try:
        with open(filename, 'r') as f:
            values = f.read().split()
            best_lambda_grad_reverse = float(values[0])
            best_lambda_loss = float(values[1])
        print(f"成功加载参数：lambda_grad_reverse = {best_lambda_grad_reverse}, lambda_loss = {best_lambda_loss}")
    except (FileNotFoundError, IndexError, ValueError):
        print(f"加载参数失败，文件 {filename} 可能不存在或内容格式有误，程序将退出。")
        sys.exit(1)

    structures, targets = get_low_fidelity_dataset(name)
    training_model(structures, targets, name,best_lambda_grad_reverse, best_lambda_loss)
    print(f"END===================================train {name} for p&u=======================================")
