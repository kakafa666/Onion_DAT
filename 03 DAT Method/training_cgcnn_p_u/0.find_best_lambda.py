import sys

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from torch.utils.data import Sampler, DataLoader
import os
import warnings
from sklearn.model_selection import train_test_split
from random import seed

import warnings
from torch.utils.data import Sampler

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from random import seed
from model_conv import ConvNetDat
from data import StruData, StruDataDat, collate_pool,collate_pool_Dat
from load_datasets import get_low_fidelity_dataset,get_val_test_dataset

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


# 回归器模型
class CrystalGraphFc(nn.Module):
    def __init__(self):
        super(CrystalGraphFc, self).__init__()
        self.conv_to_fc = nn.Linear(64, 128)
        self.conv_to_fc_softplus = nn.Softplus()
        self.fc_out = nn.Linear(128, 1)

    def forward(self, crys_fea_conv):
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea_conv))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        out = self.fc_out(crys_fea)
        return out


# 自定义的分类器模型，继承自 torch.nn.Module
class DomainGraphClassifier(nn.Module):
    def __init__(self, input_size):
        super(DomainGraphClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


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
        self.feature_extractor = convNetDat
        self.crystalGraph_regression = CrystalGraphFc()
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
        regression_output = []
        target_var = []
        (x1, x2, x3, x4), (y, labels) = batch
        input_vars = (x1, x2, x3, x4)
        y = self.normalizer.norm(y)

        # 分类过程
        features = self.forward_conv(*input_vars)
        data_with_features = list(zip(features, y, labels))
        classification_data = [item for item in data_with_features]
        classification_features = torch.stack([item[0] for item in classification_data])
        classification_labels = torch.stack([item[2] for item in classification_data])
        # 应用梯度反转层
        reversed_features = GradientReversalLayer.apply(classification_features, self.lambda_grad_reverse)
        classification_output = self.forward_classification(reversed_features)
        loss_fn_classification = nn.BCELoss()
        train_loss_classification = loss_fn_classification(classification_output, classification_labels)

        # 回归任务
        train_loss_regression = torch.tensor(0.0)
        features_r = self.forward_conv(*input_vars)
        data_with_features_r = list(zip(features_r, y, labels))
        regression_data = [item for item in data_with_features_r if item[2] == 0]
        if len(regression_data) > 0:
            regression_features = torch.stack([item[0] for item in regression_data])
            target_var = torch.tensor([item[1] for item in regression_data])
            regression_output = self.forward_regression(regression_features)
            loss_fn_regression = nn.L1Loss()
            train_loss_regression = loss_fn_regression(regression_output, target_var)

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
        y = self.normalizer.norm(y)
        features = self.forward_conv(*input_vars)
        data_with_features = [(f, target, label) for f, target, label in zip(features, y, labels)]
        regression_data = [item for item in data_with_features if item[2] == 1]
        classification_output = self.forward_classification(features)
        loss_fn_classification = nn.BCELoss()
        val_loss_classification = loss_fn_classification(classification_output, labels)

        # 回归任务
        if len(regression_data) > 0:
            regression_features = torch.stack([item[0] for item in regression_data])
            target_var = torch.tensor([item[1] for item in regression_data])
            regression_output = self.forward_regression(regression_features)

        # 计算回归损失
        loss_fn_regression = nn.L1Loss()
        if regression_output.numel() > 0:
            val_loss_regression = loss_fn_regression(regression_output, target_var)
        else:
            val_loss_regression = torch.tensor(0.0)

        # mean总损失 = mean回归损失 + mean分类损失（加权）
        val_loss_total = val_loss_regression + self.lambda_loss * val_loss_classification

        # 记录损失和相关信息
        self.log('val_total_MAE', val_loss_total, on_epoch=True, prog_bar=True, batch_size=len(target_var))
        self.log('val_r_loss', val_loss_regression, on_epoch=True, prog_bar=True, batch_size=len(target_var))
        self.log('val_c_loss', val_loss_classification, on_epoch=True, prog_bar=True, batch_size=len(labels))
        self.log('val_r_data_count', len(target_var), on_epoch=True, prog_bar=True)

        return val_loss_total

    def test_step(self, batch, batch_idx):
        (x1, x2, x3, x4), (y, label) = batch
        input_var = (x1, x2, x3, x4)
        target_var = self.normalizer.norm(y)
        y_hat_regression = self.forward_regression(input_var)
        loss_fn = nn.L1Loss()
        test_loss = loss_fn(self.normalizer.denorm(y_hat_regression), target_var) if y_hat_regression is not None else torch.tensor(0.0)
        self.log('test_MAE', test_loss, on_epoch=True, prog_bar=True, batch_size=128)


class SaveNormalizerCallback(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['normalizer'] = pl_module.normalizer  # 保存 normalizer
        return checkpoint

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'normalizer' in checkpoint:
            pl_module.normalizer = checkpoint['normalizer']
        return checkpoint


class ProportionalSampler(Sampler):
    def __init__(self, dataset, com_ratio, exp_ratio, batch_size):
        self.dataset = dataset
        self.com_ratio = com_ratio
        self.exp_ratio = exp_ratio
        self.batch_size = batch_size
        self.com_indices = [i for i, (_, output) in enumerate(dataset) if output[1] == 0]
        self.exp_indices = [i for i, (_, output) in enumerate(dataset) if output[1] == 1]
        self.com_batch_size = int(batch_size * com_ratio)
        self.exp_batch_size = batch_size - self.com_batch_size

    def __iter__(self):
        random.shuffle(self.com_indices)
        random.shuffle(self.exp_indices)
        num_com_batches = len(self.com_indices) // self.com_batch_size
        num_exp_batches = len(self.exp_indices) // self.exp_batch_size
        num_batches = min(num_com_batches, num_exp_batches)
        for i in range(num_batches):
            com_batch = self.com_indices[i * self.com_batch_size: (i + 1) * self.com_batch_size]
            exp_batch = self.exp_indices[i * self.exp_batch_size: (i + 1) * self.exp_batch_size]
            batch_indices = com_batch + exp_batch
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return min(len(self.com_indices) // self.com_batch_size, len(self.exp_indices) // self.exp_batch_size)


def training_model(train_inputs, train_outputs, lambda_grad_reverse, lambda_loss):
    dataset_train_norm = StruData(pd.Series(train_inputs), pd.Series(train_outputs))

    exp_inputs, exp_outputs, _, _ = get_val_test_dataset()

    com_ratio = len(train_inputs) / (len(train_inputs) + len(exp_inputs))
    exp_ratio = 1 - com_ratio

    train_outputs = [(value, 0) for value in train_outputs]
    exp_outputs = [(value, 1) for value in exp_outputs]

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

    dataset_train = StruDataDat(new_inputs_train, new_outputs_train)
    dataset_val = StruDataDat(new_inputs_val, new_outputs_val)

    train_sampler = ProportionalSampler(dataset_train, com_ratio, exp_ratio, batch_size=128)
    val_sampler = ProportionalSampler(dataset_val, com_ratio, exp_ratio, batch_size=128)

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_sampler=train_sampler,
        collate_fn=collate_pool_Dat,
        num_workers=2
    )

    val_loader = DataLoader(
        dataset=dataset_val,
        batch_sampler=val_sampler,
        collate_fn=collate_pool_Dat,
        num_workers=2
    )

    if len(dataset_train_norm) < 500:
        sample_data_list = [dataset_train_norm[i] for i in range(len(dataset_train_norm))]
    else:
        sample_data_list = [dataset_train_norm[i] for i in
                            random.sample(range(len(dataset_train_norm)), 500)]
    _, sample_target = collate_pool(sample_data_list)
    normalizer_train = Normalizer(sample_target)

    structures, _, = dataset_train_norm[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model_t = Cgcnn_lightning(convNetDat=ConvNetDat(orig_atom_fea_len, nbr_fea_len),
                              normalizer=normalizer_train,
                              lambda_grad_reverse=lambda_grad_reverse,
                              lambda_loss=lambda_loss)

    early_stop_callback = EarlyStopping(
        monitor="val_total_MAE",
        min_delta=0.00,
        patience=30,
        verbose=True,
        mode="min"
    )

    # logger = CSVLogger("./logs", name=f"lambda_{lambda_grad_reverse}_{lambda_loss}_DAT")

    normalizer_callback = SaveNormalizerCallback()

    trainer = pl.Trainer(max_epochs=100000, callbacks=[early_stop_callback, normalizer_callback])

    trainer.fit(model_t, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return trainer.callback_metrics.get('val_total_MAE').item()


warnings.simplefilter("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1,0"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(init_seed)

# 定义 lambda_grad_reverse 和 lambda_loss 参数的搜索范围
# lambda_grad_reverse_values = [0.0001, 0.001, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.05, 0.1]
# lambda_loss_values = [0.0001, 0.001, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.05, 0.1]
lambda_grad_reverse_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1]
lambda_loss_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1]

best_lambda_grad_reverse = None
best_lambda_loss = None
best_loss = float('inf')

all_datasets_name = ['pbe']
# all_datasets_name = ['hse']
for name in all_datasets_name:
    print(f"START===================================find {name} for p&u=======================================")
    structures, targets = get_low_fidelity_dataset(name)
    lambda_losses = {}
    for lambda_grad_reverse in lambda_grad_reverse_values:
        for lambda_loss in lambda_loss_values:
            print(f"Training with lambda_grad_reverse = {lambda_grad_reverse}, lambda_loss = {lambda_loss}")
            val_loss = training_model(structures, targets, lambda_grad_reverse, lambda_loss)
            lambda_losses[(lambda_grad_reverse, lambda_loss)] = val_loss
            print(f"Validation loss for lambda_grad_reverse = {lambda_grad_reverse}, lambda_loss = {lambda_loss}: {val_loss}")

    # 找到最佳的 lambda_grad_reverse 和 lambda_loss
    best_lambda_tuple = min(lambda_losses, key=lambda_losses.get)
    best_lambda_grad_reverse, best_lambda_loss = best_lambda_tuple
    best_loss = lambda_losses[best_lambda_tuple]
    print(f"Best lambda_grad_reverse: {best_lambda_grad_reverse}, Best lambda_loss: {best_lambda_loss}, with validation loss: {best_loss}")

    # 将最佳的 lambda_grad_reverse 和 lambda_loss 保存到文件，文件名包含数据集名称
    filename = f"./best_lambda/best_lambda_{name}.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(f"{best_lambda_grad_reverse} {best_lambda_loss}")

    # 将 lambda_losses 字典的所有内容保存到文件
    all_lambda_filename = f"./best_lambda/all_lambda_{name}.txt"
    os.makedirs(os.path.dirname(all_lambda_filename), exist_ok=True)
    with open(all_lambda_filename, 'w') as f:
        f.write("lambda_grad_reverse    lambda_loss      total_loss\n")
        for (lambda_grad_reverse, lambda_loss), val_loss in lambda_losses.items():
            f.write(f"{lambda_grad_reverse} {lambda_loss} {val_loss}\n")

    print(f"END===================================find {name} for p&u=======================================")