import argparse
import os
import sys
import warnings

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model
from data import StruData, collate_pool
from load_datasets import get_sampled_low_fidelity_dataset, get_low_fidelity_dataset, get_val_test_dataset
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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


# def load_model(inputs,outputs,path):
#     inputs = pd.Series(inputs)
#     outputs = pd.Series(outputs)
#
#     dataset = StruData(inputs, outputs)
#
#     checkpoint = torch.load(path)
#     normalizer = checkpoint['normalizer']  # 从保存的文件中加载标准化器，适应训练数据的normalizer,防止数据泄露
#
#     structures, _, = dataset[0]
#     orig_atom_fea_len = structures[0].shape[-1]
#     nbr_fea_len = structures[1].shape[-1]
#
#     model = Cgcnn_lightning(CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
#                                                 atom_fea_len=args.atom_fea_len,
#                                                 n_conv=3,
#                                                 h_fea_len=128,
#                                                 n_h=1,
#                                                 classification=False), normalizer)
#     # 加载模型权重
#     model.load_state_dict(checkpoint['state_dict'])
#
#     return model

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


def predict(structures, targets, model):

    test_inputs = pd.Series(structures)
    test_outputs = pd.Series(targets)
    dataset = StruData(test_inputs, test_outputs)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=len(test_inputs),
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_pool)

    model.eval()
    predictions = []

    with torch.no_grad():
        for input_s, target in test_loader:
            output = model(*input_s)
            predictions.extend(output.tolist())

    # mae_errors.avg,
    return predictions

def get_train_datasets_descriptors(name):
    model.now_descriptor = []
    if name == 'scan':
        structures, targets = get_low_fidelity_dataset(name)
    else:
        structures, targets = get_sampled_low_fidelity_dataset(name)
    cgcnn_model = load_model(structures, targets, path=f"./saved_models/{name}")
    pre_targets = predict(structures, targets, cgcnn_model)
    descriptors = model.now_descriptor.copy()
    file_path = f"./saved_descriptors/{name}/train_data.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {
        'descriptors': descriptors,
        'targets': targets
    }
    torch.save(train_data_dict, file_path)

    print(f"train_data_dict:{name}保存完成！", len(descriptors), len(targets))


def get_val_datasets_descriptors(name):
    model.now_descriptor = []

    val_structures, val_targets,_,_ = get_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model(val_structures, val_targets, path=f"./saved_models/{name}")
    # 通过预测方法得到数据集所对应的描述子
    pre_targets = predict(val_structures, val_targets, cgcnn_model)

    val_descriptors = model.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}/val_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    val_data_dict = {
        'descriptors': val_descriptors,
        'targets': val_targets
    }
    torch.save(val_data_dict, file_path)
    print(f"val_data_dict:{name}保存完成！", len(val_descriptors), len(val_targets))

def get_test_datasets_descriptors(name):
    model.now_descriptor = []

    _,_,test_structures, test_targets = get_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model(test_structures, test_targets, path=f"./saved_models/{name}")
    # 通过预测方法得到数据集所对应的描述子
    pre_targets = predict(test_structures, test_targets, cgcnn_model)

    test_descriptors = model.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}/test_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    test_data_dict = {
        'descriptors': test_descriptors,
        'targets': test_targets
    }
    torch.save(test_data_dict, file_path)
    print(f"test_data_dict:{name}保存完成！", len(test_descriptors), len(test_targets))


datasets_name = ['scan','hse', 'gllb-sc', 'pbe']


for name in datasets_name:
    get_train_datasets_descriptors(name)
    get_val_datasets_descriptors(name)
    get_test_datasets_descriptors(name)

