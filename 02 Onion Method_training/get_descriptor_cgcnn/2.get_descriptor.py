import argparse
import os
import sys
import warnings
import glob

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model
from data import StruData, collate_pool
from load_datasets import get_sampled_low_fidelity_dataset, get_low_fidelity_dataset, get_train_val_test_dataset
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



# def load_model_from_checkpoint(inputs,outputs, path):
#     checkpoint = torch.load(path)
#     normalizer = checkpoint['normalizer']  # 从保存的文件中加载标准化器，适应训练数据的normalizer,防止数据泄露
#     # 首先创建一个新的 CrystalGraphConvNet 实例
#
#     inputs = pd.Series(inputs)
#     outputs = pd.Series(outputs)
#     dataset = StruData(inputs,outputs)
#
#     structures, _, = dataset[0]
#     orig_atom_fea_len = structures[0].shape[-1]
#     nbr_fea_len = structures[1].shape[-1]
#
#     crystal_graph_conv_net = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
#                                                 atom_fea_len=args.atom_fea_len,
#                                                 n_conv=3,
#                                                 h_fea_len=128,
#                                                 n_h=1,
#                                                 classification=False)
#
#     # 创建一个新的 Cgcnn_lightning 实例
#     model_temp = Cgcnn_lightning(crystal_graph_conv_net, normalizer)
#
#     # 加载模型的状态字典（仅包含权重信息）
#     state_dict = checkpoint['state_dict']
#
#     # 过滤状态字典，只保留与 CrystalGraphConvNet 相关的键
#     # 通过去掉 'crystalGraphConvNet.' 前缀来调整键名
#     filtered_state_dict = {k.replace('crystalGraphConvNet.', ''): v for k, v in state_dict.items()}
#
#     # 加载过滤后的状态字典到 CrystalGraphConvNet 中
#     model_temp.crystalGraphConvNet.load_state_dict(filtered_state_dict)
#     return model_temp
def load_model_from_checkpoint(inputs,outputs, model_dir):
    # 找到指定目录下所有的检查点文件
    checkpoint_files = glob.glob(os.path.join(model_dir, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

    # 由于 save_top_k=1，只有一个检查点文件，直接获取该文件路径
    model_path = checkpoint_files[0]

    # 加载模型的状态字典（仅包含权重信息）
    state_dict = torch.load(model_path)['state_dict']
    normalizer = torch.load(model_path)['normalizer']

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

    # 过滤状态字典，只保留与 CrystalGraphConvNet 相关的键
    # 通过去掉 'crystalGraphConvNet.' 前缀来调整键名
    filtered_state_dict = {k.replace('crystalGraphConvNet.', ''): v for k, v in state_dict.items()}

    # 加载过滤后的状态字典到 CrystalGraphConvNet 中
    model_temp.crystalGraphConvNet.load_state_dict(filtered_state_dict)
    return model_temp


def predict(structures, targets, model_d):

    test_inputs = pd.Series(structures)
    test_outputs = pd.Series(targets)
    dataset = StruData(test_inputs, test_outputs)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_pool)

    model_d.eval()
    predictions = []

    with torch.no_grad():
        for input_s, target in test_loader:
            output = model_d(*input_s)
            predictions.extend(output.tolist())

    # mae_errors.avg,
    return predictions

def get_train_datasets_descriptors(name):
    model.now_descriptor = []
    if name == 'exp':
        structures, targets,_,_,_,_ = get_train_val_test_dataset()
    else:
        structures, targets = get_low_fidelity_dataset(name)
    cgcnn_model = load_model_from_checkpoint(structures, targets, model_dir=f"./saved_models/model_5")
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


def get_val_datasets_descriptors():
    model.now_descriptor = []
    _,_,val_structures, val_targets,_,_ = get_train_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model_from_checkpoint(val_structures, val_targets, model_dir=f"./saved_models/model_5")
    # 通过预测方法得到数据集所对应的描述子
    pre_targets = predict(val_structures, val_targets, cgcnn_model)

    val_descriptors = model.now_descriptor.copy()

    file_path = f"./saved_descriptors/val/val_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    val_data_dict = {
        'descriptors': val_descriptors,
        'targets': val_targets
    }
    torch.save(val_data_dict, file_path)
    print(f"val_data_dict:验证描述子保存完成！", len(val_descriptors), len(val_targets))

def get_test_datasets_descriptors():
    model.now_descriptor = []

    _,_,_,_,test_structures, test_targets = get_train_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model_from_checkpoint(test_structures, test_targets, model_dir=f"./saved_models/model_5")
    # 通过预测方法得到数据集所对应的描述子
    pre_targets = predict(test_structures, test_targets, cgcnn_model)

    test_descriptors = model.now_descriptor.copy()

    file_path = f"./saved_descriptors/test/test_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    test_data_dict = {
        'descriptors': test_descriptors,
        'targets': test_targets
    }
    torch.save(test_data_dict, file_path)
    print(f"test_data_dict:测试描述子保存完成！", len(test_descriptors), len(test_targets))


# all_datasets_name = ['exp', 'gllb-sc', 'hse', 'scan', 'pbe', 'exp', 'hse', 'scan', 'pbe', 'exp', 'hse', 'pbe', 'exp', 'hse', 'exp']
# all_datasets_name = [ (0,'exp'),(1, 'gllb-sc'), (2, 'hse'), (3, 'scan'), (4, 'pbe'),
#                      (5, 'exp'), (6, 'hse'), (7, 'scan'), (8, 'pbe'), (9, 'exp'),
#                      (10, 'hse'), (11, 'pbe'), (12, 'exp'), (13, 'hse'), (14, 'exp')]
all_datasets_name=['exp','gllb-sc', 'hse', 'scan', 'pbe']

for name in all_datasets_name:
    get_train_datasets_descriptors(name)

get_val_datasets_descriptors()
get_test_datasets_descriptors()

