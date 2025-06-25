import argparse
import os
import sys
import warnings

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model_conv_descriptor
from model_conv_descriptor import ConvNetDat
from data import StruData, collate_pool
from load_datasets import get_sampled_low_fidelity_dataset, get_low_fidelity_dataset, get_val_test_dataset
# from model import CrystalGraphConvNet

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

# 回归器模型
class CrystalGraphFc(nn.Module):
    def __init__(self):
        super(CrystalGraphFc, self).__init__()
        self.conv_to_fc = nn.Linear(64, 128)
        self.conv_to_fc_softplus = nn.Softplus()
        self.fc_out = nn.Linear(128, 1)

    def forward(self, crys_fea_conv):
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea_conv))
        # print("after conv_to_fc atom_fea: ", crys_fea.shape)

        crys_fea = self.conv_to_fc_softplus(crys_fea)
        # print("after conv_to_fc_softplus atom_fea: ", crys_fea.shape)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)
        # print("带隙预测值 output shape:", out.shape)

        return out


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

    # def forward_conv(self, *inputs):
    #     return self.feature_extractor(*inputs)

    # def forward_regression(self, features):
    #     return self.crystalGraph_regression(features)

    def validation_step(self, batch):
        x, y = batch                                      
        input_var = (x[0], x[1], x[2], x[3])
        target_var = self.normalizer.norm(y)
        # 回归任务
        y_hat = self.forward(input_var)

        loss_fn = nn.L1Loss()  # mae
        val_loss = loss_fn(y_hat, target_var)

        self.log('val_MAE', val_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return val_loss


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

    with torch.no_grad():
        for input_t, target in test_loader:
            output = model_t(*input_t)
            # 只反标准化维度为0的值，因为维度为0的值是预测的带隙值，而其他的维度可能额是材料的其他性质，在这里没用上，所以不用管
            if output.dim() == 2:
                output[:, 0] = model_t.normalizer.denorm(output[:, 0])
            predictions.extend(output.tolist())

    # mae_errors.avg,
    return predictions


def get_train_datasets_descriptors(name):
    model_conv_descriptor.now_descriptor = []
    structures, targets = get_low_fidelity_dataset(name)
    cgcnn_model = load_model_from_checkpoint(structures, targets, path=f"./saved_models/{name}_DAT.ckpt")
    pre_targets = predict(structures, targets, cgcnn_model)
    descriptors = model_conv_descriptor.now_descriptor.copy()
    file_path = f"./saved_descriptors/{name}/train_data.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    train_data_dict = {
        'descriptors': descriptors,
        'targets': targets
    }
    torch.save(train_data_dict, file_path)

    print(f"train_data_dict:{name}保存完成！", len(descriptors), len(targets))

def get_val_datasets_descriptors(name):
    model_conv_descriptor.now_descriptor = []

    val_structures, val_targets,_,_ = get_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model_from_checkpoint(val_structures, val_targets, path=f"./saved_models/{name}_DAT.ckpt")
    # 通过预测方法得到数据集所对应的描述子
    pre_targets = predict(val_structures, val_targets, cgcnn_model)

    val_descriptors = model_conv_descriptor.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}/val_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    val_data_dict = {
        'descriptors': val_descriptors,
        'targets': val_targets
    }
    torch.save(val_data_dict, file_path)
    print(f"val_data_dict:{name}_val保存完成！", len(val_descriptors), len(val_targets))

def get_test_datasets_descriptors(name):
    model_conv_descriptor.now_descriptor = []

    _,_,test_structures, test_targets = get_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model_from_checkpoint(test_structures, test_targets, path=f"./saved_models/{name}_DAT.ckpt")
    # 通过预测方法得到数据集所对应的描述子
    pre_targets = predict(test_structures, test_targets, cgcnn_model)

    test_descriptors = model_conv_descriptor.now_descriptor.copy()

    file_path = f"./saved_descriptors/{name}/test_data.pt"
    # 创建保存路径的父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    test_data_dict = {
        'descriptors': test_descriptors,
        'targets': test_targets
    }
    torch.save(test_data_dict, file_path)
    print(f"test_data_dict:{name}_test保存完成！", len(test_descriptors), len(test_targets))


# datasets_name = ['scan','hse', 'pbe','gllb-sc']
datasets_name = ['scan','hse','gllb-sc','pbe']

for name in datasets_name:
    get_val_datasets_descriptors(name)
    get_test_datasets_descriptors(name)
    
for name in datasets_name:
    get_train_datasets_descriptors(name)


