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

# import model
from data import StruData, collate_pool
from load_datasets import get_val_test_dataset
# from model_1 import CrystalGraphConvNet
from model_conv import ConvNetDat

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


def get_test_results(name):
    # model.now_descriptor = []
    _,_,test_structures, test_targets = get_val_test_dataset()
    # 加载训练好的模型
    cgcnn_model = load_model_from_checkpoint(test_structures, test_targets, path=f"./saved_models/{name}_DAT.ckpt")

    pre_targets = predict(test_structures, test_targets, cgcnn_model)
    pre_targets = [target[0] for target in pre_targets]

    targets_4_list.append(pre_targets)

    pre_targets_np = np.array(pre_targets)# 预测值
    test_targets_np = np.array(test_targets)# 真实值

    mae = np.mean(np.abs(pre_targets_np - test_targets_np))
    mse = np.mean((pre_targets_np - test_targets_np) ** 2)

    # mae = round(mae, 3)
    # mse = round(mse, 3)

    print(f"{name}  第C列->MAE: {mae:.3f}, 第D列->MSE: {mse:.3f}")
    return 0
    #return mae, mse


# G H列 非加权融合值
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

    print(f"================================= G H 列  END  ======================================")

    return 0



# all_datasets_name = ['scan','hse', 'pbe','gllb-sc']

_,_,_, test_y_real = get_val_test_dataset()  # test_y_real:真实值 1352个
targets_4_list = []  # 存放4个组的预测值 4*1352个


all_datasets_name = [ 'gllb-sc','pbe','scan','hse']
# all_datasets_name = ['hse']
for name in all_datasets_name:
    get_test_results(name)


get_mae_mse_integrated()