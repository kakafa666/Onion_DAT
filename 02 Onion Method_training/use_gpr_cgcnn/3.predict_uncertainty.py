from __future__ import annotations

import json
import os

import gpytorch
import numpy as np
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_train_dataset(name):
    train_data = torch.load(f'../get_descriptor_cgcnn/saved_descriptors/{name}/train_data.pt')

    X = np.array([e.detach().numpy() for e in train_data['descriptors']])
    Y = np.array([e.detach().numpy() for e in train_data['targets']])

    # 组合
    combined = list(zip(X, Y))
    # print("Length of dataset before cleaning: ", len(combined))

    # 遍历每一条数据，检查是否有 NaN 值
    nan_indices = []
    for i, data in enumerate(X):
        if np.isnan(data).any():
            nan_indices.append(i)

    # 打印包含 NaN 值的数据索引
    # print(f"Length of samples with NaN values: {len(nan_indices)}")

    combined_cleaned = [data for i, data in enumerate(combined) if i not in nan_indices]
    # print("Length of dataset after cleaning: ", len(combined_cleaned))

    # 拆分清洗后的数据
    X_cleaned, Y_cleaned = zip(*combined_cleaned)

    # 将清洗后的数据转换为 NumPy 数组
    X_cleaned = np.array(X_cleaned)
    Y_cleaned = np.array(Y_cleaned)

    train_x = torch.tensor(X_cleaned, dtype=torch.float32)
    train_y = torch.tensor(Y_cleaned, dtype=torch.float32)

    # print(f"{name}训练集输入的形状:{train_x.shape} 标签train_y形状:{train_y.shape}")
    return train_x,train_y


def get_test_dataset():
    test_data = torch.load(f'../get_descriptor_cgcnn/saved_descriptors/test/test_data.pt')

    X = np.array([e.detach().numpy() for e in test_data['descriptors']])
    Y = np.array([e.detach().numpy() for e in test_data['targets']])

    combined = list(zip(X, Y))

    nan_indices = []
    for i, data in enumerate(X):
        if np.isnan(data).any():
            nan_indices.append(i)

    combined_cleaned = [data for i, data in enumerate(combined) if i not in nan_indices]

    X_cleaned, Y_cleaned = zip(*combined_cleaned)

    # 将清洗后的数据转换为 NumPy 数组
    X_cleaned = np.array(X_cleaned)
    Y_cleaned = np.array(Y_cleaned)

    test_x = torch.tensor(X_cleaned, dtype=torch.float32)
    test_y = torch.tensor(Y_cleaned, dtype=torch.float32)

    return test_x, test_y

def predict_p_and_u():
    print(f"--------------------------------------")

    train_x, train_y = get_train_dataset('exp')

    state_dict = torch.load(f'./saved_gpr_models/model_5.pth')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    test_x, test_y = get_test_dataset()

    inducing_points = train_x[:135, :]
    model = GPModel(inducing_points=inducing_points)

    model.load_state_dict(state_dict)

    model.eval()
    likelihood.eval()

    # test_x = tuple(test_x)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x)) # observed_pred是一个gpytorch.distributions.MultivariateNormal类型的对象

    # 计算 MAE MSE
    y_hat = observed_pred.mean # 已经是张量形式了
    mae = torch.mean(torch.abs(y_hat - test_y))  # test_y也为张量
    mse = torch.mean((y_hat - test_y) ** 2)
    print(f"第C列-> MAE:{mae.item():.3f}  第D列-> MSE:{mse.item():.3f}")

    std = observed_pred.stddev.numpy()  # gpr模型标准差

    # return mae, mse, y_hat,std
    return 0



predict_p_and_u()


