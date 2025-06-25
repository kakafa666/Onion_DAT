from __future__ import annotations

import os
import time

import gpytorch
import numpy as np
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '-1,0'


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

def get_hypers(name):
    file_path = f"./saved_best_hypers/{name}.pt"
    loaded_data = torch.load(file_path)
    lengthscale = loaded_data['lengthscale']
    lr = loaded_data['lr']
    noise = loaded_data['noise']

    return lengthscale, lr, noise

def get_train_dataset(name):
    train_data = torch.load(f'../get_descriptor_megnet/saved_descriptors/{name}/train_data.pt')

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

    return train_x,train_y

def get_val_dataset(name):
    val_data = torch.load(f'../get_descriptor_megnet/saved_descriptors/{name}/val_data.pt')

    X = np.array([e.detach().numpy() for e in val_data['descriptors']])
    Y = np.array([e.detach().numpy() for e in val_data['targets']])

    # 组合
    combined = list(zip(X, Y))

    # 遍历每一条数据，检查是否有 NaN 值
    nan_indices = []
    for i, data in enumerate(X):
        if np.isnan(data).any():
            nan_indices.append(i)

    combined_cleaned = [data for i, data in enumerate(combined) if i not in nan_indices]

    # 拆分清洗后的数据
    X_cleaned, Y_cleaned = zip(*combined_cleaned)

    # 将清洗后的数据转换为 NumPy 数组
    X_cleaned = np.array(X_cleaned)
    Y_cleaned = np.array(Y_cleaned)

    val_x = torch.tensor(X_cleaned, dtype=torch.float32)
    val_y = torch.tensor(Y_cleaned, dtype=torch.float32)

    return val_x,val_y

def train_with_SVGPR(name):
    lengthscale, lr, noise = get_hypers(name)

    train_x, train_y = get_train_dataset(name)
    val_x, val_y = get_val_dataset(name)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    inducing_points = train_x[:135, :]
    model = GPModel(inducing_points=inducing_points)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.noise = noise
    model.covar_module.base_kernel.lengthscale = lengthscale

    model.train()
    likelihood.train()

    num_epochs = 10000
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)

    # 设置早停参数
    patience = 50
    best_val_loss = float('inf')
    best_model_state = model.state_dict()

    no_improve_count = 0

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    epochs_iter = tqdm(range(num_epochs), desc="Epoch")

    for epoch in epochs_iter:
        model.train()

        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} - Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

        # 在验证集上评估模型性能
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                output_val = model(x_val)
                val_loss += -mll(output_val, y_val).item()
        val_loss /= len(val_loader)

        # 打印训练信息
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print('%s - Epoch %d/%d - Loss: %.3f  Val Loss: %.3f' % (
            current_time, epoch + 1, num_epochs, loss.item(), val_loss
        ))

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()
            no_improve_count = 0

        else:
            no_improve_count += 1

        # 检查是否需要早停
        if no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch+1}, best validation loss: {best_val_loss:.3f}')
            # 检查保存目录是否存在，如果不存在则创建
            if not os.path.exists('./saved_gpr_models'):
                os.makedirs('./saved_gpr_models')
            torch.save(best_model_state, f'./saved_gpr_models/{name}.pth')
            break



# 'scan','hse',,'pbe'
datasets_name = ['scan','hse','gllb-sc','pbe']

for n in datasets_name:
    train_with_SVGPR(n)


