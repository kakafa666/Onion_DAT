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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


def get_hypers():
    file_path = "./saved_best_hypers/exp_train_hyper.pt"
    loaded_data = torch.load(file_path)
    lengthscale = loaded_data['lengthscale']
    lr = loaded_data['lr']
    noise = loaded_data['noise']

    return lengthscale, lr, noise


def get_train_dataset(n):
    train_data = torch.load(f'../get_descriptor_cgcnn/saved_descriptors/{n}/train_data.pt')
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

    # train_x = torch.tensor(X_cleaned, dtype=torch.float32)
    # train_y = torch.tensor(Y_cleaned, dtype=torch.float32)

    # print(f"{name}训练集输入的形状:{train_x.shape} 标签train_y形状:{train_y.shape}")
    return X_cleaned,Y_cleaned


def get_val_dataset():
    val_data = torch.load('../get_descriptor_cgcnn/saved_descriptors/val/val_data.pt')

    X = np.array([e.detach().numpy() for e in val_data['descriptors']])
    Y = np.array([e.detach().numpy() for e in val_data['targets']])

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

    val_x = torch.tensor(X_cleaned, dtype=torch.float32)
    val_y = torch.tensor(Y_cleaned, dtype=torch.float32)

    return val_x, val_y


def train_with_SVGPR(name,idx,pre_model_path):
    lengthscale, lr, noise = get_hypers()

    train_x=[]
    train_y = []
    print("name:",name)
    # for n in name:
    #     print(n)
    #     train_x_t, train_y_t = get_train_dataset(n)
    #     train_x.extend(train_x_t)
    #     train_y.extend(train_y_t)
    #
    # # 先将列表转换为 numpy.ndarray
    # train_x_np = np.array(train_x)
    # train_y_np = np.array(train_y)
    # train_x = torch.tensor(train_x_np, dtype=torch.float32)
    # train_y = torch.tensor(train_y_np, dtype=torch.float32)
    train_x_t, train_y_t = get_train_dataset('exp')
    # 再将 numpy.ndarray 转换为 PyTorch 张量
    train_x = torch.tensor(train_x_t,dtype=torch.float32)
    train_y = torch.tensor(train_y_t,dtype=torch.float32)

    val_x, val_y = get_val_dataset()

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    inducing_points = train_x[:135, :]
    model_train = GPModel(inducing_points=inducing_points)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.noise = noise
    model_train.covar_module.base_kernel.lengthscale = lengthscale

    # 如果 pre_model_path 不为空，加载预训练模型
    if pre_model_path:
        model_state = torch.load(pre_model_path)
        model_train.load_state_dict(model_state)

    model_train.train()
    likelihood.train()

    num_epochs = 10000
    optimizer = torch.optim.Adam([
        {'params': model_train.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)

    # 设置早停参数
    patience = 50
    best_val_loss = float('inf')
    best_model_state = model_train.state_dict()

    no_improve_count = 0

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model_train, num_data=train_y.size(0))

    epochs_iter = tqdm(range(num_epochs), desc="Epoch")

    for epoch in epochs_iter:
        model_train.train()

        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} - Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model_train(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

        # 在验证集上评估模型性能
        model_train.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                output_val = model_train(x_val)
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
            best_model_state = model_train.state_dict()
            no_improve_count = 0

        else:
            no_improve_count += 1

        # 检查是否需要早停
        if no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch + 1}, best validation loss: {best_val_loss:.3f}')
            if not os.path.exists('./saved_gpr_models'):
                os.makedirs('./saved_gpr_models')
            torch.save(best_model_state, f'./saved_gpr_models/model_{idx}.pth')
            break


all_datasets_name = [(1,('gllb-sc', 'hse', 'scan', 'pbe','exp',)),
                     (2,('hse', 'scan', 'pbe','exp')),
                     (3,('hse', 'pbe','exp')),
                     (4,('pbe','exp')),
                     (5,'exp')]
pre_model_path = None

for idx, name in all_datasets_name:
    print(idx, name)
    print(f"START===================================train {idx} for p&u=======================================")
    if idx == 5:
        train_with_SVGPR('exp',5, pre_model_path)
    else:
        train_with_SVGPR(name,idx, pre_model_path)
        pre_model_path=f'./saved_gpr_models/model_{idx}.pth'
    print(f"END===================================train {idx} for p&u=======================================")


