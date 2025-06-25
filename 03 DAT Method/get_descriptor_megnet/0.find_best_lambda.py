from __future__ import annotations

import os
import warnings
import random
import pytorch_lightning as pl
import torch
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph, get_element_list
# from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
# from matgl.models import MEGNet
# from matgl.utils.training import ModelLightningModule
from data_megnet import MEGNetDataset, MGLDataLoader, collate_fn_DAT
from megnet_lightning import ModelLightningModule
from models_megnet import FeatureExtractor,MegnetRegression,MegnetClassification
from sklearn.model_selection import train_test_split

from matgl.layers import BondExpansion
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from load_datasets import get_low_fidelity_dataset, get_val_test_dataset
import numpy as np


def training_model(train_inputs,train_outputs,saved_name,lambda_grad_reverse, lambda_loss):
    delete_cache()
    # train_targets = torch.tensor(targets)
    exp_inputs, exp_outputs, _, _ = get_val_test_dataset()

    # 计算 A（com）和 B（exp）数据集的比例
    com_ratio = len(train_inputs) / (len(train_inputs) + len(exp_inputs))
    exp_ratio = 1 - com_ratio
    # print("比例：",com_ratio,exp_ratio)

    train_outputs = [(value, 0) for value in train_outputs]  # 给其他数据集的数据加上标签0
    exp_outputs = [(value, 1) for value in exp_outputs]  # 给实验数据加上标签1

    com_outputs_train, com_outputs_val = train_test_split(train_outputs, test_size=0.2, random_state=42)
    exp_outputs_train, exp_outputs_val = train_test_split(exp_outputs, test_size=0.2, random_state=42)
    com_inputs_train, com_inputs_val = train_test_split(train_inputs, test_size=0.2, random_state=42)
    exp_inputs_train, exp_inputs_val = train_test_split(exp_inputs, test_size=0.2, random_state=42)

    # com_outputs_train = torch.tensor(com_outputs_train)
    # com_outputs_val = torch.tensor(com_outputs_val)
    # exp_outputs_train = torch.tensor(exp_outputs_train)
    # exp_outputs_val = torch.tensor(exp_outputs_val)

    train_structures = com_inputs_train + exp_inputs_train
    val_structures = com_inputs_val + exp_inputs_val
    train_outputs = com_outputs_train + exp_outputs_train
    val_outputs = com_outputs_val + exp_outputs_val
    y_train = [y for y, _ in train_outputs]
    flag_train = [f for _, f in train_outputs]
    y_val = [y for y, _ in val_outputs]
    flag_val = [f for _, f in val_outputs]
    y_train = torch.tensor(y_train)
    flag_train = torch.tensor(flag_train)
    y_val = torch.tensor(y_val)
    flag_val = torch.tensor(flag_val)

    # converter_inputs = train_structures + val_structures
    # elem_list = get_element_list(converter_inputs)
    converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=5.0)
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    print("构建训练数据集...")
    train_dataset = MEGNetDataset(
        structures=train_structures,  # 函数接收到的structures已经是pymatgen类型的结构了
        labels={"bandgap": y_train, "flag": flag_train},
        converter=converter,
        filename="train_dgl_graph.bin",
        filename_state_attr="train_state_attr.pt",
        name="bandgap"
    )
    print("训练数据集构建完成！")

    print("构建验证数据集...")
    val_dataset = MEGNetDataset(
        structures=val_structures,
        labels={"bandgap": y_val, "flag": flag_val},
        converter=converter,
        filename="val_dgl_graph.bin",
        filename_state_attr="val_state_attr.pt",
        name="bandgap"
    )
    print("验证数据集构建完成！")

    train_loader, val_loader = MGLDataLoader(
        # com_inputs_train, com_outputs_train, exp_inputs_train, exp_outputs_train,
        # com_inputs_val, com_outputs_val, exp_inputs_val, exp_outputs_val,
        train_dataset,
        val_dataset,
        com_ratio, exp_ratio,
        collate_fn=collate_fn_DAT,
        batch_size=64,
        num_workers=0,
        converter=converter,
        pin_memory=True
    )

    # 实例化 FeatureExtractor 类
    feature_extractor = FeatureExtractor(
        dim_node_embedding=64,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 32, 32),
        nlayers_set2set=1,
        niters_set2set=3,
        activation_type="softplus",
        include_state=True,  # 默认值
        dropout=0.0,  # 默认值
        element_types=DEFAULT_ELEMENTS,  # 使用相同的元素列表
        bond_expansion=bond_expansion,
        cutoff=5.0,
        gauss_width=0.4
    )
    # 实例化 MegnetRegression 类
    megnet_regression = MegnetRegression(
        dim_blocks_out=32,
        hidden_layer_sizes_output=(32, 16),
        activation_type="softplus"
    )
    # 实例化 MegnetClassification 类
    megnet_classification = MegnetClassification(
        dim_blocks_out=32,
        hidden_layer_sizes_output=(32, 16),
        activation_type="sigmoid",
    )

    lit_module = ModelLightningModule(feature_extractor=feature_extractor,
                                      regressor=megnet_regression,
                                      classifier=megnet_classification,
                                      loss="mse_loss",
                                      lambda_grad_reverse=lambda_grad_reverse,
                                      lambda_loss=lambda_loss)

    logger = CSVLogger("logs", name=f"train_sampled_{saved_name}")

    early_stopping_callback = EarlyStopping(
        monitor='val_total_MAE',  # 选择监控的指标，例如验证集损失
        min_delta=0.0,  # 定义监控指标的变化阈值
        patience=30,  # 在没有改善的情况下等待停止的epoch数
        verbose=True,  # Print early stopping messages
        mode='min'  # 监控指标的模式，'min'表示越小越好，'max'表示越大越好
    )

    print("开始训练...")
    trainer = pl.Trainer(max_epochs=10000, logger=logger, callbacks=early_stopping_callback)
    # 修改Trainer配置，启用分布式训练
    # trainer = Trainer(
    #     max_epochs=10000,
    #     logger=logger,
    #     callbacks=[early_stopping_callback, checkpoint_callback],
    #     accelerator="cuda",  # 使用 CUDA 设备
    #     devices=2,  # 使用 2 个 GPU
    #     strategy="ddp",  # 使用分布式数据并行 (DDP)
    #     # accumulate_grad_batches=2  # 每两步梯度累积一次
    # )
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # 每个 epoch 结束后清理缓存
    torch.cuda.empty_cache()
    print("训练结束")
    delete_cache()
    return trainer.callback_metrics.get('val_total_MAE').item()


def delete_cache():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json","train_dgl_graph.bin", "train_lattice.pt", "train_dgl_line_graph.bin", "train_state_attr.pt", "train_labels.json","val_dgl_graph.bin", "val_lattice.pt", "val_dgl_line_graph.bin", "val_state_attr.pt", "val_labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    print("删除缓存完成")


warnings.simplefilter("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1,0"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(init_seed)

# lambda_grad_reverse_values = [0.0001, 0.001, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.05, 0.1]
# lambda_loss_values = [0.0001, 0.001, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.05, 0.1]
lambda_grad_reverse_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1]
lambda_loss_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1]

best_lambda_grad_reverse = None
best_lambda_loss = None
best_loss = float('inf')

# all_datasets_name = ['hse', 'gllb-sc', 'scan', 'pbe']
all_datasets_name = [ 'gllb-sc', 'scan', 'pbe']

for name in all_datasets_name:
    print(f"START===================================find {name} for p&u=======================================")
    structures, targets = get_low_fidelity_dataset(name)
    lambda_losses = {}
    for lambda_grad_reverse in lambda_grad_reverse_values:
        for lambda_loss in lambda_loss_values:
            print(f"Training with lambda_grad_reverse = {lambda_grad_reverse}, lambda_loss = {lambda_loss}")
            val_loss = training_model(structures, targets,name, lambda_grad_reverse, lambda_loss)
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