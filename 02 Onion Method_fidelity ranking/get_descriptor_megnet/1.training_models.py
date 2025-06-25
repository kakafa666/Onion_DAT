
from __future__ import annotations

import os
import warnings

import pytorch_lightning as pl
import torch
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph, get_element_list
# from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from megnet_data import MEGNetDataset, MGLDataLoader, collate_fn

from matgl.layers import BondExpansion
from matgl.models import MEGNet
# from my_training import ModelLightningModule
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Structure
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from load_datasets import get_sampled_low_fidelity_dataset, get_low_fidelity_dataset, get_val_test_dataset

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def training_model(structures,targets,saved_name):
    delete_cache()
    train_targets = torch.tensor(targets)

    converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=5.0)
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    train_dataset = MEGNetDataset(
        structures=structures,
        labels={"bandgap": train_targets},
        converter=converter,
        filename="train_dgl_graph.bin",
        filename_state_attr="train_state_attr.pt",
        name="bandgap"
    )

    val_structures,val_targets,_,_ = get_val_test_dataset()
    val_targets = torch.tensor(val_targets)
    val_dataset = MEGNetDataset(
        structures=val_structures,
        labels={"bandgap": val_targets},
        converter=converter,
        filename="val_dgl_graph.bin",
        filename_state_attr="val_state_attr.pt",
        name="bandgap"
    )

    train_loader, val_loader = MGLDataLoader(
        train_data=train_dataset,
        val_data=val_dataset,
        collate_fn=collate_fn,
        batch_size=64,
        num_workers=4,
        pin_memory=True
    )

    model = MEGNet(
        dim_node_embedding=64,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 32, 32),
        nlayers_set2set=1,
        niters_set2set=3,
        hidden_layer_sizes_output=(32, 16),
        is_classification=False,
        activation_type="softplus",
        bond_expansion=bond_expansion,
        cutoff=5.0,
        gauss_width=0.4,
    )

    lit_module = ModelLightningModule(model=model, loss="mse_loss")
    logger = CSVLogger("logs", name=f"train_sampled_{saved_name}")

    early_stopping_callback = EarlyStopping(
        monitor='val_MAE',  # 选择监控的指标，例如验证集损失
        min_delta=0.0,  # 定义监控指标的变化阈值
        patience=200,  # 在没有改善的情况下等待停止的epoch数
        verbose=True,  # Print early stopping messages
        mode='min'  # 监控指标的模式，'min'表示越小越好，'max'表示越大越好
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_MAE',
        save_top_k=1,
        mode='min',
        dirpath=f"checkpoints/{saved_name}",  # 临时保存最佳模型的路径
        filename='{epoch:02d}-{val_MAE:.4f}'
    )

    print("开始训练...")
    # trainer = pl.Trainer(max_epochs=10000, logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    # 修改Trainer配置，启用分布式训练
    trainer = Trainer(
        max_epochs=10000,
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        accelerator="cuda",  # 使用 CUDA 设备
        devices=1,  # 使用 1 个 GPU
        # precision=16,  # 启用混合精度训练
        # strategy="ddp",  # 使用分布式数据并行 (DDP)
        # accumulate_grad_batches=2  # 每两步梯度累积一次
    )
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # 每个 epoch 结束后清理缓存
    torch.cuda.empty_cache()
    print("训练结束")

    # 获取最佳模型
    best_model_path = checkpoint_callback.best_model_path  # 替换为实际的检查点路径
    best_model = ModelLightningModule.load_from_checkpoint(checkpoint_path=best_model_path)
    best_model = best_model.model # 获取其中具体的模型，类型是MEGNet
    # 保存模型到指定路径
    print(f"开始保存 {saved_name} 模型...")
    save_path = f"./saved_models/{saved_name}"
    metadata = {"description": f"train sampled {saved_name} datasets", "training_set": f"{saved_name}"}
    best_model.save(save_path, metadata=metadata)
    print("保存完成")
    delete_cache()
    # return model
    return 0

def delete_cache():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json","train_dgl_graph.bin", "train_lattice.pt", "train_dgl_line_graph.bin", "train_state_attr.pt", "train_labels.json","val_dgl_graph.bin", "val_lattice.pt", "val_dgl_line_graph.bin", "val_state_attr.pt", "val_labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    print("删除缓存完成")

# , 'pbe'
all_datasets_name = ['hse', 'gllb-sc', 'pbe']

for name in all_datasets_name:
    print(f"START===================================train {name} for p&u=======================================")
    if name == 'scan':
        structures, targets = get_low_fidelity_dataset(name)
        training_model(structures, targets, name)
    else:
        structures, targets = get_sampled_low_fidelity_dataset(name)
        training_model(structures, targets, name)
    print(f"END===================================train {name} for p&u=======================================")





