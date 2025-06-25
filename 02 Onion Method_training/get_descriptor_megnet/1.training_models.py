
from __future__ import annotations

import os
import warnings

import pytorch_lightning as pl
import torch
from matgl.ext.pymatgen import Structure2Graph, get_element_list
# from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from data_megnet import MEGNetDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion
# from matgl.data.transformer import
from matgl.models import MEGNet
# from my_training import ModelLightningModule
# from matgl.utils.training import ModelLightningModule
from megnet_lightning import ModelLightningModule
from pymatgen.core import Structure
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from load_datasets import get_low_fidelity_dataset, get_train_val_test_dataset
from matgl.config import DEFAULT_ELEMENTS

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_my_model(idx):
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.4)

    model_t = MEGNet(
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

    # 构建模型目录路径
    model_dir = f"./checkpoints/model_{idx}/"
    # 获取该目录下所有的 .ckpt 文件
    ckpt_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]

    # 检查是否只有一个 .ckpt 文件
    if len(ckpt_files) == 1:
        checkpoint_path = os.path.join(model_dir, ckpt_files[0])
    else:
        raise ValueError(f"目录 {model_dir} 下应该只有一个 .ckpt 文件，但找到了 {len(ckpt_files)} 个。")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            new_key = key[6:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    results = model_t.load_state_dict(new_state_dict)

    # # 输出模型的所有参数和细节
    # print("加载的模型参数和细节：")
    # for name, param in model_t.named_parameters():
    #     print(f"{name}: {param.data}")

    print("模型加载成功！")

    return model_t


def training_model(structures,targets,idx, prev_model_idx):
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

    _,_,val_structures,val_targets,_,_ = get_train_val_test_dataset()
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

    if prev_model_idx > 0:
        # 当prev_model_path不为空时，加载该路径的模型
        model_t = load_my_model(prev_model_idx)
    else:
        model_t = MEGNet(
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

    lit_module = ModelLightningModule(model=model_t, loss="mse_loss")
    logger = CSVLogger("logs", name=f"train_model_{idx}")

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
        dirpath=f"checkpoints/model_{idx}",  # 临时保存最佳模型的路径
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
        devices=2,  # 使用 2 个 GPU
        strategy="ddp",  # 使用分布式数据并行 (DDP)
    )
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("训练结束")
    delete_cache()
    # # 获取最佳模型
    # best_model_path = checkpoint_callback.best_model_path  # 替换为实际的检查点路径
    # best_model = ModelLightningModule.load_from_checkpoint(checkpoint_path=best_model_path)
    # best_model = best_model.model # 获取其中具体的模型，类型是MEGNet
    # # 保存模型到指定路径
    # print(f"开始保存模型...")
    # save_path = f"./saved_models/model_{idx}"
    # metadata = {"description": f"train datasets", "idx": f"{idx}"}
    # best_model.save(save_path, metadata=metadata)
    # print("保存完成")
    # delete_cache()
    # return model_t
    return 0


def delete_cache():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json","train_dgl_graph.bin", "train_lattice.pt", "train_dgl_line_graph.bin", "train_state_attr.pt", "train_labels.json","val_dgl_graph.bin", "val_lattice.pt", "val_dgl_line_graph.bin", "val_state_attr.pt", "val_labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    print("删除缓存完成")


# (1,('gllb-sc', 'hse', 'scan', 'pbe','exp')),
#                       (2,('hse', 'scan', 'pbe', 'exp')),
#                       (3,('hse', 'pbe','exp')),
#                       (4,('pbe', 'exp')),
all_datasets_name = [(1, ('gllb-sc', 'hse', 'scan', 'pbe', 'exp')),
                     (2, ('hse', 'scan', 'pbe', 'exp')),
                     (3, ('hse', 'pbe', 'exp')),
                     (4, ('hse', 'exp')),
                     (5, 'exp')]

prev_model_idx = 0

for idx, name in all_datasets_name:
    print(idx, name)
    print(f"START===================================train {idx} for p&u=======================================")
    if idx == 5:
        structures, targets, _, _, _, _ = get_train_val_test_dataset()
        training_model(structures, targets, idx, prev_model_idx)
    else:
        all_structures = []
        all_targets = []
        for n in name:
            if n == 'exp':
                structures, targets, _, _, _, _ = get_train_val_test_dataset()
                all_structures.extend(structures)
                all_targets.extend(targets)
            else:
                structures, targets = get_low_fidelity_dataset(n)
                all_structures.extend(structures)
                all_targets.extend(targets)
        training_model(all_structures, all_targets, idx, prev_model_idx)
        # 获取上一个检查点保存的模型路径
        prev_model_idx = idx

        print(f"END===================================train {idx} for p&u=======================================")

# for idx, name in all_datasets_name:
#     print(idx, name)
#     all_structures = []
#     all_targets = []
#     print(f"START===================================train {idx} for p&u=======================================")
#     for n in name:
#         if n == 'exp':
#             structures, targets, _, _, _, _ = get_train_val_test_dataset()
#             all_structures.extend(structures)
#             all_targets.extend(targets)
#         else:
#             structures, targets = get_low_fidelity_dataset(n)
#             all_structures.extend(structures)
#             all_targets.extend(targets)
#     model= training_model(all_structures, all_targets, idx, prev_model_path)
#     # 获取上一个检查点保存的模型路径
#     prev_model_path = f'./saved_models/model_{idx}'
#
#     print(f"END===================================train {idx} for p&u=======================================")



