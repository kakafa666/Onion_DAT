"""Tools to construct a dataset of DGL graphs."""
from __future__ import annotations

import random
from dgl.data.utils import Subset
import json
import os
from typing import TYPE_CHECKING, Callable
from typing import Optional
import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from tqdm import trange

from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
from matgl.layers import BondExpansion

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

from torch.utils.data import Sampler

class ProportionalSampler(Sampler):
    def __init__(self, dataset, com_ratio, exp_ratio, batch_size):
        self.dataset = dataset
        self.com_ratio = com_ratio
        self.exp_ratio = exp_ratio
        self.batch_size = batch_size
        # 分离 com 和 exp 数据的索引
        self.com_indices = []
        self.exp_indices = []
        for idx in range(len(dataset)):
            _, _, labels = dataset[idx]
            flag = labels["flag"]
            # 检查 flag 是否为 tensor 类型并进行相应比较
            if isinstance(flag, torch.Tensor):
                # 假设 flag 是标量 tensor
                if flag.item() == 0:
                    self.com_indices.append(idx)
                else:
                    self.exp_indices.append(idx)
            else:
                if flag == 0:
                    self.com_indices.append(idx)
                else:
                    self.exp_indices.append(idx)
        self.num_batches = len(dataset) // batch_size
        # print("数据集总长度：",len(dataset))
        # print("batch有几个批次：",self.num_batches)

    def __iter__(self):
        com_remaining = self.com_indices.copy()
        exp_remaining = self.exp_indices.copy()
        while com_remaining or exp_remaining:
            # 计算当前批次中 com 和 exp 数据的数量
            num_com = int(self.batch_size * self.com_ratio)
            num_exp = self.batch_size - num_com
            # 确保不会超出剩余索引的数量
            num_com = min(num_com, len(com_remaining))
            num_exp = min(num_exp, len(exp_remaining))
            # 随机选择 com 和 exp 数据的索引
            com_selected = random.sample(com_remaining, num_com)
            exp_selected = random.sample(exp_remaining, num_exp)
            # 从剩余索引中移除已选择的索引
            for idx in com_selected:
                com_remaining.remove(idx)
            for idx in exp_selected:
                exp_remaining.remove(idx)
            # 合并索引并打乱顺序
            batch_indices = com_selected + exp_selected
            random.shuffle(batch_indices)
            # print("当前批次的索引长度：", len(batch_indices))
            # print("当前批次的索引：", batch_indices)
            yield from batch_indices
    def __len__(self):
        return len(self.dataset)



def collate_fn(batch):
    """Merge a list of dgl graphs to form a batch."""
    graphs, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    # labels = torch.tensor([next(iter(d.values())) for d in labels], dtype=torch.float32)  # type: ignore
    labels = torch.tensor([next(iter(d.values())) for d in labels], dtype=torch.float32)  # type: ignore
    state_attr = torch.stack(state_attr)
    return g, labels, state_attr

def collate_fn_DAT(batch):
    # print("collate_fn_DAT中接收到的batch有几条数据:",len(batch))
    graphs, state_attr, labels = map(list, zip(*batch))
    # print("collate_fn_DAT中接收到的batch有几条数据:", len(state_attr))
    g = dgl.batch(graphs)
    # labels = torch.tensor([next(iter(d.values())) for d in labels], dtype=torch.float32)  # type: ignore
    # 初始化存储两个值的列表
    bandgap_values = []
    flag_values = []
    for d in labels:
        # 假设字典中有 "bandgap" 和 "flag" 两个键
        bandgap_values.append(d["bandgap"])
        flag_values.append(d["flag"])
    # 将列表转换为张量
    bandgap_tensor = torch.tensor(bandgap_values, dtype=torch.float32)
    flag_tensor = torch.tensor(flag_values, dtype=torch.float32)
    # print("collate_fn_DAT中所有flags值：",flag_tensor)
    state_attr = torch.stack(state_attr)
    return g, bandgap_tensor, flag_tensor, state_attr


def collate_fn_efs(batch):
    """Merge a list of dgl graphs to form a batch."""
    graphs, line_graphs, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    l_g = dgl.batch(line_graphs)
    e = torch.tensor([d["energies"] for d in labels], dtype=torch.float32)
    f = torch.vstack([d["forces"] for d in labels])
    s = torch.vstack([d["stresses"] for d in labels])
    state_attr = torch.stack(state_attr)
    return g, l_g, state_attr, e, f, s


def MGLDataLoader(
    train_dataset,
    val_dataset,
    com_ratio: float, exp_ratio: float,
    collate_fn: Callable,
    batch_size: int,
    num_workers: int,
    converter,
    use_ddp: bool = False,
    pin_memory: bool = False,
    generator: torch.Generator | None = None,
) -> tuple:

    train_sampler=ProportionalSampler(train_dataset, com_ratio, exp_ratio, batch_size)
    val_sampler=ProportionalSampler(val_dataset, com_ratio, exp_ratio, batch_size)

    # 创建训练数据加载器
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,  # 使用实际的 batch_size
        shuffle=False,  # 是否打乱数据
        sampler=train_sampler,  # 使用自定义采样器
        collate_fn=collate_fn,  # 使用传入的 collate_fn
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_ddp=False,
        drop_last=False,  # 检查是否丢弃最后一批数据
        generator=generator,
    )

    # 创建验证数据加载器
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,  # 使用实际的 batch_size
        shuffle=False,  # 验证集不需要打乱
        sampler=val_sampler,  # 使用自定义采样器
        collate_fn=collate_fn,  # 使用传入的 collate_fn
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_ddp=False,
    )
    # print(f"--------+++++__________Sampler type: {type(train_loader.sampler)}")

    # 如果有测试数据，创建测试数据加载器
    # if com_inputs_test is not None and exp_inputs_test is not None:
    #     test_dataset = ProportionalDataset(
    #         com_inputs_test, com_outputs_test, exp_inputs_test, exp_outputs_test,
    #         com_ratio, exp_ratio, batch_size, converter
    #     )
    #     test_loader = GraphDataLoader(
    #         test_dataset,
    #         batch_size=batch_size,  # 使用实际的 batch_size
    #         shuffle=False,  # 测试集不需要打乱
    #         collate_fn=collate_fn,  # 使用传入的 collate_fn
    #         num_workers=num_workers,
    #         pin_memory=pin_memory,
    #     )
    #     return train_loader, val_loader, test_loader

    return train_loader, val_loader


class MEGNetDataset(DGLDataset):
    """Create a dataset including dgl graphs."""

    def __init__(
            self,
            structures: list | None = None,
            labels: dict[str, list] | None = None,
            converter: GraphConverter | None = None,
            filename: str = "dgl_graph.bin",  # 默认缓存文件名
            filename_state_attr: str = "state_attr.pt",  # 默认状态属性缓存文件名
            initial: float = 0.0,
            final: float = 5.0,
            num_centers: int = 100,
            width: float = 0.5,
            name: str = "MEGNETDataset",
            graph_labels: list[int | float] | None = None,
            # **kwargs
    ):
        self.filename = filename
        self.filename_state_attr = filename_state_attr
        self.converter = converter
        self.structures = structures or []
        self.labels = labels or {}
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width
        self.graph_labels = graph_labels
        # super().__init__(name=name, **kwargs)
        super().__init__(name=name)

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not.

        Returns: True if file exists.
        """
        return os.path.exists(self.filename) and os.path.exists(self.filename_state_attr)

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = len(self.structures)
        graphs = []
        state_attrs = []
        bond_expansion = BondExpansion(
            rbf_type="Gaussian",
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        for idx in trange(num_graphs):
            structure = self.structures[idx]  # type: ignore
            graph, state_attr = self.converter.get_graph(structure)  # type: ignore
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["edge_attr"] = bond_expansion(bond_dist)
            graphs.append(graph)
            state_attrs.append(state_attr)
        if self.graph_labels is not None:
            if np.array(self.graph_labels).dtype == "int64":
                state_attrs = torch.tensor(self.graph_labels).long()
            else:
                state_attrs = torch.tensor(self.graph_labels)
        else:
            state_attrs = torch.tensor(state_attrs)
        self.graphs = graphs
        self.state_attr = state_attrs
        return self.graphs, self.state_attr

    def save(self):
        """Save dgl graphs and labels."""
        save_graphs(self.filename, self.graphs, self.labels)
        torch.save(self.state_attr, self.filename_state_attr)

    def load(self):
        """Load dgl graphs and labels."""
        self.graphs, self.labels = load_graphs(self.filename)
        self.state_attr = torch.load(self.filename_state_attr)

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        return (
            self.graphs[idx],
            self.state_attr[idx],
            {k: torch.tensor(v[idx]) for k, v in self.labels.items()},
        )  # type: ignore

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)




