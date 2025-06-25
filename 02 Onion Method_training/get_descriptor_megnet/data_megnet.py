"""Tools to construct a dataset of DGL graphs."""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Callable

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


def collate_fn(batch, include_line_graph: bool = False):
    """Merge a list of dgl graphs to form a batch."""
    if include_line_graph:
        graphs, line_graphs, state_attr, labels = map(list, zip(*batch))
    else:
        graphs, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = torch.tensor([next(iter(d.values())) for d in labels], dtype=torch.float32)  # type: ignore
    state_attr = torch.stack(state_attr)
    if include_line_graph:
        l_g = dgl.batch(line_graphs)
        return g, l_g, state_attr, labels
    return g, labels, state_attr


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
    train_data: dgl.data.utils.Subset,
    val_data: dgl.data.utils.Subset,
    collate_fn: Callable,
    batch_size: int,
    num_workers: int,
    use_ddp: bool = False,
    pin_memory: bool = False,
    test_data: dgl.data.utils.Subset | None = None,
    generator: torch.Generator | None = None,
) -> tuple[GraphDataLoader, ...]:
    """Dataloader for MEGNet training.

    Args:
        train_data (dgl.data.utils.Subset): Training dataset.
        val_data (dgl.data.utils.Subset): Validation dataset.
        collate_fn (Callable): Collate function.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        use_ddp (bool, optional): Whether to use DDP. Defaults to False.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.
        test_data (dgl.data.utils.Subset | None, optional): Test dataset. Defaults to None.
        generator (torch.Generator | None, optional): Random number generator. Defaults to None.

    Returns:
        tuple[GraphDataLoader, ...]: Train, validation and test data loaders. Test data
            loader is None if test_data is None.
    """
    train_loader = GraphDataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_ddp=use_ddp,
        generator=generator,
    )

    val_loader = GraphDataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if test_data is not None:
        test_loader = GraphDataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
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




