"""Implementation of MatErials Graph Network (MEGNet) model.

Graph networks are a new machine learning (ML) paradigm that supports both relational reasoning and combinatorial
generalization. For more details on MEGNet, please refer to::

    Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. _Graph Networks as a Universal Machine Learning Framework for
    Molecules and Crystals._ Chem. Mater. 2019, 31 (9), 3564-3572. DOI: 10.1021/acs.chemmater.9b01294.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from dgl.nn import Set2Set
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.layers import MLP, ActivationFunction, BondExpansion, EdgeSet2Set, EmbeddingBlock, MEGNetBlock
from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    import dgl

    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__file__)


DEFAULT_ELEMENTS = ()

class FeatureExtractor(nn.Module, IOMixIn):
    """
    Feature extraction part of the MEGNet model.
    """
    __version__ = 1

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        ntypes_state: int | None = None,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        include_state: bool = True,
        dropout: float = 0.0,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion: BondExpansion | None = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        self.save_args(locals(), kwargs)

        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff
        self.bond_expansion = bond_expansion or BondExpansion(
            rbf_type="Gaussian", initial=0.0, final=cutoff + 1.0, num_centers=dim_edge_embedding, width=gauss_width
        )

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=len(self.element_types),
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]
        block_args = {
            "conv_hiddens": hidden_layer_sizes_conv,
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        # first block
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)] + [
            MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args)
            for _ in range(nblocks - 1)
        ]

        self.blocks = nn.ModuleList(blocks)

        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}
        self.edge_s2s = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)
        self.node_s2s = Set2Set(dim_blocks_out, **s2s_kwargs)

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(
    # features = self.feature_extractor(g, g.edata["edge_attr"], g.ndata["node_type"], state_attr) 看一下传入的这几个参数有没有错，每个参数对应什么意思？
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ):
        node_feat, edge_feat, state_feat = self.embedding(node_feat, edge_feat, state_feat)
        edge_feat = self.edge_encoder(edge_feat)
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)

        for block in self.blocks:
            output = block(graph, edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output

        node_vec = self.node_s2s(graph, node_feat)
        edge_vec = self.edge_s2s(graph, edge_feat)

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        state_feat = torch.squeeze(state_feat)

        vec = torch.hstack([node_vec, edge_vec, state_feat])

        if self.dropout:
            vec = self.dropout(vec)

        return vec

    def get_features(
            self,
            structure,
            state_feats: torch.Tensor | None = None,
            graph_converter: GraphConverter | None = None,
    ):
        """
        Get the features extracted from a structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            torch.Tensor: The extracted features.
        """
        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph
            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)
        g, state_feats_default = graph_converter.get_graph(structure)
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["edge_attr"] = self.bond_expansion(bond_dist)

        return self.forward(g, g.edata["edge_attr"], g.ndata["node_type"], state_feats)


class MegnetRegression(nn.Module, IOMixIn):
    """
    Prediction layer of MEGNet, including only regression tasks.
    """
    __version__ = 1

    def __init__(
        self,
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        activation_type: str = "softplus",
        **kwargs,
    ):
        super().__init__()

        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        dim_blocks_out = hidden_layer_sizes_conv[-1]

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation,
            activate_last=False,
        )

        # 新增的全连接层,用于输出不确定度u
        self.u_proj = MLP(
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation,
            activate_last=False,
        )

    def forward(self, vec: torch.Tensor):
        output = self.output_proj(vec)
        u = self.u_proj(vec)
        return torch.squeeze(output), torch.squeeze(u)

    def predict_structure(
        self,
        vec: torch.Tensor
    ):
        """
        Convenience method to directly predict property from features.

        Args:
            vec (torch.Tensor): The features extracted from a structure.

        Returns:
            output (torch.tensor): output property
        """
        output, u = self(vec)
        return output.detach(), u.detach()


class MegnetClassification(nn.Module, IOMixIn):
    """
    Classification part of the MEGNet model.
    """
    __version__ = 1

    def __init__(
            self,
            dim_blocks_out: int,
            hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
            activation_type: str = "softplus2",
    ):
        super().__init__()

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.output_proj = MLP(
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation,
            activate_last=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, vec: torch.Tensor):
        output = self.output_proj(vec)
        output = self.sigmoid(output)
        return torch.squeeze(output)


