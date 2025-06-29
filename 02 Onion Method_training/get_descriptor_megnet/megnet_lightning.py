"""Utils for training MatGL models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from matgl.apps.pes import Potential
from matgl.models import M3GNet

if TYPE_CHECKING:
    import dgl
    import numpy as np
    from torch.optim import Optimizer


class MatglLightningModuleMixin:
    """Mix-in class implementing common functions for training."""

    def training_step(self, batch: tuple, batch_idx: int):
        """Training step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.

        Returns:
           Total loss.
        """
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"train_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )

        return results["Total_Loss"]

    def on_train_epoch_end(self):
        """Step scheduler every epoch."""
        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch: tuple, batch_idx: int):
        """Validation step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.
        """
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"val_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )
        return results["Total_Loss"]

    def test_step(self, batch: tuple, batch_idx: int):
        """Test step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.
        """
        torch.set_grad_enabled(True)
        results, batch_size = self.step(batch)  # type: ignore
        self.log_dict(  # type: ignore
            {f"test_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,  # type: ignore
        )
        return results

    def configure_optimizers(self):
        """Configure optimizers."""
        if self.optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                eps=1e-8,
            )
        else:
            optimizer = self.optimizer
        if self.scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.decay_steps,
                eta_min=self.lr * self.decay_alpha,
            )
        else:
            scheduler = self.scheduler
        return [
            optimizer,
        ], [
            scheduler,
        ]

    def on_test_model_eval(self, *args, **kwargs):
        """
        Executed on model testing.

        Args:
            *args: Pass-through
            **kwargs: Pass-through.
        """
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.
            dataloader_idx: Data loader index.

        Returns:
            Prediction
        """
        torch.set_grad_enabled(True)
        return self.step(batch)


class ModelLightningModule(MatglLightningModuleMixin, pl.LightningModule):
    """A PyTorch.LightningModule for training MEGNet and M3GNet models."""

    def __init__(
        self,
        model,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss: str = "mse_loss",
        optimizer: Optimizer | None = None,
        scheduler=None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        sync_dist: bool = False,
        **kwargs,
    ):
        """
        Init ModelLightningModule with key parameters.

        Args:
            model: Which type of the model for training
            data_mean: average of training data
            data_std: standard deviation of training data
            loss: loss function used for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            lr: learning rate for training
            decay_steps: number of steps for decaying learning rate
            decay_alpha: parameter determines the minimum learning rate.
            sync_dist: whether sync logging across all GPU workers or not
            **kwargs: Passthrough to parent init.
        """
        super().__init__(**kwargs)

        self.model = model

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.data_mean = data_mean
        self.data_std = data_std
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha
        if loss == "mse_loss":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sync_dist = sync_dist
        self.save_hyperparameters()

    def forward(self, g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.Tensor | None = None):
        """Args:
            g: dgl Graph
            l_g: Line graph
            state_attr: State attribute.

        Returns:
            Model prediction.
        """
        if isinstance(self.model, M3GNet):
            return self.model(g=g, l_g=l_g, state_attr=state_attr)

        node_feat = g.ndata["node_type"]
        edge_feat = g.edata["edge_attr"]
        return self.model(g, edge_feat.float(), node_feat.long(), state_attr)

    def step(self, batch: tuple):
        """Args:
            batch: Batch of training data.

        Returns:
            results, batch_size
        """
        if isinstance(self.model, M3GNet):
            g, l_g, state_attr, labels = batch
            preds = self(g=g, l_g=l_g, state_attr=state_attr)
        else:
            g, labels, state_attr = batch
            preds = self(g=g, state_attr=state_attr)
            # print("预测值的形状为：", preds.shape)

        results = self.loss_fn(loss=self.loss, preds=preds, labels=labels)  # type: ignore
        batch_size = preds.numel()
        return results, batch_size

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每个训练批次结束时清理 GPU 内存"""
        torch.cuda.empty_cache()
        # 你可以根据需要添加其他逻辑
        # print(f"批次 Batch {batch_idx} finished, GPU memory 已清理!!!")

    def loss_fn(self, loss: nn.Module, labels: torch.Tensor, preds: torch.Tensor):
        """Args:
            loss: Loss function.
            labels: Labels to compute the loss.
            preds: Predictions.

        Returns:
            {"Total_Loss": total_loss, "MAE": mae, "RMSE": rmse}
        """
        scaled_pred = torch.reshape(preds * self.data_std + self.data_mean, labels.size())
        total_loss = loss(labels, scaled_pred)
        mae = self.mae(labels, scaled_pred)
        rmse = self.rmse(labels, scaled_pred)
        return {"Total_Loss": total_loss, "MAE": mae, "RMSE": rmse}


class PotentialLightningModule(MatglLightningModuleMixin, pl.LightningModule):
    """A PyTorch.LightningModule for training MatGL potentials.

    This is slightly different from the ModelLightningModel due to the need to account for energy, forces and stress
    losses.
    """

    def __init__(
        self,
        model,
        element_refs: np.ndarray | None = None,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.0,
        site_wise_weight: float = 0.0,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss: str = "mse_loss",
        optimizer: Optimizer | None = None,
        scheduler=None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        sync_dist: bool = False,
        **kwargs,
    ):
        """
        Init PotentialLightningModule with key parameters.

        Args:
            model: Which type of the model for training
            element_refs: element offset for PES
            energy_weight: relative importance of energy
            force_weight: relative importance of force
            stress_weight: relative importance of stress
            site_wise_weight: relative importance of additional site-wise predictions.
            data_mean: average of training data
            data_std: standard deviation of training data
            loss: loss function used for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            lr: learning rate for training
            decay_steps: number of steps for decaying learning rate
            decay_alpha: parameter determines the minimum learning rate.
            sync_dist: whether sync logging across all GPU workers or not
            **kwargs: Passthrough to parent init.
        """
        assert energy_weight >= 0, f"energy_weight has to be >=0. Got {energy_weight}!"
        assert force_weight >= 0, f"force_weight has to be >=0. Got {force_weight}!"
        assert stress_weight >= 0, f"stress_weight has to be >=0. Got {stress_weight}!"
        assert site_wise_weight >= 0, f"site_wise_weight has to be >=0. Got {site_wise_weight}!"

        super().__init__(**kwargs)

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.register_buffer("data_mean", torch.tensor(data_mean))
        self.register_buffer("data_std", torch.tensor(data_std))

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.site_wise_weight = site_wise_weight
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha

        self.model = Potential(
            model=model,
            element_refs=element_refs,
            calc_stresses=stress_weight != 0,
            calc_site_wise=site_wise_weight != 0,
            data_std=self.data_std,
            data_mean=self.data_mean,
        )
        if loss == "mse_loss":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sync_dist = sync_dist
        self.save_hyperparameters()

    def forward(self, g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.Tensor | None = None):
        """Args:
            g: dgl Graph
            l_g: Line graph
            state_attr: State attr.

        Returns:
            energy, force, stress, h
        """
        if self.model.calc_site_wise:
            e, f, s, h, m = self.model(g=g, l_g=l_g, state_attr=state_attr)
            return e, f.float(), s, h, m

        e, f, s, h = self.model(g=g, l_g=l_g, state_attr=state_attr)
        return e, f.float(), s, h

    def step(self, batch: tuple):
        """Args:
            batch: Batch of training data.

        Returns:
            results, batch_size
        """
        preds: tuple
        labels: tuple

        torch.set_grad_enabled(True)
        if self.model.calc_site_wise:
            g, l_g, state_attr, energies, forces, stresses, site_wise = batch
            e, f, s, _, m = self(g=g, state_attr=state_attr, l_g=l_g)
            preds = (e, f, s, m)
            labels = (energies, forces, stresses, site_wise)
        else:
            g, l_g, state_attr, energies, forces, stresses = batch
            e, f, s, _ = self(g=g, state_attr=state_attr, l_g=l_g)
            preds = (e, f, s)
            labels = (energies, forces, stresses)

        num_atoms = g.batch_num_nodes()
        results = self.loss_fn(
            loss=self.loss,  # type: ignore
            preds=preds,
            labels=labels,
            num_atoms=num_atoms,
        )
        batch_size = preds[0].numel()

        return results, batch_size

    def loss_fn(
        self,
        loss: nn.Module,
        labels: tuple,
        preds: tuple,
        num_atoms: int | None = None,
    ):
        """Compute losses for EFS.

        Args:
            loss: Loss function.
            labels: Labels.
            preds: Predictions
            num_atoms: Number of atoms.

        Returns::

            {
                "Total_Loss": total_loss,
                "Energy_MAE": e_mae,
                "Force_MAE": f_mae,
                "Stress_MAE": s_mae,
                "Energy_RMSE": e_rmse,
                "Force_RMSE": f_rmse,
                "Stress_RMSE": s_rmse,
            }

        """
        # labels and preds are (energy, force, stress, (optional) site_wise)
        e_loss = self.loss(labels[0] / num_atoms, preds[0] / num_atoms)
        f_loss = self.loss(labels[1], preds[1])

        e_mae = self.mae(labels[0] / num_atoms, preds[0] / num_atoms)
        f_mae = self.mae(labels[1], preds[1])

        e_rmse = self.rmse(labels[0] / num_atoms, preds[0] / num_atoms)
        f_rmse = self.rmse(labels[1], preds[1])

        s_mae = torch.zeros(1)
        s_rmse = torch.zeros(1)

        m_mae = torch.zeros(1)
        m_rmse = torch.zeros(1)

        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss

        if self.model.calc_stresses:
            s_loss = loss(labels[2], preds[2])
            s_mae = self.mae(labels[2], preds[2])
            s_rmse = self.rmse(labels[2], preds[2])
            total_loss = total_loss + self.stress_weight * s_loss

        if self.model.calc_site_wise:
            m_loss = loss(labels[3], preds[3])
            m_mae = self.mae(labels[3], preds[3])
            m_rmse = self.rmse(labels[3], preds[3])
            total_loss = total_loss + self.site_wise_weight * m_loss

        return {
            "Total_Loss": total_loss,
            "Energy_MAE": e_mae,
            "Force_MAE": f_mae,
            "Stress_MAE": s_mae,
            "Site_Wise_MAE": m_mae,
            "Energy_RMSE": e_rmse,
            "Force_RMSE": f_rmse,
            "Stress_RMSE": s_rmse,
            "Site_Wise_RMSE": m_rmse,
        }


def xavier_init(model: nn.Module) -> None:
    """Xavier initialization scheme for the model.

    Args:
        model (nn.Module): The model to be Xavier-initialized.
    """
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            if param.dim() < 2:
                bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[0])
                param.data.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                param.data.uniform_(-bound, bound)
