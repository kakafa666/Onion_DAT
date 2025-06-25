"""Utils for training MatGL models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from models_megnet import FeatureExtractor,MegnetRegression,MegnetClassification

if TYPE_CHECKING:
    import dgl
    import numpy as np
    from torch.optim import Optimizer



# 这两个地方的 self.lambda_ 分别承担了不同的功能，一个用于控制梯度反转的强度，另一个用于平衡不同任务损失的权重。
# 在计算总损失时乘以 self.lambda_，用于平衡不同任务损失的权重
# 在梯度反转时,用于控制梯度反转的强度
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input * -ctx.lambda_, None


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
        # print(f"---------------在training_step中，lambda_grad_reverse={self.lambda_grad_reverse},lambda_loss={self.lambda_loss}----------")
        g, real_bandgaps, flags, state_attr = batch

        # 确保特征提取器的所有参数都需要梯度
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        edge_feat = g.edata["edge_attr"]
        node_feat = g.ndata["node_type"]
        # 特征提取 self.model(g, edge_feat.float(), node_feat.long(), state_attr)
        features = self.feature_extractor(g, edge_feat.float(), node_feat.long(), state_attr)
        data_with_features = list(zip(features, real_bandgaps, flags))

        # 分类任务
        classification_data = [item for item in data_with_features]
        classification_features = torch.stack([item[0] for item in classification_data])
        classification_labels = torch.stack([item[2] for item in classification_data])
        # 应用梯度反转层
        reversed_features = GradientReversalLayer.apply(classification_features, self.lambda_grad_reverse)
        classification_output = self.classifier(reversed_features)
        loss_fn_classification = nn.BCELoss()
        train_loss_classification = loss_fn_classification(classification_output, classification_labels)

        # 回归任务
        regression_data = [item for item in data_with_features if item[2] == 0]
        if len(regression_data) > 0:
            regression_features = torch.stack([item[0].detach() for item in regression_data])
            regression_labels = torch.stack([item[1].detach() for item in regression_data])
            regression_output = self.regressor(regression_features)
            loss_fn_regression = self.loss
            train_loss_regression = loss_fn_regression(regression_output, regression_labels)

            # 更新上一个 batch 的回归损失
            self.prev_train_loss_regression = train_loss_regression
        else:
            train_loss_regression = self.prev_train_loss_regression

        # 总损失
        train_loss_total = train_loss_regression + self.lambda_loss * train_loss_classification

        # 获取优化器
        optimizer = self.optimizers()

        # 反向传播
        train_loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录损失
        self.log_dict(
            {
                "train_loss_t": train_loss_total,
                "train_loss_r": train_loss_regression,
                "train_loss_c": train_loss_classification
            },
            batch_size=len(real_bandgaps),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist
        )

        return train_loss_total

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

        g, real_bandgaps, flags, state_attr = batch

        edge_feat=g.edata["edge_attr"]
        node_feat=g.ndata["node_type"]
        # 特征提取 self.model(g, edge_feat.float(), node_feat.long(), state_attr)
        features = self.feature_extractor(g, edge_feat.float(), node_feat.long(), state_attr)
        data_with_features = list(zip(features, real_bandgaps, flags))

        # 分类任务
        classification_data = [item for item in data_with_features]
        classification_features = torch.stack([item[0] for item in classification_data])
        classification_labels = torch.stack([item[2] for item in classification_data])
        classification_output = self.classifier(classification_features)
        loss_fn_classification = nn.BCELoss()
        val_loss_classification = loss_fn_classification(classification_output, classification_labels)

        # 回归任务
        regression_data = [item for item in data_with_features if item[2] == 1]
        if len(regression_data) > 0:
            regression_features = torch.stack([item[0] for item in regression_data])
            regression_labels = torch.stack([item[1] for item in regression_data])
            regression_output = self.regressor(regression_features)
            loss_fn_regression = self.loss
            val_loss_regression = loss_fn_regression(regression_output, regression_labels)
            # 更新上一个 batch 的回归损失
            self.prev_val_loss_regression = val_loss_regression
        else:
            val_loss_regression = self.prev_val_loss_regression

        # 总损失
        val_loss_total = val_loss_regression + self.lambda_loss * val_loss_classification

        # 记录损失
        self.log_dict(
            {
                "val_total_MAE": val_loss_total,
                "val_loss_r": val_loss_regression,
                "val_loss_c": val_loss_classification
            },
            batch_size=len(real_bandgaps),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist
        )

        return val_loss_total

    def test_step(self, batch: tuple, batch_idx: int):
        """Test step.

        Args:
            batch: Data batch.
            batch_idx: Batch index.
        """
        torch.set_grad_enabled(True)
        results = self.validation_step(batch, batch_idx)
        self.log_dict(
            {f"test_{key.replace('val_', '')}": val for key, val in results.items()},
            batch_size=len(batch[1]),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist
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


class ModelLightningModule(MatglLightningModuleMixin, pl.LightningModule):
    """A PyTorch.LightningModule for training MEGNet models."""

    def __init__(
        self,
        feature_extractor,
        regressor,
        classifier,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss: str = "mse_loss",
        optimizer=None,
        scheduler=None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        sync_dist: bool = False,
        lambda_grad_reverse: float = 0.05,
        lambda_loss: float = 0.05,
        **kwargs,
    ):
        """
        Init ModelLightningModule with key parameters.

        Args:
            feature_extractor: Feature extraction model.
            regressor: Regression model.
            classifier: Classification model.
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

        self.feature_extractor = feature_extractor
        self.regressor = regressor
        self.classifier = classifier

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
        self.prev_train_loss_regression = None  # 将上一个 batch 的回归损失作为类的属性
        self.prev_val_loss_regression = None  # 将上一个 batch 的回归损失作为类的属性
        # 设置为手动优化
        self.automatic_optimization = False
        self.lambda_grad_reverse = lambda_grad_reverse
        self.lambda_loss = lambda_loss

