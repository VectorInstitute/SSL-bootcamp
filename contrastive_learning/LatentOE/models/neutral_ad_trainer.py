"""Trainer for the NeutralAD model."""
# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import sys
from typing import TYPE_CHECKING, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


sys.path.append("..")
from utils import Logger, Patience, compute_f1_score  # noqa: E402


if TYPE_CHECKING:
    from config.base import Config


class NeutralADTrainer:
    """Trainer for the NeutralAD model.

    This class implements the trainer for the NeutralAD model, which is used for
    anomaly detection. It includes methods for training the model and detecting
    outliers in the data.

    Parameters
    ----------
    model : nn.Module
        The NeutralAD model.
    loss_function : nn.Module
        The loss function used during training.
    config : Config
        A dictionary containing the configuration parameters.

    Attributes
    ----------
    loss_fun : nn.Module
        The loss function used during training.
    device : str
        The device on which the model and data are located.
    model : nn.Module
        The NeutralAD model.
    train_method : str
        The training method used for the model.
    max_epochs : int
        The maximum number of training epochs.
    warmup : int
        The number of warm-up epochs during training.

    """

    def __init__(
        self, model: nn.Module, loss_function: nn.Module, config: "Config"
    ) -> None:
        """Initialize the NeutralAD trainer."""
        self.loss_fun = loss_function
        self.device = torch.device(config["device"])
        self.model = model.to(self.device)
        self.train_method = config["train_method"]
        self.max_epochs = config["training_epochs"]
        self.warmup = 2

    def _train(
        self, epoch: int, train_loader: DataLoader, optimizer: Optimizer
    ) -> float:
        """Perform a single training epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        train_loader : DataLoader
            The training data loader.
        optimizer : torch.optim
            The optimizer used for training.

        Return
        ------
        float
            The average loss for the current epoch.

        """
        self.model.train()
        loss_all: torch.Tensor = torch.tensor(0.0).to(self.device)

        for data in train_loader:
            samples = data["sample"]
            labels = data["label"]

            z = self.model(samples)
            loss_n, loss_a = self.loss_fun(z)

            if epoch <= self.warmup:
                if self.train_method == "gt":
                    loss = torch.cat([loss_n[labels == 0], loss_a[labels == 1]], 0)
                    loss_mean = loss.mean()
                else:
                    loss = loss_n
                    loss_mean = loss.mean()
            else:
                score = loss_n - loss_a

                if self.train_method == "blind":
                    loss = loss_n
                    loss_mean = loss.mean()
                elif self.train_method == "loe_hard":
                    _, idx_n = torch.topk(
                        score,
                        int(score.shape[0] * (1 - self.contamination)),
                        largest=False,
                        sorted=False,
                    )
                    _, idx_a = torch.topk(
                        score,
                        int(score.shape[0] * self.contamination),
                        largest=True,
                        sorted=False,
                    )
                    loss = torch.cat([loss_n[idx_n], loss_a[idx_a]], 0)
                    loss_mean = loss.mean()
                elif self.train_method == "loe_soft":
                    _, idx_n = torch.topk(
                        score,
                        int(score.shape[0] * (1 - self.contamination)),
                        largest=False,
                        sorted=False,
                    )
                    _, idx_a = torch.topk(
                        score,
                        int(score.shape[0] * self.contamination),
                        largest=True,
                        sorted=False,
                    )
                    loss = torch.cat(
                        [loss_n[idx_n], 0.5 * loss_n[idx_a] + 0.5 * loss_a[idx_a]], 0
                    )
                    loss_mean = loss.mean()
                elif self.train_method == "refine":
                    _, idx_n = torch.topk(
                        loss_n,
                        int(loss_n.shape[0] * (1 - self.contamination)),
                        largest=False,
                        sorted=False,
                    )
                    loss = loss_n[idx_n]
                    loss_mean = loss.mean()
                elif self.train_method == "gt":
                    loss = torch.cat([loss_n[labels == 0], loss_a[labels == 1]], 0)
                    loss_mean = loss.mean()
                else:
                    raise ValueError(
                        f"Unknown training method: {self.train_method}. "
                        "Please choose from 'blind', 'loe_hard', 'loe_soft', 'refine', "
                        "or 'gt'."
                    )

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            loss_all += loss.sum()

        return loss_all.item() / len(train_loader.dataset)  # type: ignore

    def detect_outliers(self, loader: DataLoader) -> Tuple:
        """Detect outliers in the data using the trained model.

        Parameters
        ----------
        loader : DataLoader
            The data loader for which to detect outliers.

        Return
        ------
        tuple
            A tuple containing the area under the ROC curve (AUC), average precision
            (AP), F1 score, anomaly scores, inlier loss, and outlier loss.

        """
        model = self.model
        model.eval()

        loss_in: torch.Tensor = torch.tensor(0.0).to(self.device)
        loss_out: torch.Tensor = torch.tensor(0.0).to(self.device)
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                samples = data["sample"]
                labels = data["label"]

                z = model(samples)
                loss_n, _ = self.loss_fun(z)
                score = loss_n
                loss_in += loss_n[labels == 0].sum()
                loss_out += loss_n[labels == 1].sum()
                target_all.append(labels)
                score_all.append(score)

        score_all = torch.cat(score_all).cpu().numpy()
        target_all = np.concatenate(target_all)
        auc = roc_auc_score(target_all, score_all)
        ap = average_precision_score(target_all, score_all)
        f1 = compute_f1_score(target_all, score_all)
        return (
            auc,
            ap,
            f1,
            score_all,
            loss_in.item() / (target_all == 0).sum(),  # type: ignore
            loss_out.item() / (target_all == 1).sum(),  # type: ignore
        )

    def train(  # noqa: C901
        self,
        train_loader: DataLoader,
        contamination: float,
        optimizer: Optimizer,
        query_num: int = 0,
        scheduler: Optional[_LRScheduler] = None,
        validation_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        early_stopping: Optional[Type[Patience]] = None,
        logger: Optional[Logger] = None,
        log_every: int = 2,
    ) -> Tuple:
        """Train the NeutralAD model.

        Parameters
        ----------
        train_loader : DataLoader
            The data loader for training.
        contamination : float
            The contamination rate in the data.
        query_num : int, default=0
            The query number.
        optimizer : torch.optim.Optimizer, optional, default=None
            The optimizer used for training.
        scheduler : torch.optim.lr_scheduler, optional, default=None
            The learning rate scheduler.
        validation_loader : DataLoader, optional, default=None
            The data loader for validation.
        test_loader : DataLoader, optional, default=None
            The data loader for testing.
        early_stopping : Patience, optional, default=None
            The early stopping criteria.
        logger : Logger, optional, default=None
            The logger for logging training progress.
        log_every : int, default=2
            The frequency of logging training progress.

        Returns
        -------
        tuple:
            A tuple containing the validation loss, validation AUC, test AUC,
            test AP, test F1 score, test anomaly scores.

        """
        self.contamination = contamination
        early_stopper = early_stopping() if early_stopping is not None else None

        val_auc, val_f1, val_ap = -1.0, -1.0, -1.0
        test_auc, test_f1, test_ap, test_score = -1.0, -1.0, -1.0, None
        val_loss, valin_loss, valout_loss, testin_loss = None, None, None, None

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train(epoch, train_loader, optimizer)

            if scheduler is not None:
                scheduler.step()

            if test_loader is not None:
                (
                    test_auc,
                    test_ap,
                    test_f1,
                    test_score,
                    testin_loss,
                    testout_loss,
                ) = self.detect_outliers(test_loader)

            if validation_loader is not None:
                (
                    val_auc,
                    val_ap,
                    val_f1,
                    _,
                    valin_loss,
                    valout_loss,
                ) = self.detect_outliers(validation_loader)
                if epoch > self.warmup:
                    if early_stopper is not None and early_stopper.stop(
                        epoch,
                        valin_loss,
                        val_auc,
                        testin_loss,
                        test_auc,
                        test_ap,
                        test_f1,
                        test_score,
                        train_loss,
                    ):
                        break

            if epoch % log_every == 0 or epoch == 1:
                msg = (
                    f"Epoch: {epoch}, TR loss: {train_loss}, "
                    f"VAL loss: {valin_loss,valout_loss}, VL auc: {val_auc}, "
                    f"VL ap: {val_ap}, VL f1: {val_f1} "
                )

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

        if early_stopper is not None:
            (
                train_loss,
                val_loss,
                val_auc,
                test_loss,
                test_auc,
                test_ap,
                test_f1,
                test_score,
                best_epoch,
            ) = early_stopper.get_best_val_metrics()
            msg = (
                f"Stopping at epoch {best_epoch}, "
                f"TR loss: {train_loss}, VAL loss: {val_loss}, VAL auc: {val_auc},"
                f"TS loss: {test_loss}, TS auc: {test_auc}, TS ap: {test_ap}, "
                f"TS f1: {test_f1}"
            )
            if logger is not None:
                logger.log(msg)
                print(msg)
            else:
                print(msg)

        return val_loss, val_auc, test_auc, test_ap, test_f1, test_score
