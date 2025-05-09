"""Utility functions and classes for LOE."""
# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
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
#
# This source code is derived from A Fair Comparison of Graph Neural Networks for
# Graph Classification (ICLR 2020)(https://github.com/diningphil/gnn-comparison)
# Copyright (C)  2020  University of Pisa
# licensed under GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# The modifications include adjusting the arguments in the class 'Config'.
# The date of modifications: January, 2022

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import yaml
from sklearn.metrics import precision_recall_fscore_support


def read_config_file(dict_or_filelike: Union[Dict[str, str], str, Path]) -> Any:
    """Read a config file and return a dict.

    Supports JSON, YaML and pickle files.
    """
    if isinstance(dict_or_filelike, dict):
        return dict_or_filelike

    path = Path(dict_or_filelike)
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif path.suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            pickle.load(f)

    raise ValueError("Only JSON, YaML and pickle files supported.")


class Logger:
    """Simple logger class."""

    def __init__(self, filepath: str, mode: Literal["w", "a"], lock: Any = None):
        """Init logger.

        Parameters
        ----------
        filepath : str
            Path to the log file.
        mode : Lieral["w", "a"]
            Mode to open the file.
        lock : Any, optional
            A shared lock for multi process write access.
        """
        self.filepath = filepath
        if mode not in ["w", "a"]:
            raise AssertionError("Mode must be one of w, r or a")
        self.mode = mode
        self.lock = lock

    def log(self, message: str) -> None:
        """Log a string to the file."""
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(message + "\n")
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


def compute_f1_score(target: npt.NDArray[Any], score: npt.NDArray[Any]) -> Union[float, npt.NDArray[Any]]:
    """Compute the F1 score given the target and predicted scores."""
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    _, _, f1, _ = precision_recall_fscore_support(target, pred, average="binary")

    return f1


class EarlyStopper:
    """Base class for early stopping."""

    def stop(
        self,
        epoch: int,
        val_loss: float,
        val_auc: Optional[float] = None,
        test_loss: Optional[float] = None,
        test_auc: Optional[float] = None,
        test_ap: Optional[float] = None,
        test_f1: Optional[float] = None,
        test_score: Optional[float] = None,
        train_loss: Optional[float] = None,
    ) -> bool:
        """Return True if the training should stop."""
        raise NotImplementedError("Implement this method!")

    def get_best_val_metrics(
        self,
    ) -> Tuple[float, float, float, float, float, float, float, float, int]:
        """Return the best validation metrics."""
        return (
            self.train_loss,  # type: ignore
            self.val_loss,  # type: ignore
            self.val_auc,  # type: ignore
            self.test_loss,  # type: ignore
            self.test_auc,  # type: ignore
            self.test_ap,  # type: ignore
            self.test_f1,  # type: ignore
            self.test_score,  # type: ignore
            self.best_epoch,  # type: ignore
        )


class Patience(EarlyStopper):
    """Implement common "patience" technique."""

    def __init__(self, patience: int = 10, use_train_loss: bool = True) -> None:
        """Init."""
        self.local_val_optimum = float("inf")
        self.use_train_loss = use_train_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.train_loss = None
        (
            self.val_loss,
            self.val_auc,
        ) = (
            None,
            None,
        )
        self.test_loss, self.test_auc, self.test_ap, self.test_f1, self.test_score = (
            None,
            None,
            None,
            None,
            None,
        )

    def stop(
        self,
        epoch: int,
        val_loss: float,
        val_auc: Optional[float] = None,
        test_loss: Optional[float] = None,
        test_auc: Optional[float] = None,
        test_ap: Optional[float] = None,
        test_f1: Optional[float] = None,
        test_score: Optional[float] = None,
        train_loss: Optional[float] = None,
    ) -> bool:
        """Return True if the training should stop."""
        if self.use_train_loss:
            if train_loss <= self.local_val_optimum:  # type: ignore
                self.counter = 0
                self.local_val_optimum = train_loss  # type: ignore[assignment]
                self.best_epoch = epoch
                self.train_loss = train_loss  # type: ignore[assignment]
                self.val_loss, self.val_auc = val_loss, val_auc  # type: ignore
                (
                    self.test_loss,
                    self.test_auc,
                    self.test_ap,
                    self.test_f1,
                    self.test_score,
                ) = (
                    test_loss,  # type: ignore[assignment]
                    test_auc,  # type: ignore[assignment]
                    test_ap,  # type: ignore[assignment]
                    test_f1,  # type: ignore[assignment]
                    test_score,  # type: ignore[assignment]
                )
                return False
            self.counter += 1
            return self.counter >= self.patience
        if val_loss <= self.local_val_optimum:  # type: ignore
            self.counter = 0
            self.local_val_optimum = val_loss
            self.best_epoch = epoch
            self.train_loss = train_loss  # type: ignore[assignment]
            self.val_loss, self.val_auc = val_loss, val_auc  # type: ignore
            (
                self.test_loss,
                self.test_auc,
                self.test_ap,
                self.test_f1,
                self.test_score,
            ) = (
                test_loss,  # type: ignore[assignment]
                test_auc,  # type: ignore[assignment]
                test_ap,  # type: ignore[assignment]
                test_f1,  # type: ignore[assignment]
                test_score,  # type: ignore[assignment]
            )
            return False
        self.counter += 1
        return self.counter >= self.patience
