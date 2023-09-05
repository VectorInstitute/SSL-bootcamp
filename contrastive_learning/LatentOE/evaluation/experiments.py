"""Experiment runner for running experiments with a given model configuration."""
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
from typing import Any, Dict, Tuple

import numpy as np
from config.base import Config
from torch.utils.data import DataLoader


sys.path.append("..")
from models.losses import DCL  # noqa: E402
from models.neutral_ad import TabNeutralAD  # noqa: E402
from models.neutral_ad_trainer import NeutralADTrainer  # noqa: E402
from utils import Logger  # noqa: E402


class ExperimentRunner:
    """Class for running an experiment using a model configuration.

    This class is used to run experiments with a given model configuration.
    It includes methods for running the test phase of the experiment.

    Parameters
    ----------
    model_configuration : Dict[str, Any]
        A dictionary containing the configuration parameters for the model.
    exp_path : str
        The path to save the experiment results.

    Attributes
    ----------
    model_config : Config
        The model configuration.
    exp_path : str
        The path to save the experiment results.

    """

    def __init__(self, model_configuration: Dict[str, Any], exp_path: str) -> None:
        """Initialize the experiment runner."""
        self.model_config = Config.from_dict(model_configuration)
        self.exp_path = exp_path

    def run_test(
        self,
        train_data: Any,
        val_data: Any,
        test_data: Any,
        logger: Logger,
        contamination: float,
        query_num: int,
    ) -> Tuple:
        """Run the test phase of the experiment.

        Parameters
        ----------
        train_data : Dataset
            The training dataset.
        val_data : Dataset
            The validation dataset.
        test_data : Dataset
            The test dataset.
        logger : Logger
            The logger for logging experiment progress.
        contamination : float
            The contamination rate in the data.
        query_num : int
            The query number.

        Returns
        -------
        tuple:
            A tuple containing the validation AUC, test AUC, test average precision
            (AP), test F1 score, and test anomaly scores.

        """
        optim_class = self.model_config.optimizer  # type: ignore
        sched_class = self.model_config.scheduler  # type: ignore
        stopper_class = self.model_config.early_stopper  # type: ignore
        network = self.model_config.network  # type: ignore

        try:
            x_dim = self.model_config["x_dim"]
        except:  # noqa
            x_dim = train_data.dim_features
        try:
            batch_size = self.model_config["batch_size"]
        except:  # noqa
            batch_size = int(np.ceil(len(train_data) / 4))

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=False
        )

        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = DataLoader(
                val_data, batch_size=batch_size, shuffle=False, drop_last=False
            )

        if len(test_data) == 0:
            test_loader = None
        else:
            test_loader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False, drop_last=False
            )

        model = TabNeutralAD(network(), x_dim, config=self.model_config)
        optimizer = optim_class(
            model.parameters(),
            lr=self.model_config["learning_rate"],
            weight_decay=self.model_config["l2"],
        )

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None

        trainer = NeutralADTrainer(
            model,
            loss_function=DCL(self.model_config["loss_temp"]),
            config=self.model_config,
        )

        val_loss, val_auc, test_auc, test_ap, test_f1, test_score = trainer.train(
            train_loader=train_loader,
            contamination=contamination,
            query_num=query_num,
            optimizer=optimizer,
            scheduler=scheduler,
            validation_loader=val_loader,
            test_loader=test_loader,
            early_stopping=stopper_class,
            logger=logger,
        )

        return val_auc, test_auc, test_ap, test_f1, test_score
