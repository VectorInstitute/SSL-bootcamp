"""Loss functions for training the Latent Outlier Exposure (LOE) model."""
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
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCL(nn.Module):
    """Deep Clustering Loss (DCL) Module.

    This class implements the Deep Clustering Loss (DCL) module, which is used for
    clustering in unsupervised learning. The DCL calculates two types of loss:
    neighbor loss (loss_n) and anti-neighbor loss (loss_a) based on the similarity
    matrix between the input embeddings (z).

    Parameters
    ----------
    temperature : float, optional, default=0.1
        The temperature parameter for the DCL.

    Attributes
    ----------
    temp : float
        The temperature parameter for the DCL.

    """

    def __init__(self, temperature: float = 0.1) -> None:
        """Initialize the Deep Clustering Loss (DCL) module."""
        super(DCL, self).__init__()
        self.temp = temperature

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Deep Clustering Loss (DCL) module.

        Parameters
        ----------
        z : torch.Tensor
            The input embeddings.

        Return
        ------
        tuple
            A tuple containing two losses: neighbor loss (loss_n) and anti-neighbor
            loss (loss_a).

        """
        # Normalize the input embeddings (z) along the last dimension
        # (L2 normalization).
        z = F.normalize(z, p=2, dim=-1)

        # Split the input embeddings into the original embedding (z_ori)
        # and transformed embeddings (z_trans).
        z_ori = z[:, 0]  # n, z
        z_trans = z[:, 1:]  # n, k-1, z

        # Get the batch size, number of transformations (k), and dimension of
        # embeddings (z_dim).
        batch_size, num_trans, z_dim = z.shape

        # Calculate the similarity matrix between embeddings using exponential
        # of the dot product with temperature.
        sim_matrix = torch.exp(
            torch.matmul(z, z.permute(0, 2, 1) / self.temp)
        )  # n, k, k

        # Create a mask to remove the similarity of embeddings with themselves.
        mask = (
            torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)
        ).bool()

        # Apply the mask to the similarity matrix to get similarity values between
        # different embeddings.
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)

        # Calculate the sum of similarities for each transformation (trans_matrix)
        # to use in loss_n.
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n, k-1

        # Calculate the positive similarity between original and transformed
        # embeddings to use in loss_a.
        pos_sim = torch.exp(
            torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp
        )  # n, k-1

        # Calculate the scale factor for loss tensor normalization.
        K = num_trans - 1
        scale = 1 / np.abs(np.log(1.0 / K))

        # Calculate neighbor loss (loss_n) and anti-neighbor loss (loss_a).
        loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
        loss_n = loss_tensor.mean(1)
        loss_a = -torch.log(1 - pos_sim / trans_matrix) * scale
        loss_a = loss_a.mean(1)

        return loss_n, loss_a
