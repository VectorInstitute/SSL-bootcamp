"""NeutralAD model for anomaly detection."""
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
from typing import TYPE_CHECKING, Union

import torch
from torch import nn

from .feat_nets import FeatNets  # noqa: E402
from .tab_nets import TabNets  # noqa: E402


sys.path.append("..")
if TYPE_CHECKING:
    from config.base import Config  # noqa: E402


class TabNeutralAD(nn.Module):
    """Tabular Neutral Autoencoder for Anomaly Detection.

    This class implements a Tabular Neutral Autoencoder for anomaly detection.
    It consists of an encoder network and multiple transformation networks (trans)
    that help disentangle different aspects of the input data.

    Parameters
    ----------
    model : nn.Module
        The autoencoder model for disentangling the input data.
    x_dim : int
        The dimensionality of the input data.
    config : Config
        A dictionary containing configuration settings for the model.

    Attributes
    ----------
    enc : nn.Module
        The encoder network.
    trans : nn.ModuleList
        The list of transformation networks.
    num_trans : int
        The number of transformation networks.
    trans_type : str
        The type of transformation used, either 'forward' or 'residual'.
    device : str
        The device (CPU or GPU) to perform computations on.
    z_dim : int
        The dimensionality of the latent space representation.

    """

    def __init__(self, model: Union[FeatNets, TabNets], x_dim: int, config: "Config") -> None:
        """Initialize the Tabular Neutral Autoencoder."""
        super(TabNeutralAD, self).__init__()

        # Extract encoder and transformation networks from the given model.
        self.enc, self.trans = model._make_nets(x_dim, config)

        # Get the number of transformation networks.
        self.num_trans = config["num_trans"]

        # Get the type of transformation used, either 'forward' or 'residual'.
        self.trans_type = config["trans_type"]

        # Get the device (CPU or GPU) to perform computations on.
        self.device = config["device"]

        # Set the dimensionality of the latent space representation.
        try:
            self.z_dim = config["latent_dim"]
        except KeyError:
            if 32 <= x_dim <= 300:
                self.z_dim = 32
            elif x_dim < 32:
                self.z_dim = 2 * x_dim
            else:
                self.z_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Tabular Neutral Autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Return
        ------
            torch.Tensor: The latent representation of the input data.
        """
        # Convert the input data to the appropriate device (CPU or GPU).
        x = x.type("torch.FloatTensor").to(self.device)

        # Initialize a tensor to store the transformed versions of the input data.
        x_T = torch.empty(x.shape[0], self.num_trans, x.shape[-1]).to(x)

        # Apply the transformation networks to the input data.
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == "forward":
                x_T[:, i] = mask
            elif self.trans_type == "residual":
                x_T[:, i] = mask + x

        # Concatenate the original input data with the transformed versions.
        x_cat = torch.cat([x.unsqueeze(1), x_T], 1)

        # Encode the concatenated data to obtain the latent space representation.
        zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.num_trans + 1, self.z_dim)

        return zs
