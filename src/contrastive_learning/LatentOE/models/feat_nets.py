"""Feature networks for latent OE models."""

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
from typing import TYPE_CHECKING, Tuple

from torch import Tensor, nn


sys.path.append("..")
if TYPE_CHECKING:
    from config.base import Config


class FeatTransformNet(nn.Module):
    """Feature transformation network."""

    def __init__(self, x_dim: int, h_dim: int, bias: bool, num_layers: int) -> None:
        """Init."""
        super(FeatTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            net.append(nn.Linear(input_dim, h_dim, bias=bias))
            net.append(nn.BatchNorm1d(h_dim, affine=bias))
            net.append(nn.ReLU())
            input_dim = h_dim
        net.append(nn.Linear(input_dim, x_dim, bias=bias))

        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        out = self.net(x)

        return out


class FeatEncoder(nn.Module):
    """Feature encoder."""

    def __init__(self, x_dim: int, z_dim: int, bias: bool, num_layers: int, batch_norm: bool) -> None:
        """Init."""
        super(FeatEncoder, self).__init__()

        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, int(input_dim / 4), bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(int(input_dim / 4), affine=bias))
            enc.append(nn.ReLU())
            input_dim = int(input_dim / 4)

        self.fc = nn.Linear(input_dim, z_dim, bias=bias)
        self.enc = nn.Sequential(*enc)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        z = self.enc(x)
        z = self.fc(z)
        return z


class FeatNets:
    """Feature networks for latent OE models."""

    def _make_nets(self, x_dim: int, config: "Config") -> Tuple[nn.Module, nn.ModuleList]:
        """Return encoder and transformation networks."""
        enc_nlayers = config["enc_nlayers"]
        z_dim = config["enc_zdim"]
        trans_nlayers = config["trans_nlayers"]
        trans_hdim = config["trans_hdim"]
        num_trans = config["num_trans"]
        batch_norm = config["batch_norm"]

        enc = FeatEncoder(x_dim, z_dim, False, enc_nlayers, batch_norm)
        trans = nn.ModuleList([FeatTransformNet(x_dim, trans_hdim, False, trans_nlayers) for _ in range(num_trans)])

        return enc, trans
