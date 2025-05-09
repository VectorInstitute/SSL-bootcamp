"""Utility class for data loading."""

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
from typing import Any

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset class for loading data samples and labels."""

    def __init__(self, samples: Any, labels: Any) -> None:
        """Initialize the dataset."""
        self.labels = labels
        self.samples = samples
        self.dim_features = samples.shape[1]

    def __len__(self) -> int:  # type: ignore
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Return the data sample at the given index."""
        label = self.labels[idx]
        sample = self.samples[idx]
        return {"sample": sample, "label": label}
