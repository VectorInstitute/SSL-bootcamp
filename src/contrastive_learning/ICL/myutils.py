"""Utility functions for ICL."""

import random
from typing import Dict, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


class Utils:
    """Utility Class for ICL."""

    # remove randomness
    def set_seed(self, seed: int) -> None:
        """Set a random seed."""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_device(self, gpu_specific: bool = False) -> torch.device:
        """Return torch.device."""
        if gpu_specific:
            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                print(f"Found {n_gpu} GPUs")
                print(f"cuda name: {torch.cuda.get_device_name(0)}")
            else:
                print("GPU is unavailable!")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        return device

    # generate unique value
    def unique(self, a: float, b: float) -> int:
        """Generate unique value."""
        u = 0.5 * (a + b) * (a + b + 1) + b
        return int(u)

    def data_description(self, X: np.ndarray, y: np.ndarray) -> None:
        """Print a description of the data."""
        des_dict = {}
        des_dict["Samples"] = X.shape[0]
        des_dict["Features"] = X.shape[1]
        des_dict["Anomalies"] = sum(y)
        des_dict["Anomalies Ratio(%)"] = round((sum(y) / len(y)) * 100, 2)

        print(des_dict)

    def metric(
        self, y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Return aucroc and aucpr."""
        aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
        aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=pos_label)

        return {"aucroc": aucroc, "aucpr": aucpr}

    def sampler(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the data."""
        index_u = np.where(y_train == 0)[0]
        index_a = np.where(y_train == 1)[0]

        n = 0
        while len(index_u) >= batch_size:
            self.set_seed(n)
            index_u_batch = np.random.choice(index_u, batch_size // 2, replace=False)
            index_u = np.setdiff1d(index_u, index_u_batch)

            index_a_batch = np.random.choice(index_a, batch_size // 2, replace=True)

            index_batch = np.append(index_u_batch, index_a_batch)  # batch index
            np.random.shuffle(index_batch)  # shuffle

            if n == 0:
                X_train_new = X_train[index_batch]
                y_train_new = y_train[index_batch]
            else:
                X_train_new = np.append(
                    X_train_new,
                    X_train[index_batch],
                    axis=0,  # type: ignore
                )
                y_train_new = np.append(
                    y_train_new,
                    y_train[index_batch],  # type: ignore
                )
            n += 1

        return X_train_new, y_train_new  # type: ignore

    def sampler_2(
        self, X_train: np.ndarray, y_train: np.ndarray, step: int, batch_size: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the data."""
        index_u = np.where(y_train == 0)[0]
        index_a = np.where(y_train == 1)[0]

        for i in range(step):
            index_u_batch = np.random.choice(index_u, batch_size // 2, replace=True)
            index_a_batch = np.random.choice(index_a, batch_size // 2, replace=True)

            # batch index
            index_batch = np.append(index_u_batch, index_a_batch)
            # shuffle
            np.random.shuffle(index_batch)

            if i == 0:
                X_train_new = X_train[index_batch]
                y_train_new = y_train[index_batch]
            else:
                X_train_new = np.append(
                    X_train_new,
                    X_train[index_batch],
                    axis=0,  # type: ignore
                )
                y_train_new = np.append(
                    y_train_new,
                    y_train[index_batch],  # type: ignore
                )

        return X_train_new, y_train_new  # type: ignore

    # for PReNet
    def sampler_pairs(
        self,
        X_train_tensor: torch.Tensor,
        y_train: torch.Tensor,
        epoch: int,
        batch_num: int,
        batch_size: int,
        s_a_a: float,
        s_a_u: float,
        s_u_u: float,
    ) -> tuple:
        """Return a pair of dataloaders - one for X and one for y.

        Parameters
        ----------
        X_train_tensor : torch.Tensor
            The input X.
        y_train : np.ndarray
            The labels.
        epoch : int
            The current epoch.
        batch_num : int
            The number of batches to generate in one epoch.
        batch_size :  int
            The batch size.
        s_a_a : float
            The label for (a,a) pair.
        s_a_u : float
            The label for (a,u) pair.
        s_u_u : float
            The label for (u,u) pair.

        Returns
        -------
        tuple
            A tuple of dataloaders - one for X and one for y.
        """
        data_loader_X = []
        data_loader_y = []

        index_a = np.where(y_train == 1)[0]
        index_u = np.where(y_train == 0)[0]

        for _i in range(batch_num):  # i.e., drop_last = True
            index = []

            # pairs of (a,a); (a,u); (u,u)
            for j in range(6):
                # generate unique seed and set seed
                # seed = self.unique(epoch, i)
                # seed = self.unique(seed, j)
                # self.set_seed(seed)

                if j < 3:
                    index_sub = np.random.choice(index_a, batch_size // 4, replace=True)
                    index.append(list(index_sub))

                if j == 3:
                    index_sub = np.random.choice(index_u, batch_size // 4, replace=True)
                    index.append(list(index_sub))

                if j > 3:
                    index_sub = np.random.choice(index_u, batch_size // 2, replace=True)
                    index.append(list(index_sub))

            # index[0] + index[1] = (a,a), batch / 4
            # index[2] + index[2] = (a,u), batch / 4
            # index[4] + index[5] = (u,u), batch / 2
            index_left = index[0] + index[2] + index[4]
            index_right = index[1] + index[3] + index[5]

            X_train_tensor_left = X_train_tensor[index_left]
            X_train_tensor_right = X_train_tensor[index_right]

            # generate label
            y_train_new = np.append(np.repeat(s_a_a, batch_size // 4), np.repeat(s_a_u, batch_size // 4))
            y_train_new = np.append(y_train_new, np.repeat(s_u_u, batch_size // 2))
            y_train_new = torch.from_numpy(y_train_new).float()

            # shuffle
            index_shuffle = np.arange(len(y_train_new))
            index_shuffle = np.random.choice(index_shuffle, len(index_shuffle), replace=False)

            X_train_tensor_left = X_train_tensor_left[index_shuffle]
            X_train_tensor_right = X_train_tensor_right[index_shuffle]
            y_train_new = y_train_new[index_shuffle]

            # save
            data_loader_X.append([X_train_tensor_left, X_train_tensor_right])
            data_loader_y.append(y_train_new)

        return data_loader_X, data_loader_y

    # gradient norm
    def grad_norm(self, grad_tuple: Tuple[torch.Tensor]) -> torch.Tensor:
        """Return the norm of the gradient."""
        grad = torch.tensor([0.0])
        for i in range(len(grad_tuple)):
            grad += torch.norm(grad_tuple[i])

        return grad

    # visualize the gradient flow in network
    def plot_grad_flow(self, named_parameters: dict) -> None:
        """Plot the gradient flow in network."""
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)

    def torch_cdf_loss(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor, p: int = 1) -> torch.Tensor:
        """Calculate the First Wasserstein Distance."""
        # last-dimension is weight distribution
        # p is the norm of the distance, p=1 --> First Wasserstein Distance
        # to get a positive weight with our normalized distribution
        # we recommend combining this loss with other difference-based losses like L1

        # normalize distribution, add 1e-14 to divisor to avoid 0/0
        tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
        tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
        # make cdf with cumsum
        cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
        cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

        # choose different formulas for different norm situations
        if p == 1:
            cdf_distance = torch.sum(torch.abs((cdf_tensor_a - cdf_tensor_b)), dim=-1)
        elif p == 2:
            cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1))
        else:
            cdf_distance = torch.pow(
                torch.sum(torch.pow(torch.abs(cdf_tensor_a - cdf_tensor_b), p), dim=-1),
                1 / p,
            )

        cdf_loss = cdf_distance.mean()
        return cdf_loss

    def torch_wasserstein_loss(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        """Compute the first Wasserstein distance between two 1D distributions."""
        return self.torch_cdf_loss(tensor_a, tensor_b, p=1)

    def cal_loss(self, y: torch.Tensor, y_pred: torch.Tensor, mode: Literal["devnet"] = "devnet") -> torch.Tensor:
        """Calculate the loss like devnet in PyTorch."""
        if mode == "devnet":
            y_pred.squeeze_()

            ref = torch.randn(5000)  # sampling from the normal distribution
            dev = (y_pred - torch.mean(ref)) / torch.std(ref)
            #         print(f'mean:{torch.mean(ref)}, std:{torch.std(ref)}')
            inlier_loss = torch.abs(dev)
            outlier_loss = torch.max(5.0 - dev, torch.zeros_like(5.0 - dev))

            loss = torch.mean((1 - y) * inlier_loss + y * outlier_loss)
        else:
            raise NotImplementedError

        return loss

    def result_process(self, result_show: pd.DataFrame, name: str, std: bool = False) -> pd.DataFrame:
        """Process the result."""
        # average performance
        ave_metric = np.mean(result_show, axis=0).values
        std_metric = np.std(result_show, axis=0).values

        # statistical test
        wilcoxon_df = pd.DataFrame(data=None, index=result_show.columns, columns=result_show.columns)

        for i in range(wilcoxon_df.shape[0]):
            for j in range(wilcoxon_df.shape[1]):
                if i != j:
                    wilcoxon_df.iloc[i, j] = wilcoxon(
                        result_show.iloc[:, i] - result_show.iloc[:, j],
                        alternative="greater",
                    )[1]

        # average rank
        result_show.loc["Ave.rank"] = np.mean(result_show.rank(ascending=False, method="dense", axis=1), axis=0)

        # average metric
        if std:
            result_show.loc["Ave.metric"] = [
                str(format(round(a, 3), ".3f")) + "±" + str(format(round(s, 3), ".3f"))
                for a, s in zip(ave_metric, std_metric, strict=True)  # type: ignore
            ]
        else:
            result_show.loc["Ave.metric"] = [
                str(format(round(a, 3), ".3f"))
                for a, s in zip(ave_metric, std_metric, strict=True)  # type: ignore
            ]

        # the p-value of wilcoxon statistical test
        result_show.loc["p-value"] = wilcoxon_df.loc[name].values

        for _ in result_show.index:
            if _ in ["Ave.rank", "p-value"]:
                result_show.loc[_, :] = [format(round(_, 2), ".2f") for _ in result_show.loc[_, :].values]

        # result_show = result_show.astype('float')
        # result_show = result_show.round(2)

        return result_show
