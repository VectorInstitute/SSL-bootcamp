"""Data Generator for creating the data."""

import os
import random
from math import ceil
from pickle import UnpicklingError
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE
from myutils import Utils  # type: ignore
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataGenerator:
    """Data Generator Class.

    Only supports for generating the binary classification datasets.

    Attributes
    ----------
    seed : int, default=42
        Seed for reproducible results.
    dataset : str, optional, default=None
        The dataset name.
    test_size : float, default=0.3
        The proportion of testing set.
    generate_duplicates : bool, default=True
        Whether to generate duplicated samples when sample size is too small.
    n_samples_threshold : int, default=1000
        The threshold for generating duplicates. If `generate_duplicates` is False,
        then datasets with sample size smaller than n_samples_threshold will be dropped.
    """

    def __init__(
        self,
        seed: int = 42,
        dataset: Optional[str] = None,
        test_size: float = 0.3,
        generate_duplicates: bool = True,
        n_samples_threshold: int = 1000,
    ):
        """Initialize the Data Generator."""
        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        # dataset list
        self.dataset_list_classical: Optional[list] = None
        self.dataset_list_cv: Optional[list] = None
        self.dataset_list_nlp: Optional[list] = None
        if os.path.exists("datasets/Classical"):
            self.dataset_list_classical = [
                os.path.splitext(_)[0] for _ in os.listdir("datasets/Classical") if os.path.splitext(_)[1] == ".npz"
            ]  # classical AD datasets

        if os.path.exists("datasets/CV_by_ResNet18"):
            self.dataset_list_cv = [
                os.path.splitext(_)[0]
                for _ in os.listdir("datasets/CV_by_ResNet18")
                if os.path.splitext(_)[1] == ".npz"
            ]  # CV datasets

        if os.path.exists("datasets/NLP_by_BERT"):
            self.dataset_list_nlp = [
                os.path.splitext(_)[0] for _ in os.listdir("datasets/NLP_by_BERT") if os.path.splitext(_)[1] == ".npz"
            ]  # NLP datasets

        # myutils function
        self.utils = Utils()

    def generate_realistic_synthetic(  # noqa: C901
        self,
        X: np.ndarray,
        y: np.ndarray,
        realistic_synthetic_mode: Literal["local", "cluster", "dependency", "global"],
        alpha: int,
        percentage: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the realistic synthetic outliers.

        Currently, four types of realistic synthetic outliers can be generated:
        1. local outliers: where normal data follows the GMM distribution, and
        anomalies follow the GMM distribution with modified covariance
        2. global outliers: where normal data follows the GMM distribution, and
        anomalies follow the uniform distribution
        3. dependency outliers: where normal data follows the vine coupula
        distribution, and anomalies follow the independent distribution captured
        by GaussianKDE
        4. cluster outliers: where normal data follows the GMM distribution, and
        anomalies follow the GMM distribution with modified mean.

        Parameters
        ----------
        X : np.ndarray
            Input X.
        y : np.ndarray
            Input y.
        realistic_synthetic_mode : Literal["local", "cluster", "dependency", "global"]
            The type of generated outliers.
        alpha : int
            The scaling parameter for controlling the generated local and cluster
            anomalies.
        percentage : float
            Controls the generated global anomalies.

        Returns
        -------
        X : np.ndarray
            The generated X.
        y : np.ndarray
            The generated y.
        """
        if realistic_synthetic_mode not in ["local", "cluster", "dependency", "global"]:
            raise NotImplementedError

        # the number of normal data and anomalies
        pts_n = len(np.where(y == 0)[0])
        pts_a = len(np.where(y == 1)[0])

        # only use the normal data to fit the model
        X = X[y == 0]
        y = y[y == 0]

        # generate the synthetic normal data
        if realistic_synthetic_mode in ["local", "cluster", "global"]:
            # select the best n_components based on the BIC value
            metric_list = []
            n_components_list = list(np.arange(1, 10))

            for n_components in n_components_list:
                gm = GaussianMixture(n_components=n_components, random_state=self.seed).fit(X)
                metric_list.append(gm.bic(X))

            best_n_components = n_components_list[np.argmin(metric_list)]

            # refit based on the best n_components
            gm = GaussianMixture(n_components=best_n_components, random_state=self.seed).fit(X)

            # generate the synthetic normal data
            X_synthetic_normal = gm.sample(pts_n)[0]

            # generate the synthetic abnormal data
            if realistic_synthetic_mode == "local":
                # generate the synthetic anomalies (local outliers)
                gm.covariances_ = alpha * gm.covariances_
                X_synthetic_anomalies = gm.sample(pts_a)[0]
            if realistic_synthetic_mode == "cluster":
                # generate the clustering synthetic anomalies
                gm.means_ = alpha * gm.means_
                X_synthetic_anomalies = gm.sample(pts_a)[0]
            if realistic_synthetic_mode == "global":
                # generate the synthetic anomalies (global outliers)
                X_synthetic_anomalies = []

                for i in range(X_synthetic_normal.shape[1]):
                    low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
                    high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)

                    X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))

                X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

        # we found that copula function may have error in some datasets
        elif realistic_synthetic_mode == "dependency":
            # sampling the feature since copulas method may spend too long to fit
            if X.shape[1] > 50:
                idx = np.random.choice(np.arange(X.shape[1]), 50, replace=False)
                X = X[:, idx]

            copula = VineCopula("center")  # default is the C-vine copula
            copula.fit(pd.DataFrame(X))

            # sample to generate synthetic normal data
            X_synthetic_normal = copula.sample(pts_n).values

            # generate the synthetic anomalies (dependency outliers)
            X_synthetic_anomalies = np.zeros((pts_a, X.shape[1]))

            # using the GaussianKDE for generating independent feature
            for i in range(X.shape[1]):
                kde = GaussianKDE()
                kde.fit(X[:, i])
                X_synthetic_anomalies[:, i] = kde.sample(pts_a)
        else:
            raise NotImplementedError

        X = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)  # type: ignore # noqa: E501
        y = np.append(
            np.repeat(0, X_synthetic_normal.shape[0]),
            np.repeat(1, X_synthetic_anomalies.shape[0]),  # type: ignore
        )

        return X, y

    """
    Here we also consider the robustness of baseline models, where three types
    of noise can be added:
    1. Duplicated anomalies, which should be added to training and testing set,
    respectively.
    2. Irrelevant features, which should be added to both training and testing set.
    3. Annotation errors (Label flips), which should be only added to the training set.
    """

    def add_duplicated_anomalies(
        self, X: np.ndarray, y: np.ndarray, duplicate_times: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add duplicated anomalies to the dataset."""
        if duplicate_times > 1:
            # index of normal and anomaly data
            idx_n = np.where(y == 0)[0]
            idx_a = np.where(y == 1)[0]

            # generate duplicated anomalies
            idx_a = np.random.choice(idx_a, int(len(idx_a) * duplicate_times))

            idx = np.append(idx_n, idx_a)
            random.shuffle(idx)
            X = X[idx]
            y = y[idx]

        return X, y

    def add_irrelevant_features(
        self, X: np.ndarray, y: np.ndarray, noise_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add irrelevant features to the dataset."""
        # adding uniform noise
        if noise_ratio != 0.0:
            noise_dim = int(noise_ratio / (1 - noise_ratio) * X.shape[1])
            if noise_dim > 0:
                X_noise = []
                for _i in range(noise_dim):
                    idx = np.random.choice(np.arange(X.shape[1]), 1)
                    X_min = np.min(X[:, idx])
                    X_max = np.max(X[:, idx])

                    X_noise.append(np.random.uniform(X_min, X_max, size=(X.shape[0], 1)))

                # concat the irrelevant noise feature
                X_noise = np.hstack(X_noise)
                X = np.concatenate((X, X_noise), axis=1)
                # shuffle the dimension
                idx = np.random.choice(np.arange(X.shape[1]), X.shape[1], replace=False)
                X = X[:, idx]

        return X, y

    def add_label_contamination(
        self, X: np.ndarray, y: np.ndarray, noise_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add label contamination to the dataset."""
        if noise_ratio != 0.0:
            # here we consider the label flips situation: a label is randomly
            # flipped to another class with probability p (i.e., noise ratio)
            idx_flips = np.random.choice(np.arange(len(y)), int(len(y) * noise_ratio), replace=False)
            y[idx_flips] = 1 - y[idx_flips]  # change 0 to 1 and 1 to 0

        return X, y

    def generator(  # noqa: C901
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        minmax: bool = True,
        la: Optional[Union[float, int]] = None,
        at_least_one_labeled: bool = False,
        realistic_synthetic_mode: Optional[Literal["local", "cluster", "dependency", "global"]] = None,
        alpha: int = 5,
        percentage: float = 0.1,
        noise_type: Optional[
            Literal[
                "duplicated_anomalies",
                "irrelevant_features",
                "label_contamination",
                "anomaly_contamination",
            ]
        ] = None,
        duplicate_times: int = 2,
        contam_ratio: float = 1.00,
        noise_ratio: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """Generate the data.

        Parameters
        ----------
        X : np.ndarray, optional, default=None
            Input X.
        y : np.ndarray, optional, default=None
            Input y.
        minmax : bool, default=True
            Whether to use the minmax scaling.
        la : Optional[Union[float, int]], default=None
            Labeled anomalies. Can be either the ratio of labeled anomalies or
            the number of labeled anomalies
        at_least_one_labeled : bool, default=False
            Whether to guarantee at least one labeled anomalies in the training set.
        realistic_synthetic_mode : Optional[Literal["local", "cluster", "dependency", "global"]], default=None
            The type of generated outliers.
        alpha : int, default=5
            The scaling parameter for controlling the generated local and cluster
            anomalies.
        percentage : float, default=0.1
            Controls the generated global anomalies.
        noise_type : Optional[Literal["duplicated_anomalies", "irrelevant_features", "label_contamination", "anomaly_contamination"]], default=None
            The type of noise.
        duplicate_times : int, default=2
            The number of duplicated anomalies.
        contam_ratio : float, default=1.00
            The ratio of anomaly contamination.
        noise_ratio : float, default=0.05
            The ratio of noise.
        """  # noqa: E501, W505
        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # load dataset
        if self.dataset is None:
            assert X is not None and y is not None, "For customized dataset, you should provide the X and y!"
        else:
            if self.dataset_list_classical is not None and self.dataset in self.dataset_list_classical:
                data = np.load(
                    os.path.join("datasets", "Classical", self.dataset + ".npz"),
                    allow_pickle=True,
                )
            elif self.dataset_list_cv is not None and self.dataset in self.dataset_list_cv:
                data = np.load(
                    os.path.join("datasets", "CV_by_ResNet18", self.dataset + ".npz"),
                    allow_pickle=True,
                )
            elif self.dataset_list_nlp is not None and self.dataset in self.dataset_list_nlp:
                data = np.load(
                    os.path.join("datasets", "NLP_by_BERT", self.dataset + ".npz"),
                    allow_pickle=True,
                )
            else:
                raise NotImplementedError

            X = data["X"]
            y = data["y"]

        # number of labeled anomalies in the original data
        if isinstance(la, float):
            if at_least_one_labeled:
                ceil(sum(y) * (1 - self.test_size) * la)  # type: ignore
            else:
                int(sum(y) * (1 - self.test_size) * la)  # type: ignore
        elif isinstance(la, int):
            pass
        else:
            raise TypeError("The type of `la` should be either float or int!")

        # if the dataset is too small, generating duplicate samples up to
        # n_samples_threshold
        if len(y) < self.n_samples_threshold and self.generate_duplicates:  # type: ignore # noqa: E501
            print(f"generating duplicate samples for dataset {self.dataset}...")
            self.utils.set_seed(self.seed)
            idx_duplicate = np.random.choice(
                np.arange(len(y)),
                self.n_samples_threshold,
                replace=True,  # type: ignore # noqa: E501
            )
            X = X[idx_duplicate]
            y = y[idx_duplicate]

        # if the dataset is too large, subsampling for considering the computational
        # cost
        if len(y) > 10000:  # type: ignore
            print(f"subsampling for dataset {self.dataset}...")
            self.utils.set_seed(self.seed)
            idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)  # type: ignore # noqa: E501
            X = X[idx_sample]
            y = y[idx_sample]

        # whether to generate realistic synthetic outliers
        if realistic_synthetic_mode is not None:
            # we save the generated dependency anomalies, since the Vine Copula
            # could spend too long for generation
            if realistic_synthetic_mode == "dependency":
                if not os.path.exists("datasets/synthetic"):
                    os.makedirs("datasets/synthetic")

                filepath = (
                    "dependency_anomalies_"  # type: ignore
                    + self.dataset
                    + "_"
                    + str(self.seed)
                    + ".npz"
                )
                try:
                    data_dependency = np.load(
                        os.path.join("datasets", "synthetic", filepath),
                        allow_pickle=True,
                    )
                    X = data_dependency["X"]
                    y = data_dependency["y"]

                except (OSError, ValueError, EOFError, UnpicklingError):
                    print("Generating dependency anomalies...")
                    X, y = self.generate_realistic_synthetic(
                        X,
                        y,
                        realistic_synthetic_mode=realistic_synthetic_mode,
                        alpha=alpha,
                        percentage=percentage,
                    )
                    np.savez_compressed(os.path.join("datasets", "synthetic", filepath), X=X, y=y)
            else:
                X, y = self.generate_realistic_synthetic(
                    X,
                    y,
                    realistic_synthetic_mode=realistic_synthetic_mode,
                    alpha=alpha,
                    percentage=percentage,
                )

        # whether to add different types of noise for testing the robustness of
        # benchmark models
        if noise_type == "irrelevant_features":
            X, y = self.add_irrelevant_features(X, y, noise_ratio=noise_ratio)
        print(f"current noise type: {noise_type}")

        # show the statistic
        self.utils.data_description(X=X, y=y)

        # splitting the current data to the training set and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=True, stratify=y)

        # we respectively generate the duplicated anomalies for the training and
        # testing set
        if noise_type == "duplicated_anomalies":
            X_train, y_train = self.add_duplicated_anomalies(X_train, y_train, duplicate_times=duplicate_times)
            X_test, y_test = self.add_duplicated_anomalies(X_test, y_test, duplicate_times=duplicate_times)

        # notice that label contamination can only be added in the training set
        if noise_type == "label_contamination":
            X_train, y_train = self.add_label_contamination(X_train, y_train, noise_ratio=noise_ratio)

        # minmax scaling
        if minmax:
            scaler = MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if isinstance(la, float):
            if at_least_one_labeled:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
            else:
                idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)
        elif isinstance(la, int):
            if la > len(idx_anomaly):
                raise AssertionError(
                    f"The number of labeled anomalies are greater than the total anomalies: {len(idx_anomaly)} !"
                )
            idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
        else:
            raise TypeError("The type of `la` should be either float or int!")

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        # whether to remove the anomaly contamination in the unlabeled data
        # if noise_type == "anomaly_contamination":
        #     idx_unlabeled_anomaly = self.remove_anomaly_contamination(
        #         idx_unlabeled_anomaly, contam_ratio
        #     )

        # unlabeled data = normal data + unlabeled anomalies
        # (which is considered as contamination)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
