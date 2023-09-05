"""PYOD wrapper."""
from typing import Dict, Optional

import numpy as np
from deepod.models.icl import ICL
from myutils import Utils
from pyod.models.lof import LOF


class PYOD:
    """PYOD wrapper."""

    def __init__(self, seed: int, model_name: str, tune: bool = False) -> None:
        """Initialize the wrapper.

        Parameters
        ----------
        seed : int
            Random seed for reproducible results.
        model_name : str
            The name of the model.
        tune : bool
            If necessary, tune the hyper-parameter based on the validation set
            constructed by the labeled anomalies
        """
        self.seed = seed
        self.utils = Utils()

        self.model_name = model_name
        self.model_dict = {
            "ICL": ICL,
        }

        self.tune = tune

    def grid_hp(self, model_name: str) -> Optional[list]:
        """Define the hyper-parameter search grid for different unsupervised m0del."""
        param_grid_dict: Dict[str, Optional[list]] = {
            "IForest": [10, 50, 100, 500],  # n_estimators, default=100
            "OCSVM": ["linear", "poly", "rbf", "sigmoid"],  # kernel, default='rbf',
            "ABOD": [3, 5, 7, 9],  # n_neighbors, default=5
            "CBLOF": [4, 6, 8, 10],  # n_clusters, default=8
            "COF": [5, 10, 20, 50],  # n_neighbors, default=20
            "AOM": None,
            "COPOD": None,
            "ECOD": None,
            "FeatureBagging": [3, 5, 10, 20],  # n_estimators, default=10
            "HBOS": [3, 5, 10, 20],  # n_bins, default=10
            "KNN": [3, 5, 10, 20],  # n_neighbors, default=5
            "LMDD": ["aad", "var", "iqr"],  # dis_measure, default='aad'
            "LODA": [3, 5, 10, 20],  # n_bins, default=10
            "LOF": [5, 10, 20, 50],  # n_neighbors, default=20
            "LOCI": [0.1, 0.25, 0.5, 0.75],  # alpha, default=0.5
            "LSCP": [3, 5, 10, 20],  # n_bins, default=10
            "MAD": None,
            "MCD": None,
            "PCA": [0.25, 0.5, 0.75, None],  # n_components
            "ROD": None,
            "SOD": [5, 10, 20, 50],  # n_neighbors, default=20
            "SOS": [2.5, 4.5, 7.5, 10.0],  # perplexity, default=4.5
            "VAE": None,
            "AutoEncoder": None,
            "SOGAAL": [10, 20, 50, 100],  # stop_epochs, default=20
            "MOGAAL": [10, 20, 50, 100],  # stop_epochs, default=20
            "XGBOD": None,
            "DeepSVDD": [20, 50, 100, 200],  # epochs, default=100
        }

        return param_grid_dict[model_name]

    def grid_search(
        self, X_train: np.ndarray, y_train: np.ndarray, ratio: float
    ) -> Optional[dict]:
        """Run a grid search for unsupervised models.

        Return the best hyper-parameters. The ratio refers to the ground truth
        anomaly ratio of input dataset.
        """
        # set seed
        self.utils.set_seed(self.seed)
        # get the hyper-parameter grid
        param_grid = self.grid_hp(self.model_name)

        if param_grid is not None:
            # index of normal ana abnormal samples
            idx_a = np.where(y_train == 1)[0]
            idx_n = np.where(y_train == 0)[0]
            idx_n = np.random.choice(
                idx_n, int((len(idx_a) * (1 - ratio)) / ratio), replace=True
            )

            idx = np.append(idx_n, idx_a)  # combine
            np.random.shuffle(idx)  # shuffle

            # valiation set (and the same anomaly ratio as in the original dataset)
            X_val = X_train[idx]
            y_val = y_train[idx]

            # fitting
            metric_list = []
            for _param in param_grid:
                try:
                    if self.model_name == "ICL":
                        self.clf = ICL(device="cpu")
                        self.model = self.clf(X_train)

                    else:
                        raise NotImplementedError

                except Exception:
                    metric_list.append(0.0)
                    continue

                try:
                    # model performance on the validation set
                    score_val = self.model.decision_function(X_val)
                    metric = self.utils.metric(
                        y_true=y_val, y_score=score_val, pos_label=1
                    )
                    metric_list.append(metric["aucpr"])

                except Exception:
                    metric_list.append(0.0)
                    continue

            best_param = param_grid[np.argmax(metric_list)]

        else:
            metric_list = None
            best_param = None

        print(
            f"The candidate hyper-parameter of {self.model_name}: {param_grid},",
            f" corresponding metric: {metric_list}",
            f" the best candidate: {best_param}",
        )

        return best_param

    def fit(  # noqa: C901
        self, X_train: np.ndarray, y_train: np.ndarray, ratio: Optional[float] = None
    ) -> "PYOD":
        """Fit the model."""
        if self.model_name in ["AutoEncoder", "VAE"]:
            # only use the normal samples to fit the model
            idx_n = np.where(y_train == 0)[0]
            X_train = X_train[idx_n]
            y_train = y_train[idx_n]

        # selecting the best hyper-parameters of unsupervised model for fair
        # comparison (if labeled anomalies is available)
        if sum(y_train) > 0 and self.tune:
            assert ratio is not None
            best_param = self.grid_search(X_train, y_train, ratio)
        else:
            best_param = None

        print(f"best param: {best_param}")

        # set seed
        self.utils.set_seed(self.seed)

        # fit best on the best param
        if best_param is not None:
            if self.model_name in ["IForest", "FeatureBagging"]:
                self.model = self.model_dict[self.model_name](
                    n_estimators=best_param
                ).fit(X_train)

            elif self.model_name == "OCSVM":
                self.model = self.model_dict[self.model_name](kernel=best_param).fit(
                    X_train
                )
            elif self.model_name in ["ABOD", "COF", "KNN", "LOF", "SOD"]:
                self.model = self.model_dict[self.model_name](
                    n_neighbors=best_param
                ).fit(X_train)
            elif self.model_name == "CBLOF":
                self.model = self.model_dict[self.model_name](
                    n_clusters=best_param
                ).fit(X_train)
            elif self.model_name in ["HBOS", "LODA"]:
                self.model = self.model_dict[self.model_name](n_bins=best_param).fit(
                    X_train
                )
            elif self.model_name == "LMDD":
                self.model = self.model_dict[self.model_name](
                    dis_measure=best_param
                ).fit(X_train)
            elif self.model_name == "LOCI":
                self.model = self.model_dict[self.model_name](alpha=best_param).fit(
                    X_train
                )
            elif self.model_name == "LSCP":
                self.model = self.model_dict[self.model_name](
                    detector_list=[LOF(), LOF()], n_bins=best_param
                ).fit(X_train)
            elif self.model_name == "PCA":
                self.model = self.model_dict[self.model_name](
                    n_components=best_param
                ).fit(X_train)
            elif self.model_name == "SOS":
                self.model = self.model_dict[self.model_name](
                    perplexity=best_param
                ).fit(X_train)
            elif self.model_name in ["SOGAAL", "MOGAAL"]:
                self.model = self.model_dict[self.model_name](
                    stop_epochs=best_param
                ).fit(X_train)
            elif self.model_name == "DeepSVDD":
                self.model = self.model_dict[self.model_name](epochs=best_param).fit(
                    X_train
                )
            elif self.model_name == "ICL":
                self.clf = ICL(device="cpu")
                self.model = self.clf(X_train)
            else:
                raise NotImplementedError
        else:
            # unsupervised method would ignore the y labels
            self.model = self.model_dict[self.model_name]().fit(X_train, y_train)

        return self

    # from pyod: for consistency, outliers are assigned with larger anomaly scores
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Predict the decision score of the input samples."""
        score = self.model.decision_function(X)
        return score
