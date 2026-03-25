"""Bi-cross-validation for dynamic functional connectivity models."""

import os
import pickle
import yaml
import shutil
import json

import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold

from osl_dynamics.config_api.pipeline import run_pipeline_from_file
from osl_dynamics.utils.array_ops import npz2list


class CVSplit:
    """A class to split rows and columns in (bi-)cross-validation.

    Parameters
    ----------
    split_row : dict, optional
        Configuration for row splitting.
    split_column : dict, optional
        Configuration for column splitting.
    strategy : str
        ``"combination"`` or ``"pairing"`` to combine row and column splits.
    kwargs
        Additional configurations (e.g., ``random_state``).
    """

    def __init__(self, split_row=None, split_column=None,
                 strategy="combination", **kwargs):
        self.split_row = split_row
        self.split_column = split_column
        self.strategy = strategy
        self.kwargs = kwargs

        # Validate input
        if strategy not in ["combination", "pairing"]:
            raise ValueError("Strategy must be 'combination' or 'pairing'.")
        if not (split_row or split_column):
            raise ValueError(
                "At least one of split_row or split_column must be provided."
            )

        # Initialize row and column splitters
        self.row_splitter = (
            self._init_splitter(split_row) if split_row else None
        )
        self.column_splitter = (
            self._init_splitter(split_column) if split_column else None
        )

    def _init_splitter(self, split_config):
        """Initialize a cross-validation splitter (ShuffleSplit or KFold).

        Parameters
        ----------
        split_config : dict
            Configuration for the splitter.

        Returns
        -------
        tuple
            Initialized splitter and the number of samples.
        """
        n_samples = split_config["n_samples"]
        method = split_config.get("method", "ShuffleSplit")
        method_kwargs = split_config.get("method_kwargs", {})

        if method == "ShuffleSplit":
            splitter = ShuffleSplit(
                n_splits=method_kwargs["n_splits"],
                train_size=method_kwargs["train_size"],
                **self.kwargs,
            )
        elif method == "KFold":
            splitter = KFold(n_splits=method_kwargs["n_splits"], **self.kwargs)
        else:
            raise ValueError(
                "Unsupported cross-validation method: "
                "use 'ShuffleSplit' or 'KFold'."
            )

        return splitter, n_samples

    def split(self):
        """Generate splits for rows and columns.

        Yields
        ------
        tuple
            Train and test indices for rows and columns.
        """
        row_splits = (
            list(self.row_splitter[0].split(range(self.row_splitter[1])))
            if self.row_splitter
            else [([], [])]
        )
        column_splits = (
            list(
                self.column_splitter[0].split(range(self.column_splitter[1]))
            )
            if self.column_splitter
            else [([], [])]
        )

        if self.strategy == "combination":
            for row_train, row_test in row_splits:
                for col_train, col_test in column_splits:
                    yield row_train, row_test, col_train, col_test
        elif self.strategy == "pairing":
            n_splits = min(len(row_splits), len(column_splits))
            for i in range(n_splits):
                yield (*row_splits[i], *column_splits[i])

    def get_n_splits(self):
        """Get the total number of splits based on the pairing strategy.

        Returns
        -------
        int
            Total number of cross-validation realizations.
        """
        row_splits = (
            len(list(self.row_splitter[0].split(range(self.row_splitter[1]))))
            if self.row_splitter
            else 1
        )
        column_splits = (
            len(
                list(
                    self.column_splitter[0].split(
                        range(self.column_splitter[1])
                    )
                )
            )
            if self.column_splitter
            else 1
        )

        if self.strategy == "combination":
            return row_splits * column_splits
        elif self.strategy == "pairing":
            return min(row_splits, column_splits)

    def save(self, save_dir):
        """Save the splits to disk.

        Parameters
        ----------
        save_dir : str
            Directory to save the splits.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, split in enumerate(self.split()):
            if self.row_splitter and self.column_splitter:
                row_train, row_test, col_train, col_test = split
                split_dict = {
                    "row_train": sorted(row_train.tolist()),
                    "row_test": sorted(row_test.tolist()),
                    "column_X": sorted(col_train.tolist()),
                    "column_Y": sorted(col_test.tolist()),
                }
            elif self.row_splitter:
                row_train, row_test, _, _ = split
                split_dict = {
                    "row_train": sorted(row_train.tolist()),
                    "row_test": sorted(row_test.tolist()),
                }
            elif self.column_splitter:
                _, _, col_train, col_test = split
                split_dict = {
                    "column_X": sorted(col_train.tolist()),
                    "column_Y": sorted(col_test.tolist()),
                }
            else:
                raise ValueError("No row or column splitter is defined.")

            file_path = os.path.join(save_dir, f"fold_indices_{i + 1}.json")
            with open(file_path, "w") as f:
                json.dump(split_dict, f, indent=4)


class CVBase:
    """Base class for bi-cross validation in dFC models.

    The idea comes from `k-means bi cross validation
    <https://www.tandfonline.com/doi/full/10.1080/10618600.2019.1647846>`_.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples.
    n_channels : int, optional
        The number of channels (the length of each sample).
    row_indices : str or list, optional
        The list of row indices. Read from file if it's a string.
    column_indices : str or list, optional
        The list of column indices. Read from file if it's a string.
    save_dir : str, optional
        Directory to save partition indices.
    partition_rows : int
        The number of row partitions.
    partition_columns : int
        The number of column partitions.
    """

    def __init__(
        self,
        n_samples=None,
        n_channels=None,
        row_indices=None,
        column_indices=None,
        save_dir=None,
        partition_rows=2,
        partition_columns=2,
    ):
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.save_dir = save_dir

        # Initialise the class using row_indices/column_indices
        if (row_indices is not None) and (column_indices is not None):
            if isinstance(row_indices, str):
                row_indices = npz2list(np.load(row_indices))
            if isinstance(column_indices, str):
                column_indices = npz2list(np.load(column_indices))
            self.row_indices = row_indices
            self.column_indices = column_indices
            self.partition_rows = len(self.row_indices)
            self.partition_columns = len(self.column_indices)
            # Update the number of samples and number of channels
            self.n_samples = sum(len(arr) for arr in self.row_indices)
            self.n_channels = sum(len(arr) for arr in self.column_indices)
        else:
            self.partition_rows = partition_rows
            self.partition_columns = partition_columns
            self.partition_indices()

    def partition_indices(self):
        """Generate partition indices.

        If ``save_dir`` is not None, save the indices to disk.
        """
        # Generate random row and column indices
        row_indices = np.arange(self.n_samples)
        column_indices = np.arange(self.n_channels)
        np.random.shuffle(row_indices)
        np.random.shuffle(column_indices)

        # Divide rows into partitions
        self.row_indices = np.array_split(row_indices, self.partition_rows)

        # Divide columns into partitions
        self.column_indices = np.array_split(
            column_indices, self.partition_columns
        )

        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            np.savez(
                os.path.join(self.save_dir, "row_indices.npz"),
                *self.row_indices,
            )
            np.savez(
                os.path.join(self.save_dir, "column_indices.npz"),
                *self.column_indices,
            )

    def fold_indices(self, r, s):
        """Given the partitions, return the indices of fold (r, s).

        Fold (r, s) treats the r-th row subset as "test", and the s-th
        column subset as response.

        Parameters
        ----------
        r : int
            The row index.
        s : int
            The column index.

        Returns
        -------
        row_train : list
            Row indices for X_train and Y_train.
        row_test : list
            Row indices for X_test and Y_test.
        column_X : list
            Column indices for X_train and X_test.
        column_Y : list
            Column indices for Y_train and Y_test.
        """
        row_train = []
        row_test = []
        column_X = []
        column_Y = []

        for i, row_index in enumerate(self.row_indices):
            if i == r:
                row_test.extend(row_index)
            else:
                row_train.extend(row_index)

        for j, column_index in enumerate(self.column_indices):
            if j == s:
                column_Y.extend(column_index)
            else:
                column_X.extend(column_index)

        row_train = sorted(list(map(int, row_train)))
        row_test = sorted(list(map(int, row_test)))
        column_X = sorted(list(map(int, column_X)))
        column_Y = sorted(list(map(int, column_Y)))

        return row_train, row_test, column_X, column_Y


class NCV:
    """Naive Cross-Validation.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:

        - ``save_dir``: Directory to save results.
        - ``indices``: Path to JSON file with row_train/row_test indices.
        - ``load_data``: Data loading configuration.
        - ``model``: Model configuration dict (e.g., ``{"hmm": {...}}``).
    """

    def __init__(self, config):
        if not os.path.exists(config["save_dir"]):
            os.makedirs(config["save_dir"])
        self.save_dir = config["save_dir"]
        self.indices = config["indices"]
        self.load_data = config["load_data"]
        self.model, self.model_kwargs = next(iter(config["model"].items()))

        with open(config["indices"], "r") as file:
            indices = json.load(file)
        self.row_train = indices["row_train"]
        self.row_test = indices["row_test"]

    def train(self):
        """Train the model on the training set."""
        prepare_config = {}
        prepare_config["load_data"] = self.load_data
        prepare_config[f"train_{self.model}"] = self.model_kwargs
        prepare_config["keep_list"] = self.row_train

        prepare_config["load_data"]["kwargs"]["store_dir"] = (
            f"{self.save_dir}/tmp_train/"
        )
        with open(f"{self.save_dir}/prepared_config.yaml", "w") as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(
            f"{self.save_dir}/prepared_config.yaml", self.save_dir
        )

        return f"{self.save_dir}/model/"

    def naive_validation(self):
        """Evaluate the model on the test set."""
        prepare_config = {}
        prepare_config["load_data"] = self.load_data
        prepare_config["load_data"]["kwargs"]["store_dir"] = (
            f"{self.save_dir}/tmp_validate/"
        )
        # For Sliding Window Correlation, use log likelihood instead
        if self.model == "swc":
            prepare_config["log_likelihood"] = {
                "static_FC": False,
                "infer_alpha": True,
            }
        else:
            prepare_config["free_energy"] = {}
        prepare_config["keep_list"] = self.row_test

        with open(
            f"{self.save_dir}/prepared_ncv_config.yaml", "w"
        ) as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(
            f"{self.save_dir}/prepared_ncv_config.yaml", self.save_dir
        )

    def validate(self):
        """Run the full naive cross-validation pipeline."""
        n = self.model_kwargs["config_kwargs"].get(
            "n_states", self.model_kwargs["config_kwargs"].get("n_modes")
        )
        if n > 1:
            self.train()
            self.naive_validation()


class BCV:
    """Bi-Cross-Validation for dynamic functional connectivity models.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:

        - ``save_dir``: Directory to save results.
        - ``indices``: Path to JSON file with row/column indices.
        - ``load_data``: Data loading configuration.
        - ``model``: Model configuration dict (e.g., ``{"hmm": {...}}``).
        - ``cv_variant``: Cross-validation variant (``"1"``, ``"2"``,
          ``"3"``, or ``"4"``).
        - ``mode``: Mode string (e.g., ``"bcv_1"``), used to determine
          whether to save temporal parameters.
    n_temp_save : int, optional
        Number of realisations for which to preserve temporal parameters.
    """

    def __init__(self, config, n_temp_save=5):
        if not os.path.exists(config["save_dir"]):
            os.makedirs(config["save_dir"])
        self.save_dir = config["save_dir"]
        self.indices = config["indices"]
        self.load_data = config["load_data"]
        self.model, self.model_kwargs = next(iter(config["model"].items()))

        with open(config["indices"], "r") as file:
            indices = json.load(file)
        self.row_train = indices["row_train"]
        self.row_test = indices["row_test"]
        self.column_X = indices.get("column_X", None)
        self.column_Y = indices.get("column_Y", None)

        self.cv_variant = str(config.get("cv_variant", "1"))

        # In bi-cross-validation, we preserve n_temp_save realisations
        # of the time courses
        _, bcv_index = config["mode"].rsplit("_", 1)
        if int(bcv_index) > n_temp_save:
            self.save_temp = False
        else:
            self.save_temp = True

    def full_train(self, row, column, save_dir=None):
        """Train the model on a subset of rows and columns.

        Parameters
        ----------
        row : list
            Row indices (sessions) to use for training.
        column : list
            Column indices (channels) to use for training.
        save_dir : str, optional
            Directory to save results.

        Returns
        -------
        spatial : dict
            Paths to means and covariances files.
        temporal : str
            Path to alpha pickle file.
        """
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, "full_train/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prepare_config = {}
        prepare_config["load_data"] = self.load_data
        prepare_config["load_data"]["kwargs"]["store_dir"] = (
            f"{save_dir}/tmp/"
        )

        if "select" not in prepare_config["load_data"]["prepare"].keys():
            prepare_config["load_data"]["prepare"]["select"] = {}
        prepare_config["load_data"]["prepare"]["select"]["channels"] = column

        prepare_config[f"train_{self.model}"] = self.model_kwargs
        prepare_config[f"train_{self.model}"]["config_kwargs"][
            "n_channels"
        ] = len(column)
        prepare_config["keep_list"] = row

        with open(f"{save_dir}/prepared_config.yaml", "w") as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f"{save_dir}/prepared_config.yaml", save_dir)
        params_dir = f"{save_dir}/inf_params/"
        return (
            {
                "means": f"{params_dir}/means.npy",
                "covs": f"{params_dir}/covs.npy",
            },
            f"{params_dir}/alp.pkl",
        )

    def infer_spatial(self, row, column, temporal, save_dir=None,
                      method="sample"):
        """Infer spatial patterns from temporal information.

        Parameters
        ----------
        row : list
            Row indices (sessions) to use.
        column : list
            Column indices (channels) to use.
        temporal : str
            Path to temporal parameters (alpha pickle).
        save_dir : str, optional
            Directory to save results.
        method : str, optional
            Method for dual estimation.

        Returns
        -------
        dict or str
            Paths to means and covariances files, or ``"0"`` if
            n_states/n_modes is 1.
        """
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, "infer_spatial/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Do nothing if n_states = 1 or n_modes = 1
        if (
            self.model_kwargs["config_kwargs"].get(
                "n_states",
                self.model_kwargs["config_kwargs"].get("n_modes"),
            )
            == 1
        ):
            return "0"

        if not os.path.exists(f"{save_dir}inf_params/"):
            os.makedirs(f"{save_dir}inf_params/")

        shutil.move(temporal, f"{save_dir}inf_params/")

        prepare_config = {}
        prepare_config["load_data"] = self.load_data
        prepare_config["load_data"]["kwargs"]["store_dir"] = (
            f"{save_dir}/tmp/"
        )

        if "select" not in prepare_config["load_data"]["prepare"].keys():
            prepare_config["load_data"]["prepare"]["select"] = {}
        prepare_config["load_data"]["prepare"]["select"]["channels"] = column

        prepare_config[f"build_{self.model}"] = {}
        prepare_config[f"build_{self.model}"]["config_kwargs"] = (
            self.model_kwargs["config_kwargs"]
        )
        prepare_config[f"build_{self.model}"]["config_kwargs"][
            "n_channels"
        ] = len(column)

        method = self.model_kwargs.get("infer_spatial", "sample")
        prepare_config["dual_estimation"] = {
            "concatenate": True,
            "method": method,
        }

        prepare_config["keep_list"] = row

        with open(f"{save_dir}/prepared_config.yaml", "w") as file:
            yaml.safe_dump(
                prepare_config, file, default_flow_style=False,
                sort_keys=False,
            )

        run_pipeline_from_file(
            f"{save_dir}/prepared_config.yaml", save_dir
        )

        # Delete the time courses if self.save_temp is False
        if not self.save_temp:
            if os.path.exists(f"{save_dir}/inf_params/alp.pkl"):
                os.remove(f"{save_dir}/inf_params/alp.pkl")

        return {
            "means": f"{save_dir}/dual_estimates/means.npy",
            "covs": f"{save_dir}/dual_estimates/covs.npy",
        }

    def infer_temporal(self, row, column, spatial, save_dir=None):
        """Infer temporal patterns from spatial information.

        Parameters
        ----------
        row : list
            Row indices (sessions) to use.
        column : list
            Column indices (channels) to use.
        spatial : dict
            Paths to means and covariances files.
        save_dir : str, optional
            Directory to save results.

        Returns
        -------
        str
            Path to alpha pickle file, or ``"0"`` if n_states/n_modes is 1.
        """
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, "infer_temporal/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Do nothing if n_states = 1 or n_modes = 1
        if (
            self.model_kwargs["config_kwargs"].get(
                "n_states",
                self.model_kwargs["config_kwargs"].get("n_modes"),
            )
            == 1
        ):
            return "0"

        prepare_config = {}
        prepare_config["load_data"] = self.load_data
        prepare_config["load_data"]["kwargs"]["store_dir"] = (
            f"{save_dir}/tmp/"
        )

        if "select" not in prepare_config["load_data"]["prepare"].keys():
            prepare_config["load_data"]["prepare"]["select"] = {}
        prepare_config["load_data"]["prepare"]["select"]["channels"] = column

        prepare_config[f"train_{self.model}"] = self.model_kwargs
        prepare_config[f"train_{self.model}"]["config_kwargs"][
            "n_channels"
        ] = len(column)
        prepare_config["keep_list"] = row

        # Fix the means and covariances
        prepare_config[f"train_{self.model}"]["config_kwargs"][
            "learn_means"
        ] = False
        prepare_config[f"train_{self.model}"]["config_kwargs"][
            "learn_covariances"
        ] = False
        prepare_config[f"train_{self.model}"]["config_kwargs"][
            "initial_means"
        ] = spatial["means"]
        prepare_config[f"train_{self.model}"]["config_kwargs"][
            "initial_covariances"
        ] = spatial["covs"]

        with open(f"{save_dir}/prepared_config.yaml", "w") as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(
            f"{save_dir}/prepared_config.yaml", save_dir
        )
        params_dir = f"{save_dir}/inf_params/"
        return f"{params_dir}/alp.pkl"

    def calculate_error(self, row, column, temporal, spatial, save_dir=None):
        """Calculate the error metric on held-out data.

        Parameters
        ----------
        row : list
            Row indices (sessions) to use.
        column : list
            Column indices (channels) to use.
        temporal : str
            Path to temporal parameters (alpha pickle).
        spatial : dict
            Paths to means and covariances files.
        save_dir : str, optional
            Directory to save results.

        Returns
        -------
        str
            Path to metrics JSON file.
        """
        if save_dir is None:
            save_dir = os.path.join(self.save_dir, "calculate_error/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(f"{save_dir}inf_params/"):
            os.makedirs(f"{save_dir}inf_params/")

        if os.path.exists(temporal):
            shutil.move(temporal, f"{save_dir}inf_params/")

        prepare_config = {}
        prepare_config["load_data"] = self.load_data
        prepare_config["load_data"]["kwargs"]["store_dir"] = (
            f"{save_dir}/tmp/"
        )

        if "select" not in prepare_config["load_data"]["prepare"].keys():
            prepare_config["load_data"]["prepare"]["select"] = {}
        prepare_config["load_data"]["prepare"]["select"]["channels"] = column

        n_states_or_modes = self.model_kwargs["config_kwargs"].get(
            "n_states",
            self.model_kwargs["config_kwargs"].get("n_modes", None),
        )
        if n_states_or_modes > 1:
            prepare_config[f"build_{self.model}"] = {}
            prepare_config[f"build_{self.model}"]["config_kwargs"] = (
                self.model_kwargs["config_kwargs"]
            )
            prepare_config[f"build_{self.model}"]["config_kwargs"][
                "n_channels"
            ] = len(column)
            prepare_config[f"build_{self.model}"]["config_kwargs"][
                "initial_means"
            ] = spatial["means"]
            prepare_config[f"build_{self.model}"]["config_kwargs"][
                "initial_covariances"
            ] = spatial["covs"]
            prepare_config["log_likelihood"] = {"static_FC": False}
        else:
            prepare_config["log_likelihood"] = {
                "static_FC": True,
                "spatial": spatial,
            }

        prepare_config["keep_list"] = row

        with open(f"{save_dir}/prepared_config.yaml", "w") as file:
            yaml.safe_dump(
                prepare_config, file, default_flow_style=False,
                sort_keys=False,
            )
        run_pipeline_from_file(
            f"{save_dir}/prepared_config.yaml", save_dir
        )

        # Delete the time courses if self.save_temp is False
        if not self.save_temp:
            if os.path.exists(f"{save_dir}/inf_params/alp.pkl"):
                os.remove(f"{save_dir}/inf_params/alp.pkl")

        return f"{save_dir}/metrics.json"

    def split_column(self, column_1, column_2, spatial, save_dir=None):
        """Split spatial parameters by column indices.

        Parameters
        ----------
        column_1 : list
            First set of column indices.
        column_2 : list
            Second set of column indices.
        spatial : dict
            Paths to means and covariances files.
        save_dir : list of str, optional
            Two directories to save split results.

        Returns
        -------
        tuple of dict
            Paths to split means and covariances files.
        """
        if save_dir is None:
            save_dir = [
                os.path.join(self.save_dir, "column_1_spatial/"),
                os.path.join(self.save_dir, "column_2_spatial/"),
            ]
        for d in save_dir:
            if not os.path.exists(d):
                os.makedirs(d)

        means = np.load(spatial["means"])
        covs = np.load(spatial["covs"])

        means_1 = means[:, column_1]
        means_2 = means[:, column_2]
        covs_1 = (covs[:, :, column_1])[:, column_1, :]
        covs_2 = (covs[:, :, column_2])[:, column_2, :]

        np.save(f"{save_dir[0]}means.npy", means_1)
        np.save(f"{save_dir[0]}covs.npy", covs_1)
        np.save(f"{save_dir[1]}means.npy", means_2)
        np.save(f"{save_dir[1]}covs.npy", covs_2)

        return (
            {
                "means": f"{save_dir[0]}means.npy",
                "covs": f"{save_dir[0]}covs.npy",
            },
            {
                "means": f"{save_dir[1]}means.npy",
                "covs": f"{save_dir[1]}covs.npy",
            },
        )

    def split_row(self, row_1, row_2, temporal, save_dir=None):
        """Split temporal parameters by row indices.

        Parameters
        ----------
        row_1 : list
            First set of row indices.
        row_2 : list
            Second set of row indices.
        temporal : str
            Path to temporal parameters (alpha pickle).
        save_dir : list of str, optional
            Two directories to save split results.

        Returns
        -------
        tuple of str
            Paths to split alpha pickle files.
        """
        if save_dir is None:
            save_dir = [
                os.path.join(self.save_dir, "row_1_temporal/"),
                os.path.join(self.save_dir, "row_2_temporal/"),
            ]
        for d in save_dir:
            if not os.path.exists(d):
                os.makedirs(d)

        with open(temporal, "rb") as file:
            alpha = pickle.load(file)

        alpha_1 = [alpha[i] for i in row_1]
        alpha_2 = [alpha[i] for i in row_2]

        with open(f"{save_dir[0]}/alp.pkl", "wb") as file:
            pickle.dump(alpha_1, file)
        with open(f"{save_dir[1]}/alp.pkl", "wb") as file:
            pickle.dump(alpha_2, file)

        # Remove the original file to save disk space
        os.remove(temporal)

        return f"{save_dir[0]}/alp.pkl", f"{save_dir[1]}/alp.pkl"

    def validate(self):
        """Run the bi-cross-validation pipeline.

        Executes the selected CV variant (1-4).
        """
        if self.cv_variant == "1":
            spatial_Y_train, temporal_Y_train = self.full_train(
                self.row_train, self.column_Y,
                save_dir=os.path.join(self.save_dir, "Y_train/"),
            )
            spatial_X_train = self.infer_spatial(
                self.row_train, self.column_X, temporal_Y_train,
                save_dir=os.path.join(self.save_dir, "X_train/"),
            )
            temporal_X_test = self.infer_temporal(
                self.row_test, self.column_X, spatial_X_train,
                save_dir=os.path.join(self.save_dir, "X_test/"),
            )
            metric = self.calculate_error(
                self.row_test, self.column_Y, temporal_X_test,
                spatial_Y_train,
                save_dir=os.path.join(self.save_dir, "Y_test/"),
            )
        elif self.cv_variant == "2":
            spatial_X_train, temporal_X_train = self.full_train(
                self.row_train, self.column_X,
                save_dir=os.path.join(self.save_dir, "X_train/"),
            )
            spatial_Y_train = self.infer_spatial(
                self.row_train, self.column_Y, temporal_X_train,
                save_dir=os.path.join(self.save_dir, "Y_train/"),
            )
            temporal_X_test = self.infer_temporal(
                self.row_test, self.column_X, spatial_X_train,
                save_dir=os.path.join(self.save_dir, "X_test/"),
            )
            metric = self.calculate_error(
                self.row_test, self.column_Y, temporal_X_test,
                spatial_Y_train,
                save_dir=os.path.join(self.save_dir, "Y_test/"),
            )
        elif self.cv_variant == "3":
            spatial_XY_train, _ = self.full_train(
                self.row_train,
                sorted(self.column_X + self.column_Y),
                save_dir=os.path.join(self.save_dir, "XY_train/"),
            )
            spatial_X_train, spatial_Y_train = self.split_column(
                self.column_X, self.column_Y, spatial_XY_train,
                save_dir=[
                    os.path.join(self.save_dir, "X_train/"),
                    os.path.join(self.save_dir, "Y_train/"),
                ],
            )
            temporal_X_test = self.infer_temporal(
                self.row_test, self.column_X, spatial_X_train,
                save_dir=os.path.join(self.save_dir, "X_test/"),
            )
            metric = self.calculate_error(
                self.row_test, self.column_Y, temporal_X_test,
                spatial_Y_train,
                save_dir=os.path.join(self.save_dir, "Y_test/"),
            )
        elif self.cv_variant == "4":
            _, temporal_X_traintest = self.full_train(
                sorted(self.row_train + self.row_test), self.column_X,
                save_dir=os.path.join(self.save_dir, "X_traintest/"),
            )
            temporal_X_train, temporal_X_test = self.split_row(
                self.row_train, self.row_test, temporal_X_traintest,
                save_dir=[
                    os.path.join(self.save_dir, "X_train/"),
                    os.path.join(self.save_dir, "X_test/"),
                ],
            )
            spatial_Y_train = self.infer_spatial(
                self.row_train, self.column_Y, temporal_X_train,
                save_dir=os.path.join(self.save_dir, "Y_train/"),
            )
            metric = self.calculate_error(
                self.row_test, self.column_Y, temporal_X_test,
                spatial_Y_train,
                save_dir=os.path.join(self.save_dir, "Y_test/"),
            )
        else:
            raise ValueError("Currently the cv_variant unavailable!")
