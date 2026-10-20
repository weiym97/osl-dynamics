"""Splitting sessions (rows) and channels (columns) for cross-validation."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

_logger = logging.getLogger("osl-dynamics")

_FOLD_KEYS = ["row_train", "row_test", "column_X", "column_Y"]


class CVSplit:
    """Split rows (sessions) and/or columns (channels) for cross-validation.

    Parameters
    ----------
    split_row : dict, optional
        How to split the rows. Must contain :code:`n_samples` (the number of
        sessions). Can contain :code:`method` (:code:`"ShuffleSplit"` or
        :code:`"KFold"`, defaults to :code:`"ShuffleSplit"`) and
        :code:`method_kwargs` (keyword arguments for the scikit-learn class,
        e.g. :code:`n_splits` and :code:`train_size`).
    split_column : dict, optional
        How to split the columns. Same format as :code:`split_row`, with
        :code:`n_samples` the number of channels.
    strategy : str, optional
        How to combine the row and column splits. :code:`"combination"` uses
        every pair of row and column splits. :code:`"pairing"` pairs the
        :code:`i`-th row split with the :code:`i`-th column split.
    **kwargs
        Keyword arguments passed to every scikit-learn splitter,
        e.g. :code:`random_state`.

    Examples
    --------
    Bi-cross-validation with 100 paired splits of 80% training sessions and
    50% channels::

        splitter = CVSplit(
            split_row={
                "n_samples": 1003,
                "method": "ShuffleSplit",
                "method_kwargs": {"n_splits": 100, "train_size": 0.8},
            },
            split_column={
                "n_samples": 50,
                "method": "ShuffleSplit",
                "method_kwargs": {"n_splits": 100, "train_size": 0.5},
            },
            strategy="pairing",
            random_state=0,
        )
        splitter.save("bcv_partition")
    """

    def __init__(
        self,
        split_row: Optional[Dict] = None,
        split_column: Optional[Dict] = None,
        strategy: str = "combination",
        **kwargs,
    ) -> None:
        if strategy not in ["combination", "pairing"]:
            raise ValueError("strategy must be 'combination' or 'pairing'.")
        if not (split_row or split_column):
            raise ValueError(
                "At least one of split_row or split_column must be passed."
            )

        self.split_row = split_row
        self.split_column = split_column
        self.strategy = strategy
        self.kwargs = kwargs

        self.row_splitter = self._init_splitter(split_row) if split_row else None
        self.column_splitter = (
            self._init_splitter(split_column) if split_column else None
        )

    def _init_splitter(self, split_config: Dict) -> Tuple[object, int]:
        """Create a scikit-learn splitter.

        Parameters
        ----------
        split_config : dict
            See :code:`split_row` in the class docstring.

        Returns
        -------
        splitter : sklearn.model_selection.ShuffleSplit or KFold
            Splitter.
        n_samples : int
            Number of items to split.
        """
        n_samples = split_config["n_samples"]
        method = split_config.get("method", "ShuffleSplit")
        method_kwargs = dict(split_config.get("method_kwargs", {}))

        if method == "ShuffleSplit":
            splitter = ShuffleSplit(**method_kwargs, **self.kwargs)
        elif method == "KFold":
            splitter = KFold(**method_kwargs, **self.kwargs)
        else:
            raise ValueError("method must be 'ShuffleSplit' or 'KFold'.")

        return splitter, n_samples

    def split(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Generate the splits.

        Yields
        ------
        row_train : np.ndarray
            Training rows. Empty if :code:`split_row` was not passed.
        row_test : np.ndarray
            Test rows. Empty if :code:`split_row` was not passed.
        column_X : np.ndarray
            First set of columns. Empty if :code:`split_column` was not passed.
        column_Y : np.ndarray
            Second set of columns. Empty if :code:`split_column` was not
            passed.
        """
        row_splits = self._get_splits(self.row_splitter)
        column_splits = self._get_splits(self.column_splitter)

        if self.strategy == "combination":
            for row_train, row_test in row_splits:
                for column_X, column_Y in column_splits:
                    yield row_train, row_test, column_X, column_Y
        else:
            n_splits = min(len(row_splits), len(column_splits))
            for i in range(n_splits):
                yield (*row_splits[i], *column_splits[i])

    def get_n_splits(self) -> int:
        """Total number of splits.

        Returns
        -------
        n_splits : int
            Number of splits.
        """
        n_row_splits = len(self._get_splits(self.row_splitter))
        n_column_splits = len(self._get_splits(self.column_splitter))
        if self.strategy == "combination":
            return n_row_splits * n_column_splits
        return min(n_row_splits, n_column_splits)

    def get_folds(self) -> List[Dict[str, List[int]]]:
        """Get the splits as a list of folds.

        Returns
        -------
        folds : list of dict
            Each fold contains sorted lists of indices with keys
            :code:`"row_train"`, :code:`"row_test"` (if :code:`split_row`
            was passed), :code:`"column_X"` and :code:`"column_Y"` (if
            :code:`split_column` was passed).
        """
        folds = []
        for row_train, row_test, column_X, column_Y in self.split():
            fold = {}
            if self.row_splitter:
                fold["row_train"] = sorted(np.asarray(row_train).tolist())
                fold["row_test"] = sorted(np.asarray(row_test).tolist())
            if self.column_splitter:
                fold["column_X"] = sorted(np.asarray(column_X).tolist())
                fold["column_Y"] = sorted(np.asarray(column_Y).tolist())
            folds.append(fold)
        return folds

    def save(self, save_dir: str) -> None:
        """Save each fold to :code:`<save_dir>/fold_indices_<i>.json`.

        Fold numbers start at 1. Each file is written to a temporary file
        first and then renamed, so jobs reading the files in parallel never
        see a partially written file.

        Parameters
        ----------
        save_dir : str
            Directory to save the folds to.
        """
        os.makedirs(save_dir, exist_ok=True)
        for i, fold in enumerate(self.get_folds()):
            save_fold(fold, f"{save_dir}/fold_indices_{i + 1}.json")
        _logger.info(f"Saved {self.get_n_splits()} folds to {save_dir}")

    @staticmethod
    def _get_splits(splitter: Optional[Tuple[object, int]]) -> list:
        if splitter is None:
            return [([], [])]
        splitter, n_samples = splitter
        return list(splitter.split(np.arange(n_samples)))


def save_fold(fold: Dict[str, List[int]], filename: str) -> None:
    """Save a fold as a JSON file.

    The file is written to a temporary file first and then renamed.

    Parameters
    ----------
    fold : dict
        Fold indices.
    filename : str
        Path to the JSON file.
    """
    tmp_filename = f"{filename}.{os.getpid()}.tmp"
    with open(tmp_filename, "w") as file:
        json.dump(fold, file, indent=4)
    os.replace(tmp_filename, filename)


def load_fold(
    fold: Union[str, Path, Dict],
    keys: Optional[List[str]] = None,
    n_sessions: Optional[int] = None,
    n_channels: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Load and validate a fold.

    Parameters
    ----------
    fold : str or dict
        Path to a JSON file saved by :code:`CVSplit.save`, or a dict.
    keys : list of str, optional
        Keys that must be in the fold. Defaults to :code:`"row_train"`,
        :code:`"row_test"`, :code:`"column_X"` and :code:`"column_Y"`.
    n_sessions : int, optional
        Number of sessions. Used to check the row indices.
    n_channels : int, optional
        Number of channels. Used to check the column indices.

    Returns
    -------
    fold : dict
        Fold with sorted lists of indices.
    """
    if isinstance(fold, (str, Path)):
        with open(fold, "r") as file:
            fold = json.load(file)

    keys = _FOLD_KEYS if keys is None else keys
    missing = [key for key in keys if key not in fold]
    if missing:
        raise ValueError(f"fold is missing {missing}.")

    fold = {key: sorted(int(i) for i in fold[key]) for key in fold}

    for key, size in [("row", n_sessions), ("column", n_channels)]:
        indices = [fold[k] for k in fold if k.startswith(key)]
        for k in [k for k in fold if k.startswith(key)]:
            if len(fold[k]) == 0:
                raise ValueError(f"{k} is empty.")
            if len(set(fold[k])) != len(fold[k]):
                raise ValueError(f"{k} contains repeated indices.")
            if size is not None and (fold[k][0] < 0 or fold[k][-1] >= size):
                raise ValueError(f"{k} contains indices outside [0, {size}).")
        if len(indices) == 2 and set(indices[0]) & set(indices[1]):
            raise ValueError(f"The {key} indices of the fold overlap.")

    return fold
