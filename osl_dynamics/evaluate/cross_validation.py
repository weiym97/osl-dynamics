"""Bi-cross-validation and naive cross-validation."""

import glob
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from osl_dynamics.evaluate.model_ops import (
    fit_static,
    get_model_ops,
    static_log_likelihood,
    subset_data,
)
from osl_dynamics.evaluate.splits import CVSplit, load_fold
from osl_dynamics.utils.misc import save

_logger = logging.getLogger("osl-dynamics")


class BCV:
    """Bi-cross-validation.

    Bi-cross-validation holds out sessions (rows) and channels (columns).
    For a fold with training sessions, test sessions and two sets of
    channels, X and Y, the steps are:

    1. **Y_train**: train the model on the training sessions and channels Y.
       This gives the spatial parameters of Y (e.g. state covariances).
    2. **X_train**: on the training sessions and channels X, estimate the
       spatial parameters of X with the time courses from step 1 fixed.
    3. **X_test**: on the test sessions and channels X, estimate the time
       courses with the spatial parameters from step 2 fixed.
    4. **Y_test**: calculate the log-likelihood of the test sessions and
       channels Y using the time courses from step 3 and the spatial
       parameters from step 1.

    The time courses used in step 4 are inferred from channels X only, so
    the test data in channels Y are never used to fit any parameter.

    Models with one state/mode are a static model: a multivariate normal is
    fitted to the training sessions and channels Y and evaluated on the test
    sessions and channels Y.

    Parameters
    ----------
    model : str
        Model to cross-validate. Options are :code:`"hmm"`,
        :code:`"dynemo"`, :code:`"mdynemo"` and :code:`"swc"`.
    config_kwargs : dict
        Keyword arguments for the model's :code:`Config`, e.g.
        :code:`n_states` (or :code:`n_modes`), :code:`learn_means` and
        :code:`learn_covariances`. :code:`n_channels` is set from the data.
    init_kwargs : dict, optional
        Keyword arguments for the model's initialization method.
    fit_kwargs : dict, optional
        Keyword arguments for the model's :code:`fit` method.
    spatial_method : str, optional
        How to estimate the spatial parameters in step 2. See
        :py:mod:`osl_dynamics.evaluate.model_ops`. Defaults to
        :code:`"sample"` for the HMM and :code:`"mle"` for DyNeMo.
    save_temporal : bool, optional
        Should we save the time courses? They can be large.

    Examples
    --------
    ::

        from osl_dynamics.data import Data
        from osl_dynamics.evaluate import BCV

        data = Data("training_data")
        data.standardize()

        bcv = BCV(
            "hmm",
            config_kwargs={
                "n_states": 6,
                "learn_means": False,
                "learn_covariances": True,
            },
        )
        log_likelihood = bcv.validate(
            data,
            fold="bcv_partition/fold_indices_1.json",
            output_dir="results/hmm_state_6/bcv_1",
        )
    """

    def __init__(
        self,
        model: str,
        config_kwargs: dict,
        init_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        spatial_method: Optional[str] = None,
        save_temporal: bool = True,
    ) -> None:
        self.ops = get_model_ops(
            model, config_kwargs, init_kwargs, fit_kwargs, spatial_method
        )
        self.save_temporal = save_temporal

    def validate(
        self,
        data,
        fold: Union[str, Dict],
        output_dir: Optional[str] = None,
        save_temporal: Optional[bool] = None,
    ) -> float:
        """Bi-cross-validate the model on one fold.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Prepared data.
        fold : str or dict
            Path to a fold JSON file (see :py:class:`CVSplit`) or a dict with
            keys :code:`"row_train"`, :code:`"row_test"`, :code:`"column_X"`
            and :code:`"column_Y"`.
        output_dir : str, optional
            Directory to save the results to. If :code:`None`, nothing is
            saved.
        save_temporal : bool, optional
            Overrides :code:`self.save_temporal` for this fold.

        Returns
        -------
        log_likelihood : float
            Bi-cross-validated log-likelihood averaged over time points.
        """
        fold = load_fold(fold, n_sessions=data.n_sessions, n_channels=data.n_channels)
        row_train, row_test = fold["row_train"], fold["row_test"]
        column_X, column_Y = fold["column_X"], fold["column_Y"]

        save_temporal = self.save_temporal if save_temporal is None else save_temporal
        dirs = _step_dirs(output_dir)

        if self.ops.n_states == 1:
            _logger.info("Static model: Y_train")
            with _subset(data, row_train, column_Y, output_dir) as Y_train:
                spatial_Y = fit_static(Y_train, self.ops.learn_means)
            _save_spatial(spatial_Y, dirs["Y_train"], "inf_params")

            _logger.info("Static model: Y_test")
            with _subset(data, row_test, column_Y, output_dir) as Y_test:
                log_likelihood = static_log_likelihood(Y_test, spatial_Y)

        else:
            _logger.info("BCV step 1/4 (Y_train): training the model")
            with _subset(data, row_train, column_Y, output_dir) as Y_train:
                spatial_Y, temporal_train = self.ops.fit(Y_train)
            _save_spatial(spatial_Y, dirs["Y_train"], "inf_params")
            if save_temporal:
                _save_temporal(self.ops, temporal_train, dirs["X_train"])

            _logger.info("BCV step 2/4 (X_train): inferring spatial parameters")
            with _subset(data, row_train, column_X, output_dir) as X_train:
                spatial_X = self.ops.infer_spatial(X_train, temporal_train)
            del temporal_train
            _save_spatial(spatial_X, dirs["X_train"], "dual_estimates")

            _logger.info("BCV step 3/4 (X_test): inferring time courses")
            with _subset(data, row_test, column_X, output_dir) as X_test:
                temporal_test = self.ops.infer_temporal(X_test, spatial_X)
            if save_temporal:
                _save_temporal(self.ops, temporal_test, dirs["Y_test"])

            _logger.info("BCV step 4/4 (Y_test): calculating the log-likelihood")
            with _subset(data, row_test, column_Y, output_dir) as Y_test:
                log_likelihood = self.ops.log_likelihood(
                    Y_test, temporal_test, spatial_Y
                )

        _logger.info(f"Bi-cross-validated log-likelihood: {log_likelihood}")
        if output_dir is not None:
            _save_json(
                {"log_likelihood": log_likelihood}, f"{dirs['Y_test']}/metrics.json"
            )
        return log_likelihood

    def run(
        self,
        data,
        folds: Union[CVSplit, str, List[Dict]],
        output_dir: str,
        n_temp_save: Optional[int] = 5,
        overwrite: bool = False,
    ) -> np.ndarray:
        """Bi-cross-validate the model on every fold, one after another.

        Fold :code:`i` is saved to :code:`<output_dir>/bcv_<i>`. Folds that
        have already finished are skipped (unless :code:`overwrite=True`),
        so an interrupted run can be resumed.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Prepared data.
        folds : CVSplit, str or list of dict
            Folds. Can be a :py:class:`CVSplit`, a directory of fold JSON
            files saved by :py:meth:`CVSplit.save` or a list of folds.
        output_dir : str
            Directory to save the results to.
        n_temp_save : int, optional
            The time courses are saved for the first :code:`n_temp_save`
            folds. If :code:`None`, they are saved for every fold.
        overwrite : bool, optional
            Should we re-run folds that have already finished?

        Returns
        -------
        log_likelihoods : np.ndarray
            Bi-cross-validated log-likelihood for each fold.
        """
        log_likelihoods = []
        for i, fold in enumerate(get_folds(folds), start=1):
            fold_dir = f"{output_dir}/bcv_{i}"
            metrics_file = f"{fold_dir}/Y_test/metrics.json"
            if os.path.exists(metrics_file) and not overwrite:
                _logger.info(f"Fold {i} has already finished: {metrics_file}")
                with open(metrics_file, "r") as file:
                    log_likelihoods.append(json.load(file)["log_likelihood"])
                continue

            _logger.info(f"Bi-cross-validation fold {i}")
            save_temporal = self.save_temporal and (
                n_temp_save is None or i <= n_temp_save
            )
            log_likelihoods.append(
                self.validate(data, fold, fold_dir, save_temporal=save_temporal)
            )
        return np.array(log_likelihoods)


class NCV:
    """Naive cross-validation.

    The model is trained on the training sessions (with all channels) and
    the variational free energy is calculated on the test sessions. For SWC,
    the log-likelihood of the test sessions (with the states assigned to the
    test windows) is calculated instead.

    Parameters
    ----------
    model : str
        Model to cross-validate. Options are :code:`"hmm"`,
        :code:`"dynemo"`, :code:`"mdynemo"` and :code:`"swc"`.
    config_kwargs : dict
        Keyword arguments for the model's :code:`Config`.
    init_kwargs : dict, optional
        Keyword arguments for the model's initialization method.
    fit_kwargs : dict, optional
        Keyword arguments for the model's :code:`fit` method.
    save_model : bool, optional
        Should we save the trained model to :code:`<output_dir>/model`?
    """

    def __init__(
        self,
        model: str,
        config_kwargs: dict,
        init_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        save_model: bool = False,
    ) -> None:
        self.ops = get_model_ops(model, config_kwargs, init_kwargs, fit_kwargs)
        self.save_model = save_model

    def validate(
        self,
        data,
        fold: Union[str, Dict],
        output_dir: Optional[str] = None,
    ) -> Optional[float]:
        """Naive cross-validate the model on one fold.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Prepared data.
        fold : str or dict
            Path to a fold JSON file or a dict with keys :code:`"row_train"`
            and :code:`"row_test"`.
        output_dir : str, optional
            Directory to save the results to. If :code:`None`, nothing is
            saved.

        Returns
        -------
        metric : float or None
            Free energy (or log-likelihood for SWC) of the test sessions.
            :code:`None` for a model with one state/mode, which is skipped.
        """
        if self.ops.n_states == 1:
            _logger.info("Naive cross-validation is skipped for one state/mode.")
            return None

        fold = load_fold(
            fold, keys=["row_train", "row_test"], n_sessions=data.n_sessions
        )

        _logger.info("NCV: training the model")
        with _subset(data, fold["row_train"], None, output_dir) as train_data:
            model = self.ops.train(train_data)
        if output_dir is not None and self.save_model:
            model.save(f"{output_dir}/model")

        _logger.info("NCV: evaluating the test sessions")
        with _subset(data, fold["row_test"], None, output_dir) as test_data:
            metric = self.ops.ncv_score(model, test_data)
        del model

        _logger.info(f"Naive cross-validated {self.ops.ncv_metric}: {metric}")
        if output_dir is not None:
            if self.ops.ncv_metric == "free_energy":
                _save_json([metric], f"{output_dir}/ncv_free_energy.json")
            else:
                _save_json({"log_likelihood": metric}, f"{output_dir}/metrics.json")
        return metric


def get_folds(folds: Union[CVSplit, str, List[Dict]]) -> List[Union[str, Dict]]:
    """Get a list of folds.

    Parameters
    ----------
    folds : CVSplit, str or list
        A :py:class:`CVSplit`, a directory containing
        :code:`fold_indices_<i>.json` files or a list of folds.

    Returns
    -------
    folds : list
        Folds (dicts or paths to JSON files) in order.
    """
    if isinstance(folds, CVSplit):
        return folds.get_folds()
    if isinstance(folds, str):
        files = glob.glob(f"{folds}/fold_indices_*.json")
        if len(files) == 0:
            raise FileNotFoundError(f"No fold_indices_*.json files in {folds}.")
        return sorted(files, key=lambda f: int(re.findall(r"(\d+)\.json$", f)[0]))
    return list(folds)


@contextmanager
def _subset(
    data, rows: List[int], columns: Optional[List[int]], output_dir: Optional[str]
) -> Iterator:
    """Context manager for a temporary subset of the data.

    The subset's files are stored in a temporary directory (inside
    :code:`<output_dir>/tmp` if :code:`output_dir` is passed), which is
    deleted afterwards.
    """
    parent_dir = None
    if output_dir is not None:
        parent_dir = f"{output_dir}/tmp"
        os.makedirs(parent_dir, exist_ok=True)
    store_dir = tempfile.mkdtemp(prefix="subset_", dir=parent_dir)
    subset = subset_data(data, rows, columns, store_dir=store_dir)
    try:
        yield subset
    finally:
        subset.delete_dir()
        shutil.rmtree(store_dir, ignore_errors=True)
        if parent_dir is not None and not os.listdir(parent_dir):
            os.rmdir(parent_dir)


def _step_dirs(output_dir: Optional[str]) -> Dict[str, Optional[str]]:
    steps = ["Y_train", "X_train", "X_test", "Y_test"]
    if output_dir is None:
        return {step: None for step in steps}
    return {step: f"{output_dir}/{step}" for step in steps}


def _save_spatial(spatial: Dict[str, np.ndarray], step_dir: Optional[str], name: str):
    if step_dir is None:
        return
    os.makedirs(f"{step_dir}/{name}", exist_ok=True)
    for key, value in spatial.items():
        save(f"{step_dir}/{name}/{key}.npy", value)


def _save_temporal(ops, temporal, step_dir: Optional[str]):
    if step_dir is None:
        return
    os.makedirs(f"{step_dir}/inf_params", exist_ok=True)
    for filename, value in ops.temporal_files(temporal).items():
        save(f"{step_dir}/inf_params/{filename}", value)


def _save_json(obj, filename: str) -> None:
    """Save a JSON file via a temporary file so it is never half-written."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tmp_filename = f"{filename}.{os.getpid()}.tmp"
    with open(tmp_filename, "w") as file:
        json.dump(obj, file)
    os.replace(tmp_filename, filename)
