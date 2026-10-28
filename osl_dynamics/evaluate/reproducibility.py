"""Reproducibility of models across runs and halves of the data."""

import logging
import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from osl_dynamics.evaluate.cross_validation import (
    _save_json,
    _save_spatial,
    _save_temporal,
    _subset,
)
from osl_dynamics.evaluate.model_ops import ModelOps, fit_static, get_model_ops
from osl_dynamics.evaluate.splits import load_fold

_logger = logging.getLogger("osl-dynamics")


def repeat(
    model: Union[str, ModelOps],
    data,
    output_dir: Optional[str] = None,
    config_kwargs: Optional[dict] = None,
    init_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
    save_temporal: bool = True,
) -> Dict[str, float]:
    """Train a model on all of the data.

    Calling this several times shows how much the trained model varies
    between runs. This function creates:

    - :code:`<output_dir>/model`: the trained model.
    - :code:`<output_dir>/inf_params`: the spatial parameters and (if
      :code:`save_temporal=True`) the time courses.
    - :code:`<output_dir>/metrics/metrics.json`: the free energy (and the
      model evidence for the HMM, or the log-likelihood for SWC).

    A model with one state/mode is a static model: only its mean and
    covariance are saved.

    Parameters
    ----------
    model : str or ModelOps
        Model name (:code:`"hmm"`, :code:`"dynemo"`, :code:`"mdynemo"` or
        :code:`"swc"`) or model operations from
        :py:func:`osl_dynamics.evaluate.model_ops.get_model_ops`.
    data : osl_dynamics.data.Data
        Prepared data.
    output_dir : str, optional
        Directory to save the results to. If :code:`None`, nothing is saved.
    config_kwargs : dict, optional
        Keyword arguments for the model's :code:`Config`. Must be passed if
        :code:`model` is a str.
    init_kwargs : dict, optional
        Keyword arguments for the model's initialization method.
    fit_kwargs : dict, optional
        Keyword arguments for the model's :code:`fit` method.
    save_temporal : bool, optional
        Should we save the time courses?

    Returns
    -------
    metrics : dict
        Metrics of the trained model. Empty for a static model.
    """
    ops = _get_ops(model, config_kwargs, init_kwargs, fit_kwargs)
    if ops.is_static:
        spatial = fit_static(data, ops.learn_means)
        _save_spatial(spatial, output_dir, "inf_params")
        return {}

    trained_model = ops.train(data)
    metrics = ops.training_metrics(trained_model, data)
    if output_dir is not None:
        trained_model.save(f"{output_dir}/model")
        _save_spatial(ops.get_spatial(trained_model), output_dir, "inf_params")
        if save_temporal:
            _save_temporal(ops, ops.get_temporal(trained_model, data), output_dir)
        _save_json(metrics, f"{output_dir}/metrics/metrics.json")

    _logger.info(f"Training metrics: {metrics}")
    return metrics


def split_half(
    model: Union[str, ModelOps],
    data,
    fold: Union[str, Dict],
    output_dir: Optional[str] = None,
    config_kwargs: Optional[dict] = None,
    init_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
    save_temporal: bool = True,
) -> float:
    """Split-half reproducibility of the spatial parameters.

    The model is trained separately on the two halves of the sessions
    (:code:`"row_train"` and :code:`"row_test"` in the fold). The states of
    the two models are paired to minimize the Riemannian distance between
    their covariances (correlations for M-DyNeMo) and the mean distance
    between paired states is returned. Lower is more reproducible.

    This function creates:

    - :code:`<output_dir>/partition_<i>/model` and
      :code:`<output_dir>/partition_<i>/inf_params` for each half.
    - :code:`<output_dir>/reproducibility.json`: the mean distance and the
      pairing of the states.
    - :code:`<output_dir>/riemannian_distances.npy`: the distance between
      every pair of states.

    Parameters
    ----------
    model : str or ModelOps
        Model name or model operations (see :py:func:`repeat`).
    data : osl_dynamics.data.Data
        Prepared data.
    fold : str or dict
        Path to a fold JSON file or a dict with keys :code:`"row_train"`
        and :code:`"row_test"` (the two halves).
    output_dir : str, optional
        Directory to save the results to. If :code:`None`, nothing is saved.
    config_kwargs : dict, optional
        Keyword arguments for the model's :code:`Config`.
    init_kwargs : dict, optional
        Keyword arguments for the model's initialization method.
    fit_kwargs : dict, optional
        Keyword arguments for the model's :code:`fit` method.
    save_temporal : bool, optional
        Should we save the time courses?

    Returns
    -------
    distance : float
        Mean Riemannian distance between paired states.
    """
    ops = _get_ops(model, config_kwargs, init_kwargs, fit_kwargs)
    fold = load_fold(fold, keys=["row_train", "row_test"], n_sessions=data.n_sessions)

    matrices = []
    for i, rows in enumerate([fold["row_train"], fold["row_test"]], start=1):
        _logger.info(f"Split-half: training on half {i}")
        partition_dir = None if output_dir is None else f"{output_dir}/partition_{i}"
        with _subset(data, rows, None, partition_dir) as half:
            if ops.is_static:
                spatial = fit_static(half, ops.learn_means)
            else:
                trained_model = ops.train(half)
                spatial = ops.get_spatial(trained_model)
                if partition_dir is not None:
                    trained_model.save(f"{partition_dir}/model")
                    if save_temporal:
                        _save_temporal(
                            ops, ops.get_temporal(trained_model, half), partition_dir
                        )
                del trained_model
        _save_spatial(spatial, partition_dir, "inf_params")
        matrices.append(spatial["covs"] if "covs" in spatial else spatial["corrs"])

    distances = pairwise_riemannian_distances(matrices[0], matrices[1])
    row, col = linear_sum_assignment(distances)
    distance = float(np.mean(distances[row, col]))

    _logger.info(f"Split-half mean Riemannian distance: {distance}")
    if output_dir is not None:
        np.save(f"{output_dir}/riemannian_distances.npy", distances)
        _save_json(
            {
                "riemannian_distance": distance,
                "pairing": {"row": row.tolist(), "col": col.tolist()},
            },
            f"{output_dir}/reproducibility.json",
        )
    return distance


def pairwise_riemannian_distances(
    matrices_1: np.ndarray,
    matrices_2: np.ndarray,
    eps_values: Sequence[float] = (0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5),
    threshold: float = 1e-3,
) -> np.ndarray:
    """Riemannian distance between every pair of matrices from two sets.

    If the distance cannot be calculated, the matrices are regularized by
    adding a small value to the diagonal, trying each value in
    :code:`eps_values` in turn.

    Parameters
    ----------
    matrices_1 : np.ndarray
        First set of matrices. Shape must be (n_1, n_channels, n_channels).
    matrices_2 : np.ndarray
        Second set of matrices. Shape must be (n_2, n_channels, n_channels).
    eps_values : sequence of float, optional
        Values to add to the diagonal of the matrices.
    threshold : float, optional
        Passed to :py:func:`osl_dynamics.inference.metrics.riemannian_distance`.

    Returns
    -------
    distances : np.ndarray
        Riemannian distances. Shape is (n_1, n_2).
    """
    from osl_dynamics.inference.metrics import riemannian_distance

    matrices_1 = np.asarray(matrices_1, dtype=np.float64)
    matrices_2 = np.asarray(matrices_2, dtype=np.float64)
    if matrices_1.shape[1:] != matrices_2.shape[1:]:
        raise ValueError("matrices_1 and matrices_2 must have the same shape.")
    identity = np.eye(matrices_1.shape[-1])

    for eps in eps_values:
        try:
            return np.array(
                [
                    [
                        riemannian_distance(
                            m1 + eps * identity, m2 + eps * identity, threshold
                        )
                        for m2 in matrices_2
                    ]
                    for m1 in matrices_1
                ]
            )
        except np.linalg.LinAlgError:
            warnings.warn(f"Riemannian distance failed for eps={eps}.")

    raise ValueError("Riemannian distance failed for all eps values.")


def _get_ops(
    model: Union[str, ModelOps],
    config_kwargs: Optional[dict],
    init_kwargs: Optional[dict],
    fit_kwargs: Optional[dict],
) -> ModelOps:
    if isinstance(model, ModelOps):
        return model
    if config_kwargs is None:
        raise ValueError("config_kwargs must be passed if model is a str.")
    return get_model_ops(model, config_kwargs, init_kwargs, fit_kwargs)
