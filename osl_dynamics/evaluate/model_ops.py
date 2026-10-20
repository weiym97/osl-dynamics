"""Model-specific operations used by the evaluation tools.

Bi-cross-validation needs four operations for each model:

1. Train the model (:code:`fit`), which gives the spatial parameters
   (observation model) and the temporal parameters (state/mode time courses).
2. Estimate the spatial parameters given fixed time courses
   (:code:`infer_spatial`).
3. Estimate the time courses given fixed spatial parameters
   (:code:`infer_temporal`).
4. Calculate the log-likelihood given both (:code:`log_likelihood`).

Each model family implements these in a :py:class:`ModelOps` subclass.
Spatial parameters are a dict of arrays (e.g. :code:`{"means": ...,
"covs": ...}`) and temporal parameters are a list with one array per session.

Models with one state/mode are a static model (a single multivariate
normal). They are handled by :py:func:`fit_static` and
:py:func:`static_log_likelihood` for every model.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular

from osl_dynamics.utils.misc import override_dict_defaults

_logger = logging.getLogger("osl-dynamics")


def time_series(data) -> List[np.ndarray]:
    """Get the prepared time series of the sessions in :code:`data.keep`.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.

    Returns
    -------
    ts : list of np.ndarray
        Time series for each session in :code:`data.keep`.
    """
    ts = data.time_series(prepared=True, concatenate=False)
    if isinstance(ts, np.ndarray):
        ts = [ts]
    return [ts[i] for i in data.keep]


def subset_data(
    data,
    rows: Optional[List[int]] = None,
    columns: Optional[List[int]] = None,
    store_dir: Optional[str] = None,
):
    """Create a new Data object containing a subset of sessions and channels.

    The new object only contains the requested sessions and channels, so
    functions that use every session in a Data object (rather than
    :code:`data.keep`) cannot see other sessions.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Prepared data.
    rows : list of int, optional
        Sessions to keep. Defaults to all sessions.
    columns : list of int, optional
        Channels to keep. Defaults to all channels.
    store_dir : str, optional
        Directory for the new Data object's files. If
        :code:`data.load_memmaps=True`, each session is written to this
        directory one at a time so the subset is never held in memory.

    Returns
    -------
    subset : osl_dynamics.data.Data
        Data object. Call :code:`subset.delete_dir()` when it is no longer
        needed.
    """
    from osl_dynamics.data import Data

    arrays = data.arrays
    rows = range(len(arrays)) if rows is None else rows
    columns = slice(None) if columns is None else list(columns)

    kwargs = {
        "sampling_frequency": data.sampling_frequency,
        "load_memmaps": data.load_memmaps,
        "buffer_size": data.buffer_size,
        "use_tfrecord": data.use_tfrecord,
        "n_jobs": data.n_jobs,
    }
    if store_dir is not None:
        kwargs["store_dir"] = store_dir

    if data.load_memmaps:
        if store_dir is None:
            raise ValueError("store_dir must be passed if data.load_memmaps=True.")
        os.makedirs(store_dir, exist_ok=True)
        inputs = []
        for i in rows:
            filename = f"{store_dir}/subset_{i}.npy"
            np.save(filename, np.asarray(arrays[i][:, columns], dtype=np.float32))
            inputs.append(filename)
    else:
        inputs = [np.asarray(arrays[i][:, columns], dtype=np.float32) for i in rows]

    return Data(inputs, **kwargs)


def gaussian_log_likelihood(
    x: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """Log-likelihood of each sample under a multivariate normal.

    Parameters
    ----------
    x : np.ndarray
        Data. Shape must be (n_samples, n_channels).
    mean : np.ndarray
        Mean. Shape must be (n_channels,).
    cov : np.ndarray
        Covariance. Shape must be (n_channels, n_channels).

    Returns
    -------
    log_likelihood : np.ndarray
        Log-likelihood of each sample. Shape is (n_samples,).
    """
    n_channels = x.shape[-1]
    cholesky = np.linalg.cholesky(cov)
    z = solve_triangular(cholesky, (x - mean).T, lower=True)
    log_det = 2 * np.sum(np.log(np.diag(cholesky)))
    return -0.5 * (n_channels * np.log(2 * np.pi) + log_det + np.sum(z**2, axis=0))


def mixture_log_likelihood(
    x: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """Log-likelihood of each sample under a time-varying multivariate normal.

    Parameters
    ----------
    x : np.ndarray
        Data. Shape must be (n_samples, n_channels).
    means : np.ndarray
        Mean at each time point. Shape must be (n_samples, n_channels).
    covs : np.ndarray
        Covariance at each time point. Shape must be
        (n_samples, n_channels, n_channels).
    chunk_size : int, optional
        Number of samples to process at once. By default, chunks of roughly
        100 MB are used.

    Returns
    -------
    log_likelihood : np.ndarray
        Log-likelihood of each sample. Shape is (n_samples,).
    """
    n_samples, n_channels = x.shape
    if chunk_size is None:
        chunk_size = max(int(1e8 // (8 * n_channels**2)), 1)

    ll = np.empty(n_samples)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        d = x[start:end] - means[start:end]
        cholesky = np.linalg.cholesky(covs[start:end])
        z = np.linalg.solve(cholesky, d[..., None])[..., 0]
        log_det = 2 * np.sum(np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1)
        ll[start:end] = -0.5 * (
            n_channels * np.log(2 * np.pi) + log_det + np.sum(z**2, axis=-1)
        )
    return ll


def fit_static(data, learn_means: bool = False) -> Dict[str, np.ndarray]:
    """Fit a static model (a single multivariate normal).

    This is the model with one state or mode. The covariance is the biased
    sample covariance (normalised by the number of samples) of all sessions
    concatenated. The mean is zero unless :code:`learn_means=True`.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data. Only the sessions in :code:`data.keep` are used.
    learn_means : bool, optional
        Should we estimate the mean?

    Returns
    -------
    spatial : dict
        :code:`"means"` with shape (1, n_channels) and :code:`"covs"` with
        shape (1, n_channels, n_channels).
    """
    ts = time_series(data)
    n_samples = sum(x.shape[0] for x in ts)
    n_channels = ts[0].shape[1]

    # The covariance is always calculated around the sample mean
    sample_mean = sum(np.sum(x, axis=0, dtype=np.float64) for x in ts) / n_samples
    cov = np.zeros([n_channels, n_channels])
    for x in ts:
        d = x - sample_mean
        cov += d.T @ d
    cov /= n_samples

    mean = sample_mean if learn_means else np.zeros(n_channels)
    return {"means": mean[None, :], "covs": cov[None, :, :]}


def static_log_likelihood(data, spatial: Dict[str, np.ndarray]) -> float:
    """Average log-likelihood per time point under a static model.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data. Only the sessions in :code:`data.keep` are used.
    spatial : dict
        Output of :py:func:`fit_static`.

    Returns
    -------
    log_likelihood : float
        Log-likelihood averaged over time points.
    """
    mean = spatial["means"].reshape(-1)
    cov = spatial["covs"].reshape(mean.shape[0], mean.shape[0])
    total = 0.0
    n_samples = 0
    for x in time_series(data):
        total += np.sum(gaussian_log_likelihood(x, mean, cov))
        n_samples += x.shape[0]
    return float(total / n_samples)


def _as_list(arrays: Union[List[np.ndarray], np.ndarray]) -> List[np.ndarray]:
    """Make sure per-session arrays are in a list."""
    if isinstance(arrays, np.ndarray):
        return [arrays]
    return list(arrays)


def _check_n_sessions(ts: List[np.ndarray], temporal: List[np.ndarray]) -> None:
    if len(ts) != len(temporal):
        raise ValueError(
            f"The data has {len(ts)} sessions but the time courses have "
            f"{len(temporal)} sessions."
        )


def _clear_session() -> None:
    """Free memory used by TensorFlow models that are no longer needed."""
    import tensorflow as tf

    tf.keras.backend.clear_session()


class ModelOps:
    """Base class for model-specific operations.

    Parameters
    ----------
    config_kwargs : dict
        Keyword arguments for the model's :code:`Config`. :code:`n_channels`
        is set from the data, so it does not need to be passed.
    init_kwargs : dict, optional
        Keyword arguments for the model's initialization method.
    fit_kwargs : dict, optional
        Keyword arguments for the model's :code:`fit` method.
    spatial_method : str, optional
        How to estimate the spatial parameters given the time courses.
        See the subclass for the options.
    """

    name = None
    default_config_kwargs = {}
    default_init_kwargs = {}
    spatial_methods = ()
    default_spatial_method = None

    def __init__(
        self,
        config_kwargs: dict,
        init_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        spatial_method: Optional[str] = None,
    ) -> None:
        self.config_kwargs = override_dict_defaults(
            self.default_config_kwargs, dict(config_kwargs)
        )
        self.init_kwargs = override_dict_defaults(
            self.default_init_kwargs, dict(init_kwargs or {})
        )
        self.fit_kwargs = dict(fit_kwargs or {})

        spatial_method = spatial_method or self.default_spatial_method
        if spatial_method not in self.spatial_methods:
            raise ValueError(
                f"spatial_method must be one of {self.spatial_methods} for "
                f"{self.name}, got '{spatial_method}'."
            )
        self.spatial_method = spatial_method

        if self.n_states is None:
            raise ValueError("config_kwargs must contain n_states or n_modes.")

    @property
    def n_states(self) -> Optional[int]:
        """Number of states/modes."""
        return self.config_kwargs.get("n_states", self.config_kwargs.get("n_modes"))

    @property
    def learn_means(self) -> bool:
        """Does the model learn the means?"""
        return bool(self.config_kwargs.get("learn_means", False))

    def train(self, data, spatial: Optional[Dict[str, np.ndarray]] = None):
        """Build, initialize and train a model.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Training data.
        spatial : dict, optional
            If passed, the spatial parameters are fixed to these values and
            only the temporal parameters are trained.

        Returns
        -------
        model : osl-dynamics model
            Trained model.
        """
        raise NotImplementedError

    def get_spatial(self, model) -> Dict[str, np.ndarray]:
        """Get the spatial parameters of a model."""
        raise NotImplementedError

    def get_temporal(self, model, data) -> List[np.ndarray]:
        """Get the time courses of a model for each session in the data."""
        raise NotImplementedError

    def fit(self, data) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        """Train the model and return its spatial and temporal parameters.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Training data.

        Returns
        -------
        spatial : dict
            Spatial parameters.
        temporal : list of np.ndarray
            Time courses for each session.
        """
        model = self.train(data)
        spatial = self.get_spatial(model)
        temporal = self.get_temporal(model, data)
        del model
        _clear_session()
        return spatial, temporal

    def infer_spatial(self, data, temporal: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Estimate the spatial parameters given fixed time courses.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Data.
        temporal : list of np.ndarray
            Time courses for each session in the data.

        Returns
        -------
        spatial : dict
            Spatial parameters.
        """
        raise NotImplementedError

    def infer_temporal(self, data, spatial: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Estimate the time courses given fixed spatial parameters.

        The temporal part of the model (e.g. transition probabilities or the
        inference network) is trained on the data with the spatial parameters
        fixed, using the same initialization and training settings as
        :code:`fit`.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Data.
        spatial : dict
            Spatial parameters.

        Returns
        -------
        temporal : list of np.ndarray
            Time courses for each session.
        """
        model = self.train(data, spatial=spatial)
        temporal = self.get_temporal(model, data)
        del model
        _clear_session()
        return temporal

    def log_likelihood(
        self, data, temporal: List[np.ndarray], spatial: Dict[str, np.ndarray]
    ) -> float:
        """Average log-likelihood per time point given fixed parameters.

        The time courses are used as passed: they are never re-inferred from
        the data.

        Parameters
        ----------
        data : osl_dynamics.data.Data
            Data.
        temporal : list of np.ndarray
            Time courses for each session in the data.
        spatial : dict
            Spatial parameters.

        Returns
        -------
        log_likelihood : float
            Log-likelihood averaged over time points.
        """
        raise NotImplementedError

    #: Metric used for naive cross-validation.
    ncv_metric = "free_energy"

    def ncv_score(self, model, data) -> float:
        """Naive cross-validation metric of a trained model on some data.

        Parameters
        ----------
        model : osl-dynamics model
            Trained model.
        data : osl_dynamics.data.Data
            Test data.

        Returns
        -------
        score : float
            Variational free energy.
        """
        return float(model.free_energy(data))

    def temporal_files(self, temporal) -> Dict[str, list]:
        """Files to save the time courses to."""
        return {"alp.pkl": temporal}


class HMMOps(ModelOps):
    """Operations for the :py:mod:`HMM <osl_dynamics.models.hmm>`.

    Spatial parameters are the state means and covariances. Temporal
    parameters are the state probabilities.

    The default :code:`config_kwargs` and :code:`init_kwargs` are the same
    as :py:func:`osl_dynamics.config_api.wrappers.train_hmm`.

    :code:`spatial_method` can be:

    - :code:`"sample"` (default): assign each time point to its most probable
      state and calculate the sample mean and covariance of each state.
    - :code:`"mle"`: weight each time point by the state probabilities (the
      maximum likelihood estimate given the state probabilities).
    - :code:`"weighted"`: same as :code:`"mle"` for the HMM.

    States with no time points assigned are given an identity covariance.
    """

    name = "hmm"
    default_config_kwargs = {
        "sequence_length": 200,
        "batch_size": 256,
        "learning_rate": 0.01,
        "n_epochs": 20,
    }
    default_init_kwargs = {"n_init": 5, "n_epochs": 2}
    spatial_methods = ("sample", "mle", "weighted")
    default_spatial_method = "sample"

    def _config(self, n_channels: int, spatial: Optional[Dict] = None):
        from osl_dynamics.models import hmm

        config_kwargs = dict(self.config_kwargs, n_channels=n_channels)
        if spatial is not None:
            config_kwargs.update(
                learn_means=False,
                learn_covariances=False,
                initial_means=spatial["means"],
                initial_covariances=spatial["covs"],
            )
        return hmm.Config(**config_kwargs)

    def train(self, data, spatial: Optional[Dict[str, np.ndarray]] = None):
        from osl_dynamics.models import hmm

        model = hmm.Model(self._config(data.n_channels, spatial))
        model.random_state_time_course_initialization(data, **self.init_kwargs)
        model.fit(data, **self.fit_kwargs)
        return model

    def get_spatial(self, model) -> Dict[str, np.ndarray]:
        means, covs = model.get_means_covariances()
        return {"means": means, "covs": covs}

    def get_temporal(self, model, data) -> List[np.ndarray]:
        return _as_list(model.get_alpha(data))

    def infer_spatial(self, data, temporal: List[np.ndarray]) -> Dict[str, np.ndarray]:
        from osl_dynamics.analysis.post_hoc import hmm_dual_estimation
        from osl_dynamics.inference.modes import argmax_time_courses

        ts = time_series(data)
        alpha = _as_list(temporal)
        _check_n_sessions(ts, alpha)

        # Pool the time points of all sessions
        x = np.concatenate([t[: a.shape[0]] for t, a in zip(ts, alpha)])
        a = np.concatenate(alpha).astype(np.float64)
        if self.spatial_method == "sample":
            a = argmax_time_courses(a).astype(np.float64)

        config = self._config(data.n_channels)
        n_states, n_channels = a.shape[1], x.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            means, covs = hmm_dual_estimation(
                x,
                a,
                zero_mean=not config.learn_means,
                diagonal_covariances=config.diagonal_covariances,
                eps=config.covariances_epsilon,
                n_jobs=None,
            )
        means = np.reshape(means, (n_states, n_channels))
        covs = np.reshape(covs, (n_states, n_channels, n_channels))

        # States with no time points
        for state in np.flatnonzero(np.sum(a, axis=0) == 0):
            _logger.warning(
                f"No time points assigned to state {state}. "
                "Using an identity covariance."
            )
            means[state] = 0
            covs[state] = (1 + config.covariances_epsilon) * np.eye(n_channels)

        return {"means": means, "covs": covs}

    def log_likelihood(
        self, data, temporal: List[np.ndarray], spatial: Dict[str, np.ndarray]
    ) -> float:
        from osl_dynamics.models import hmm

        ts = time_series(data)
        gamma = _as_list(temporal)
        _check_n_sessions(ts, gamma)

        model = hmm.Model(self._config(data.n_channels, spatial))
        total = 0.0
        n_samples = 0
        for x, g in zip(ts, gamma):
            x = np.asarray(x[: g.shape[0]], dtype=np.float32)
            ll = model.get_log_likelihood(x[None, ...])[0]
            total += np.sum(ll * g)
            n_samples += g.shape[0]
        del model
        _clear_session()

        return float(total / n_samples)

    def evidence(self, model, data) -> float:
        """Model evidence of a trained HMM on some data."""
        return float(model.evidence(data))


_MODEL_OPS = {"hmm": HMMOps}


def get_model_ops(
    model: str,
    config_kwargs: dict,
    init_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
    spatial_method: Optional[str] = None,
) -> ModelOps:
    """Get the operations for a model.

    Parameters
    ----------
    model : str
        Model name. Options are :code:`"hmm"`, :code:`"dynemo"`,
        :code:`"mdynemo"` and :code:`"swc"`.
    config_kwargs : dict
        Keyword arguments for the model's :code:`Config`.
    init_kwargs : dict, optional
        Keyword arguments for the model's initialization method.
    fit_kwargs : dict, optional
        Keyword arguments for the model's :code:`fit` method.
    spatial_method : str, optional
        How to estimate the spatial parameters given the time courses.

    Returns
    -------
    ops : ModelOps
        Model operations.
    """
    model = model.lower()
    if model not in _MODEL_OPS:
        raise NotImplementedError(
            f"{model} is not supported. Options are {', '.join(_MODEL_OPS)}."
        )
    return _MODEL_OPS[model](config_kwargs, init_kwargs, fit_kwargs, spatial_method)
