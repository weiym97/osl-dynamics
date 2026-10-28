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


def batched_gaussian_log_likelihood(
    x: np.ndarray, means: np.ndarray, covs: np.ndarray
) -> np.ndarray:
    """Log-likelihood of each sample under its own multivariate normal.

    Parameters
    ----------
    x : np.ndarray
        Data. Shape must be (n_samples, n_channels).
    means : np.ndarray
        Mean for each sample. Shape must be (n_samples, n_channels).
    covs : np.ndarray
        Covariance for each sample. Shape must be
        (n_samples, n_channels, n_channels).

    Returns
    -------
    log_likelihood : np.ndarray
        Log-likelihood of each sample. Shape is (n_samples,).
    """
    n_channels = x.shape[-1]
    cholesky = np.linalg.cholesky(covs)
    z = np.linalg.solve(cholesky, (x - means)[..., None])[..., 0]
    log_det = 2 * np.sum(np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * (n_channels * np.log(2 * np.pi) + log_det + np.sum(z**2, axis=-1))


def _chunks(n_samples: int, n_channels: int) -> List[slice]:
    """Slices of time points so each chunk of covariances is about 100 MB."""
    chunk_size = max(int(1e8 // (8 * n_channels**2)), 1)
    return [
        slice(start, min(start + chunk_size, n_samples))
        for start in range(0, n_samples, chunk_size)
    ]


def weighted_moments(
    x: np.ndarray,
    alpha: np.ndarray,
    zero_mean: bool,
    eps: float = 0.0,
    diagonal_covariances: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of each state/mode weighted by a time course.

    Parameters
    ----------
    x : np.ndarray
        Data. Shape must be (n_samples, n_channels).
    alpha : np.ndarray
        Weight of each state/mode at each time point. Shape must be
        (n_samples, n_states). Pass a one-hot time course for the sample
        mean and covariance of the time points assigned to each state.
    zero_mean : bool
        Should the means be zero?
    eps : float, optional
        Value added to the diagonal of each covariance.
    diagonal_covariances : bool, optional
        Should we only estimate the variances?

    Returns
    -------
    means : np.ndarray
        Means. Shape is (n_states, n_channels).
    covs : np.ndarray
        Covariances. Shape is (n_states, n_channels, n_channels). States
        with too few time points (non-zero weights) for a full-rank
        covariance get a zero mean and an identity covariance (plus
        :code:`eps`), the same as states with no time points.
    """
    from osl_dynamics.analysis.post_hoc import hmm_dual_estimation

    n_states, n_channels = alpha.shape[1], x.shape[1]

    # Minimum number of time points for a full-rank covariance
    min_samples = 1 if diagonal_covariances else n_channels
    if not zero_mean:
        min_samples += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        means, covs = hmm_dual_estimation(
            np.asarray(x, dtype=np.float64),
            np.asarray(alpha, dtype=np.float64),
            zero_mean=zero_mean,
            diagonal_covariances=diagonal_covariances,
            eps=eps,
            n_jobs=None,
        )
    means = np.reshape(means, (n_states, n_channels))
    covs = np.reshape(covs, (n_states, n_channels, n_channels))

    n_samples = np.count_nonzero(np.asarray(alpha) > 0, axis=0)
    for state in np.flatnonzero(n_samples < min_samples):
        _logger.warning(
            f"State/mode {state} has {n_samples[state]} time points, which is "
            f"too few for a full-rank covariance. Using an identity covariance."
        )
        means[state] = 0
        covs[state] = (1 + eps) * np.eye(n_channels)

    return means, covs


def fit_observation_model(
    x: np.ndarray,
    time_courses: List[np.ndarray],
    build_layers,
    mix,
    sequence_length: int,
    batch_size: int,
    learning_rate: float,
    n_epochs: int,
) -> list:
    """Maximum likelihood estimate of an observation model with fixed time
    courses.

    The observation model parameters are trained with Adam to maximize the
    log-likelihood of the data with the time courses held fixed. The data
    are split into sequences, which are shuffled each epoch.

    Parameters
    ----------
    x : np.ndarray
        Data. Shape must be (n_samples, n_channels).
    time_courses : list of np.ndarray
        Time courses (e.g. :code:`[alpha]` or :code:`[alpha, beta]`). Each
        has shape (n_samples, n_modes).
    build_layers : callable
        Function with no arguments that returns a list of osl-dynamics layers
        (e.g. :code:`VectorsLayer`, :code:`CovarianceMatricesLayer`).
    mix : callable
        Function :code:`mix(params, time_courses)` that returns the mean and
        covariance at each time point, where :code:`params` are the outputs
        of the layers.
    sequence_length : int
        Length of each sequence.
    batch_size : int
        Number of sequences in each batch.
    learning_rate : float
        Learning rate.
    n_epochs : int
        Number of epochs.

    Returns
    -------
    params : list of np.ndarray
        The output of each layer after training.
    """
    import tensorflow as tf
    import tensorflow_probability as tfp

    # Split into sequences, dropping the remainder
    n_sequences = x.shape[0] // sequence_length
    if n_sequences == 0:
        raise ValueError("There must be at least one sequence of data.")
    n_samples = n_sequences * sequence_length

    def _sequences(a):
        a = np.asarray(a[:n_samples], dtype=np.float32)
        return a.reshape(n_sequences, sequence_length, -1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (_sequences(x), tuple(_sequences(tc) for tc in time_courses))
    )
    dataset = dataset.shuffle(n_sequences).batch(batch_size)

    obs_layers = build_layers()
    dummy = tf.zeros([1, sequence_length, x.shape[1]])
    for layer in obs_layers:
        layer(dummy)  # build the layer
    variables = [v for layer in obs_layers for v in layer.trainable_variables]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(x_batch, tc_batch):
        with tf.GradientTape() as tape:
            params = [layer(x_batch) for layer in obs_layers]
            mu, sigma = mix(params, tc_batch)
            mvn = tfp.distributions.MultivariateNormalTriL(
                loc=mu,
                scale_tril=tf.linalg.cholesky(sigma),
                allow_nan_stats=False,
            )
            loss = -tf.reduce_mean(mvn.log_prob(x_batch))
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    for epoch in range(n_epochs):
        losses = [float(train_step(x_batch, tc_batch)) for x_batch, tc_batch in dataset]
        _logger.info(f"Epoch {epoch + 1}/{n_epochs}: loss={np.mean(losses):.6f}")

    return [layer(dummy).numpy() for layer in obs_layers]


def _pool(ts: List[np.ndarray], *time_courses: List[np.ndarray]) -> tuple:
    """Concatenate the sessions of data and time courses.

    The data of each session is trimmed to the length of its time course.
    """
    for tc in time_courses:
        _check_n_sessions(ts, tc)
    lengths = [a.shape[0] for a in time_courses[0]]
    x = np.concatenate([t[:n] for t, n in zip(ts, lengths)])
    return (x,) + tuple(np.concatenate(tc) for tc in time_courses)


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

    def training_metrics(self, model, data) -> Dict[str, float]:
        """Metrics of a trained model on its training data.

        Parameters
        ----------
        model : osl-dynamics model
            Trained model.
        data : osl_dynamics.data.Data
            Training data.

        Returns
        -------
        metrics : dict
            Variational free energy.
        """
        return {"free_energy": float(model.free_energy(data))}

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

    States with too few time points for a full-rank covariance (e.g. no
    time points assigned with :code:`"sample"`) are given an identity
    covariance.
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
        from osl_dynamics.inference.modes import argmax_time_courses

        # Pool the time points of all sessions
        x, alpha = _pool(time_series(data), _as_list(temporal))
        if self.spatial_method == "sample":
            alpha = argmax_time_courses(alpha)

        config = self._config(data.n_channels)
        means, covs = weighted_moments(
            x,
            alpha,
            zero_mean=not config.learn_means,
            eps=config.covariances_epsilon,
            diagonal_covariances=config.diagonal_covariances,
        )
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

    def training_metrics(self, model, data) -> Dict[str, float]:
        """Free energy and model evidence of a trained model on its data."""
        return {
            "free_energy": float(model.free_energy(data)),
            "evidence": float(model.evidence(data)),
        }


class DyNeMoOps(ModelOps):
    """Operations for :py:mod:`DyNeMo <osl_dynamics.models.dynemo>`.

    Spatial parameters are the mode means and covariances. Temporal
    parameters are the mixing coefficients (:code:`alpha`).

    The default :code:`config_kwargs` and :code:`init_kwargs` are the same
    as :py:func:`osl_dynamics.config_api.wrappers.train_dynemo`.

    :code:`spatial_method` can be:

    - :code:`"mle"` (default): maximize the log-likelihood of the data
      with :code:`alpha` fixed. The mean and covariance at each time point
      are :math:`m_t = \\sum_j \\alpha_{jt} \\mu_j` and
      :math:`C_t = \\sum_j \\alpha_{jt} D_j`. There is no closed form, so the
      parameters are trained with Adam using the :code:`learning_rate`,
      :code:`n_epochs`, :code:`sequence_length` and :code:`batch_size` in
      :code:`config_kwargs`, starting from the :code:`"weighted"` estimate.
    - :code:`"weighted"`: covariance of each mode weighted by :code:`alpha`.
      This is a closed-form approximation.
    - :code:`"sample"`: assign each time point to the mode with the largest
      :code:`alpha` and calculate the sample covariance of each mode.
    """

    name = "dynemo"
    default_config_kwargs = {
        "sequence_length": 200,
        "inference_n_units": 64,
        "inference_normalization": "layer",
        "model_n_units": 64,
        "model_normalization": "layer",
        "learn_alpha_temperature": True,
        "initial_alpha_temperature": 1.0,
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.01,
        "lr_decay": 0.1,
        "n_epochs": 40,
    }
    default_init_kwargs = {"n_init": 5, "n_epochs": 2, "take": 1}
    spatial_methods = ("mle", "weighted", "sample")
    default_spatial_method = "mle"

    def _config(self, n_channels: int, spatial: Optional[Dict] = None):
        from osl_dynamics.models import dynemo

        config_kwargs = dict(self.config_kwargs, n_channels=n_channels)
        if spatial is not None:
            config_kwargs.update(
                learn_means=False,
                learn_covariances=False,
                initial_means=spatial["means"],
                initial_covariances=spatial["covs"],
            )
        return dynemo.Config(**config_kwargs)

    def train(self, data, spatial: Optional[Dict[str, np.ndarray]] = None):
        from osl_dynamics.models import dynemo

        model = dynemo.Model(self._config(data.n_channels, spatial))
        model.set_regularizers(data)
        model.random_subset_initialization(data, **self.init_kwargs)
        model.fit(data, **self.fit_kwargs)
        return model

    def get_spatial(self, model) -> Dict[str, np.ndarray]:
        means, covs = model.get_means_covariances()
        return {"means": means, "covs": covs}

    def get_temporal(self, model, data) -> List[np.ndarray]:
        return _as_list(model.get_alpha(data))

    def infer_spatial(self, data, temporal: List[np.ndarray]) -> Dict[str, np.ndarray]:
        from osl_dynamics.inference.modes import argmax_time_courses

        config = self._config(data.n_channels)
        x, alpha = _pool(time_series(data), _as_list(temporal))

        weights = (
            argmax_time_courses(alpha) if self.spatial_method == "sample" else alpha
        )
        means, covs = weighted_moments(
            x,
            weights,
            zero_mean=not config.learn_means,
            eps=config.covariances_epsilon,
            diagonal_covariances=config.diagonal_covariances,
        )
        if self.spatial_method != "mle":
            return {"means": means, "covs": covs}

        from osl_dynamics.inference.layers import (
            CovarianceMatricesLayer,
            DiagonalMatricesLayer,
            MixMatricesLayer,
            MixVectorsLayer,
        )

        n_modes, n_channels = alpha.shape[1], x.shape[1]
        initial_means = means if config.learn_means else np.zeros_like(means)
        initial_covs = covs

        def build_layers():
            covs_layer = (
                DiagonalMatricesLayer
                if config.diagonal_covariances
                else (CovarianceMatricesLayer)
            )
            return [
                _means_layer(n_modes, n_channels, config.learn_means, initial_means),
                covs_layer(
                    n_modes,
                    n_channels,
                    True,
                    initial_covs.astype(np.float32),
                    config.covariances_epsilon,
                ),
            ]

        mix_vectors, mix_matrices = MixVectorsLayer(), MixMatricesLayer()

        def mix(params, time_courses):
            mu, D = params
            (a,) = time_courses
            return mix_vectors([a, mu]), mix_matrices([a, D])

        means, covs = fit_observation_model(
            x,
            [alpha],
            build_layers,
            mix,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            n_epochs=config.n_epochs,
        )
        _clear_session()
        return {"means": means, "covs": covs}

    def log_likelihood(
        self, data, temporal: List[np.ndarray], spatial: Dict[str, np.ndarray]
    ) -> float:
        ts = time_series(data)
        alpha = _as_list(temporal)
        _check_n_sessions(ts, alpha)

        means, covs = spatial["means"], spatial["covs"]
        total = 0.0
        n_samples = 0
        for x, a in zip(ts, alpha):
            x = x[: a.shape[0]]
            for chunk in _chunks(a.shape[0], x.shape[1]):
                m = a[chunk] @ means
                C = np.einsum("tk,kij->tij", a[chunk], covs)
                total += np.sum(batched_gaussian_log_likelihood(x[chunk], m, C))
            n_samples += a.shape[0]
        return float(total / n_samples)


class MDyNeMoOps(ModelOps):
    """Operations for :py:mod:`M-DyNeMo <osl_dynamics.models.mdynemo>`.

    Spatial parameters are the mode means, standard deviations and
    correlations. Temporal parameters are the mixing coefficients for the
    power (:code:`alpha`) and for the functional connectivity
    (:code:`beta`). The covariance at each time point is
    :math:`C_t = G_t F_t G_t`, with :math:`G_t = \\sum_j \\alpha_{jt} E_j`
    and :math:`F_t = \\sum_j \\beta_{jt} R_j`.

    The default :code:`config_kwargs` and :code:`init_kwargs` are the same
    as :py:func:`osl_dynamics.config_api.wrappers.train_mdynemo`. The
    correlations are initialized with k-means on sliding window correlations
    of the training data. :code:`init_kwargs` can contain
    :code:`corrs_init_kwargs` for this (see :code:`train_mdynemo`).
    :code:`n_corr_modes` defaults to :code:`n_modes`.

    :code:`spatial_method` can be:

    - :code:`"mle"` (default): maximize the log-likelihood of the data with
      :code:`alpha` and :code:`beta` fixed, using Adam (see
      :py:class:`DyNeMoOps`), starting from the :code:`"weighted"` estimate.
    - :code:`"weighted"`: standard deviations from the :code:`alpha`-weighted
      variances and correlations from the :code:`beta`-weighted covariances.
    - :code:`"sample"`: same as :code:`"weighted"` with the time points
      assigned to the mode with the largest :code:`alpha`/:code:`beta`.

    The channels of the data are used as the channels of the model, i.e.
    :code:`pca_components` is the identity.
    """

    name = "mdynemo"
    default_config_kwargs = {
        "sequence_length": 200,
        "inference_n_units": 64,
        "inference_normalization": "layer",
        "model_n_units": 64,
        "model_normalization": "layer",
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.01,
        "lr_decay": 0.1,
        "n_epochs": 40,
    }
    default_init_kwargs = {"n_init": 5, "n_epochs": 5, "take": 1}
    spatial_methods = ("mle", "weighted", "sample")
    default_spatial_method = "mle"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.corrs_init_kwargs = self.init_kwargs.pop("corrs_init_kwargs", {})
        if "n_corr_modes" not in self.config_kwargs:
            self.config_kwargs["n_corr_modes"] = self.n_states

    def _config(self, n_channels: int, spatial: Optional[Dict] = None):
        from osl_dynamics.models import mdynemo

        config_kwargs = dict(self.config_kwargs, n_channels=n_channels)
        if spatial is not None:
            config_kwargs.update(
                learn_means=False,
                learn_stds=False,
                learn_corrs=False,
                initial_means=spatial["means"],
                initial_stds=spatial["stds"],
                initial_corrs=spatial["corrs"],
            )
        return mdynemo.Config(**config_kwargs)

    def _initial_corrs(self, data, config) -> np.ndarray:
        """Initial correlations from k-means on sliding window correlations.

        This is the initialization used by
        :py:func:`osl_dynamics.config_api.wrappers.train_mdynemo`, applied to
        the sessions in the data only.
        """
        from sklearn.cluster import KMeans

        from osl_dynamics.analysis.connectivity import sliding_window_connectivity
        from osl_dynamics.utils.array_ops import cov2corr

        fs = data.sampling_frequency
        defaults = {"random_state": None, "n_init": "auto", "init": "k-means++"}
        if fs is not None:
            defaults.update(window_length=fs * 2, step_size=fs // 25)
        kwargs = override_dict_defaults(defaults, self.corrs_init_kwargs)
        if "window_length" not in kwargs or "step_size" not in kwargs:
            raise ValueError(
                "The data has no sampling_frequency, so window_length and "
                "step_size must be passed in init_kwargs['corrs_init_kwargs']."
            )

        tv_corr = sliding_window_connectivity(
            time_series(data),
            window_length=int(kwargs["window_length"]),
            step_size=int(kwargs["step_size"]),
            conn_type="corr",
            concatenate=True,
            n_jobs=data.n_jobs,
        )
        tv_corr = np.reshape(tv_corr, (tv_corr.shape[0], -1))
        kmeans = KMeans(
            n_clusters=config.n_corr_modes,
            n_init=kwargs["n_init"],
            init=kwargs["init"],
            random_state=kwargs["random_state"],
        ).fit(tv_corr)
        corrs = kmeans.cluster_centers_.reshape(
            config.n_corr_modes, config.n_channels, config.n_channels
        )
        return cov2corr(corrs)

    def train(self, data, spatial: Optional[Dict[str, np.ndarray]] = None):
        from osl_dynamics.models import mdynemo

        config = self._config(data.n_channels, spatial)
        if spatial is None:
            config.initial_corrs = self._initial_corrs(data, config)
        model = mdynemo.Model(config)
        model.random_subset_initialization(data, **self.init_kwargs)
        model.fit(data, **self.fit_kwargs)
        return model

    def get_spatial(self, model) -> Dict[str, np.ndarray]:
        means, stds, corrs = model.get_means_stds_corrs()
        return {"means": means, "stds": stds, "corrs": corrs}

    def get_temporal(self, model, data) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        alpha, beta = model.get_mode_time_courses(data)
        return _as_list(alpha), _as_list(beta)

    def infer_spatial(
        self, data, temporal: Tuple[List[np.ndarray], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        from osl_dynamics.inference.modes import argmax_time_courses
        from osl_dynamics.utils.array_ops import cov2corr

        config = self._config(data.n_channels)
        alpha, beta = temporal
        x, alpha, beta = _pool(time_series(data), _as_list(alpha), _as_list(beta))
        n_modes, n_channels = alpha.shape[1], x.shape[1]

        if self.spatial_method == "sample":
            alpha_weights = argmax_time_courses(alpha)
            beta_weights = argmax_time_courses(beta)
        else:
            alpha_weights, beta_weights = alpha, beta

        means, alpha_covs = weighted_moments(
            x, alpha_weights, zero_mean=not config.learn_means
        )
        _, beta_covs = weighted_moments(x, beta_weights, zero_mean=True)
        stds = np.sqrt(np.diagonal(alpha_covs, axis1=1, axis2=2))
        stds = np.array([np.diag(s) for s in stds]) + config.stds_epsilon * np.eye(
            n_channels
        )
        corrs = cov2corr(beta_covs)
        if self.spatial_method != "mle":
            return {"means": means, "stds": stds, "corrs": corrs}

        from osl_dynamics.inference.layers import (
            CorrelationMatricesLayer,
            DiagonalMatricesLayer,
            MixMatricesLayer,
            MixVectorsLayer,
        )

        n_corr_modes = beta.shape[1]
        initial_means = means if config.learn_means else np.zeros_like(means)
        initial_stds = np.maximum(np.diagonal(stds, axis1=1, axis2=2), 1e-6)

        def build_layers():
            return [
                _means_layer(n_modes, n_channels, config.learn_means, initial_means),
                DiagonalMatricesLayer(
                    n_modes,
                    n_channels,
                    True,
                    initial_stds.astype(np.float32),
                    config.stds_epsilon,
                ),
                CorrelationMatricesLayer(
                    n_corr_modes,
                    n_channels,
                    True,
                    corrs.astype(np.float32),
                    config.corrs_epsilon,
                ),
            ]

        mix_means = MixVectorsLayer()
        mix_stds = MixMatricesLayer()
        mix_corrs = MixMatricesLayer()

        def mix(params, time_courses):
            mu, E, R = params
            a, b = time_courses
            G = mix_stds([a, E])
            return mix_means([a, mu]), G @ mix_corrs([b, R]) @ G

        means, stds, corrs = fit_observation_model(
            x,
            [alpha, beta],
            build_layers,
            mix,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            n_epochs=config.n_epochs,
        )
        _clear_session()
        return {"means": means, "stds": stds, "corrs": corrs}

    def log_likelihood(
        self,
        data,
        temporal: Tuple[List[np.ndarray], List[np.ndarray]],
        spatial: Dict[str, np.ndarray],
    ) -> float:
        ts = time_series(data)
        alpha, beta = (_as_list(tc) for tc in temporal)
        _check_n_sessions(ts, alpha)
        _check_n_sessions(ts, beta)

        means, stds, corrs = spatial["means"], spatial["stds"], spatial["corrs"]
        total = 0.0
        n_samples = 0
        for x, a, b in zip(ts, alpha, beta):
            x = x[: a.shape[0]]
            for chunk in _chunks(a.shape[0], x.shape[1]):
                m = a[chunk] @ means
                G = np.einsum("tk,kij->tij", a[chunk], stds)
                F = np.einsum("tk,kij->tij", b[chunk], corrs)
                total += np.sum(batched_gaussian_log_likelihood(x[chunk], m, G @ F @ G))
            n_samples += a.shape[0]
        return float(total / n_samples)

    def temporal_files(self, temporal) -> Dict[str, list]:
        alpha, beta = temporal
        return {"alp.pkl": alpha, "bet.pkl": beta}


class SWCOps(ModelOps):
    """Operations for :py:mod:`SWC <osl_dynamics.models.swc>`.

    Spatial parameters are the state means (zero) and covariances. Temporal
    parameters are the state assigned to each window.

    The default :code:`config_kwargs` are :code:`window_length=100` and
    :code:`window_offset=75`. The spatial parameters given the states are
    always the average window covariance of each state, so every
    :code:`spatial_method` gives the same result.
    """

    name = "swc"
    default_config_kwargs = {
        "window_length": 100,
        "window_offset": 75,
        "learn_means": False,
        "learn_covariances": True,
    }
    spatial_methods = ("sample", "mle", "weighted")
    default_spatial_method = "sample"
    ncv_metric = "log_likelihood"

    def _config(self, n_channels: int, spatial: Optional[Dict] = None):
        from osl_dynamics.models import swc

        config_kwargs = dict(self.config_kwargs, n_channels=n_channels)
        if spatial is not None:
            config_kwargs.update(
                learn_covariances=False,
                initial_means=spatial["means"],
                initial_covariances=spatial["covs"],
            )
        return swc.Config(**config_kwargs)

    def train(self, data, spatial: Optional[Dict[str, np.ndarray]] = None):
        from osl_dynamics.models import swc

        model = swc.Model(self._config(data.n_channels, spatial))
        model.fit(data)
        return model

    def get_spatial(self, model) -> Dict[str, np.ndarray]:
        means, covs = model.get_means_covariances()
        return {"means": means, "covs": covs}

    def get_temporal(self, model, data) -> List[np.ndarray]:
        return _as_list(model.get_alpha(data))

    def fit(self, data) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        from osl_dynamics.models import swc

        model = swc.Model(self._config(data.n_channels))
        alpha = model.fit(data)
        return self.get_spatial(model), _as_list(alpha)

    def infer_temporal(self, data, spatial: Dict[str, np.ndarray]) -> List[np.ndarray]:
        from osl_dynamics.models import swc

        # With fixed covariances, each window is assigned to the nearest state
        model = swc.Model(self._config(data.n_channels, spatial))
        return self.get_temporal(model, data)

    def infer_spatial(self, data, temporal: List[np.ndarray]) -> Dict[str, np.ndarray]:
        from osl_dynamics.models import swc

        model = swc.Model(self._config(data.n_channels))
        means, covs = model.dual_estimation(data, _as_list(temporal))
        return {"means": means, "covs": covs}

    def log_likelihood(
        self, data, temporal: List[np.ndarray], spatial: Dict[str, np.ndarray]
    ) -> float:
        from osl_dynamics.models import swc

        model = swc.Model(self._config(data.n_channels, spatial))
        return model.log_likelihood(data, _as_list(temporal))

    def ncv_score(self, model, data) -> float:
        """Log-likelihood of the data with the states assigned to its windows."""
        return model.log_likelihood(data, _as_list(model.get_alpha(data)))

    def training_metrics(self, model, data) -> Dict[str, float]:
        """Log-likelihood of a trained model on its data."""
        return {"log_likelihood": self.ncv_score(model, data)}


def _means_layer(n: int, m: int, learn: bool, initial_value: np.ndarray):
    from osl_dynamics.inference.layers import VectorsLayer

    return VectorsLayer(n, m, learn, initial_value.astype(np.float32))


_MODEL_OPS = {
    "hmm": HMMOps,
    "dynemo": DyNeMoOps,
    "mdynemo": MDyNeMoOps,
    "swc": SWCOps,
}


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
