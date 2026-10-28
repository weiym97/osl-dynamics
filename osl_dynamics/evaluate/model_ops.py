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
:py:func:`static_log_likelihood` for every model without session-specific
parameters (HIVE and DIVE have them).
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


def batched_precision_log_likelihood(
    x: np.ndarray, means: np.ndarray, precisions: np.ndarray
) -> np.ndarray:
    """Log-likelihood of each sample under its own multivariate normal,
    parameterised by a precision matrix.

    Parameters
    ----------
    x : np.ndarray
        Data. Shape must be (n_samples, n_channels).
    means : np.ndarray
        Mean for each sample. Shape must be (n_samples, n_channels).
    precisions : np.ndarray
        Precision matrix for each sample. Shape must be
        (n_samples, n_channels, n_channels).

    Returns
    -------
    log_likelihood : np.ndarray
        Log-likelihood of each sample. Shape is (n_samples,).
    """
    n_channels = x.shape[-1]
    cholesky = np.linalg.cholesky(precisions)
    z = np.einsum("...ji,...j->...i", cholesky, x - means)
    log_det = 2 * np.sum(np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * (n_channels * np.log(2 * np.pi) - log_det + np.sum(z**2, axis=-1))


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
    precisions: bool = False,
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
        covariance (or precision if :code:`precisions=True`) at each time
        point, where :code:`params` are the outputs of the layers.
    sequence_length : int
        Length of each sequence.
    batch_size : int
        Number of sequences in each batch.
    learning_rate : float
        Learning rate.
    n_epochs : int
        Number of epochs.
    precisions : bool, optional
        Does :code:`mix` return precision matrices?

    Returns
    -------
    params : list of np.ndarray
        The output of each layer after training.
    """
    import tensorflow as tf
    import tensorflow_probability as tfp

    from osl_dynamics.inference.layers import precision_log_prob

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
            mu, matrix = mix(params, tc_batch)
            if precisions:
                log_prob = precision_log_prob(x_batch, mu, matrix)
            else:
                mvn = tfp.distributions.MultivariateNormalTriL(
                    loc=mu,
                    scale_tril=tf.linalg.cholesky(matrix),
                    allow_nan_stats=False,
                )
                log_prob = mvn.log_prob(x_batch)
            loss = -tf.reduce_mean(log_prob)
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
    best_of : int, optional
        Number of times :code:`fit` trains the model. The run with the
        lowest :py:meth:`selection_score` on the training data is kept.
    """

    name = None
    default_config_kwargs = {}
    default_init_kwargs = {}
    spatial_methods = ()
    default_spatial_method = None
    #: Does the model have session-specific parameters? Such a model is not
    #: static with one state/mode.
    session_specific = False

    def __init__(
        self,
        config_kwargs: dict,
        init_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        spatial_method: Optional[str] = None,
        best_of: int = 1,
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

        if best_of < 1:
            raise ValueError("best_of must be at least 1.")
        self.best_of = best_of

    @property
    def n_states(self) -> Optional[int]:
        """Number of states/modes."""
        return self.config_kwargs.get("n_states", self.config_kwargs.get("n_modes"))

    @property
    def is_static(self) -> bool:
        """Is the model a static model (one multivariate normal)?"""
        return self.n_states == 1 and not self.session_specific

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

        The model is trained :code:`best_of` times and the run with the
        lowest :py:meth:`selection_score` is kept.

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
        best_score = np.inf
        for run in range(self.best_of):
            model = self.train(data)
            # One run needs no score
            score = 0.0
            if self.best_of > 1:
                score = self.selection_score(model, data)
                _logger.info(f"Run {run + 1}/{self.best_of}: score={score:.4f}")
            if score < best_score:
                best_score = score
                spatial = self.get_spatial(model)
                temporal = self.get_temporal(model, data)
            del model
            _clear_session()
        return spatial, temporal

    def selection_score(self, model, data) -> float:
        """Score used to choose between the :code:`best_of` runs of
        :code:`fit` (lower is better).

        Parameters
        ----------
        model : osl-dynamics model
            Trained model.
        data : osl_dynamics.data.Data
            Training data.

        Returns
        -------
        score : float
            Variational free energy.
        """
        return float(model.free_energy(data))

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

    def log_likelihood_controls(
        self, data, temporal, spatial: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Log-likelihoods of control models, saved with the BCV result.

        Takes the same arguments as :py:meth:`log_likelihood`. The base class
        has no controls.

        Returns
        -------
        controls : dict
            Log-likelihood of each control, averaged over time points.
        """
        return {}

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

    With :code:`mix_precisions=True` in :code:`config_kwargs`, the precision
    at each time point is :math:`P_t = \\sum_j \\alpha_{jt} D_j^{-1}`: the
    :code:`"mle"` estimate and the log-likelihood mix the precisions. The
    spatial parameters are still the covariances :math:`D_j`.
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
        from osl_dynamics.models.obs_mod import invert_matrices

        n_modes, n_channels = alpha.shape[1], x.shape[1]
        initial_means = means if config.learn_means else np.zeros_like(means)
        # With mix_precisions=True, the layer learns the precisions
        initial_covs = (
            invert_matrices(covs, config.diagonal_covariances)
            if config.mix_precisions
            else covs
        )

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
            precisions=config.mix_precisions,
        )
        if config.mix_precisions:
            covs = invert_matrices(covs, config.diagonal_covariances)
        _clear_session()
        return {"means": means, "covs": covs}

    def log_likelihood(
        self, data, temporal: List[np.ndarray], spatial: Dict[str, np.ndarray]
    ) -> float:
        ts = time_series(data)
        alpha = _as_list(temporal)
        _check_n_sessions(ts, alpha)

        means, covs = spatial["means"], spatial["covs"]
        mix_precisions = self.config_kwargs.get("mix_precisions", False)
        matrices = np.linalg.inv(covs) if mix_precisions else covs
        total = 0.0
        n_samples = 0
        for x, a in zip(ts, alpha):
            x = x[: a.shape[0]]
            for chunk in _chunks(a.shape[0], x.shape[1]):
                m = a[chunk] @ means
                D = np.einsum("tk,kij->tij", a[chunk], matrices)
                if mix_precisions:
                    total += np.sum(batched_precision_log_likelihood(x[chunk], m, D))
                else:
                    total += np.sum(batched_gaussian_log_likelihood(x[chunk], m, D))
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

        if self.best_of > 1:
            raise ValueError("best_of is not supported for SWC.")
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


_EMBEDDINGS_LAYER = "session_id_embeddings"

# Layers that turn a session embedding into a deviation, for each parameter
_DEV_LAYERS = [
    "{}_spatial_embeddings",
    "{}_dev_decoder",
    "{}_dev_map",
    "norm_{}_dev_map",
    "{}_dev_mag_mod_beta",
]


def _add_session_ids(data) -> None:
    """Label each session in the data with its index."""
    if "session_id" not in [label.name for label in data.session_labels]:
        data.add_session_labels("session_id", np.arange(data.n_sessions), "categorical")


def _session_sequences(
    ts: List[np.ndarray], time_courses: List[np.ndarray], sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the data and time courses of each session into sequences.

    The remainder of each session is dropped. Returns the sequences of data,
    time courses and session indices.
    """
    x, tc, ids = [], [], []
    for i, (t, c) in enumerate(zip(ts, time_courses)):
        n_sequences = c.shape[0] // sequence_length
        n_samples = n_sequences * sequence_length
        shape = (n_sequences, sequence_length, -1)
        x.append(np.asarray(t[:n_samples], dtype=np.float32).reshape(shape))
        tc.append(np.asarray(c[:n_samples], dtype=np.float32).reshape(shape))
        ids.append(np.full((n_sequences, sequence_length), i, dtype=np.int32))
    return np.concatenate(x), np.concatenate(tc), np.concatenate(ids)


def _predict_magnitudes(
    train_X: Dict[str, np.ndarray],
    train_Y: Dict[str, np.ndarray],
    test_X: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Predict the deviation magnitudes of channels Y from those of channels X.

    For each state/mode, the magnitudes of the training sessions for
    channels Y are regressed on their magnitudes for channels X, and the fit
    is applied to the test sessions' magnitudes for channels X. Negative
    predictions are set to zero.

    Parameters
    ----------
    train_X, train_Y, test_X : dict
        Magnitudes with keys :code:`"<param>_magnitude"`. Each value has
        shape (n_sessions, n_states).

    Returns
    -------
    magnitudes : dict
        Predicted magnitudes of the test sessions for channels Y.
    """
    predicted = {}
    for key in train_Y:
        if not key.endswith("_magnitude"):
            continue
        x, y, test = train_X[key], train_Y[key], test_X[key]
        prediction = np.empty_like(test)
        for k in range(test.shape[1]):
            slope, intercept = np.polyfit(x[:, k], y[:, k], 1)
            prediction[:, k] = intercept + slope * test[:, k]
        predicted[key] = np.maximum(prediction, 0)
    return predicted


class HIVEOps(ModelOps):
    """Operations for :py:mod:`HIVE <osl_dynamics.models.hive>`.

    HIVE gives each session its own state means and covariances, generated
    from a session embedding by a decoder. Bi-cross-validation treats each
    parameter according to what it is indexed by:

    - **Channel-specific** parameters are learnt on the training sessions,
      separately for channels X (step 2) and Y (step 1). These are the group
      means/covariances and the decoder that turns an embedding into a
      deviation (spatial embeddings, MLP, deviation map, its layer
      normalisation and the head giving the prior on the deviation
      magnitudes).
    - **Session-specific parameters shared by all channels** are inferred
      from channels X for the test sessions (step 3) and used for channels Y
      (step 4). These are the state probabilities and the session
      embeddings. The training sessions' embeddings are fixed in step 2, so
      the decoders of X and Y share one embedding space.
    - **Deviation magnitudes** are session- and channel-specific. For
      channels Y of a test session they are predicted from its magnitudes
      for channels X, with a regression for each state learnt on the
      training sessions.

    Spatial parameters are a dict with the group :code:`"means"` and
    :code:`"covs"`, the :code:`"weights"` of the channel-specific layers
    and the posterior deviation :code:`"magnitudes"` of the training
    sessions. Temporal parameters are a dict with the state probabilities
    (:code:`"gamma"`), the session :code:`"embeddings"` and the deviation
    :code:`"magnitudes"`.

    Step 2 maximizes HIVE's loss with the state probabilities fixed (Adam,
    :code:`n_epochs` epochs). Step 3 trains HIVE on channels X of the test
    sessions with the channel-specific layers fixed. Each test session
    starts at the mean training embedding and magnitudes.

    With one state, HIVE is not a static model: each session has its own
    covariance.

    :py:meth:`log_likelihood_controls` gives the log-likelihood with zero
    deviations (:code:`"log_likelihood_group"`, i.e. the group parameters)
    and with the embeddings and magnitudes shuffled between the test
    sessions (:code:`"log_likelihood_shuffled"`). :code:`log_likelihood`
    minus these is the gain from session-specific parameters.

    Only the :code:`session_id` embedding is supported. The default
    :code:`config_kwargs` and :code:`init_kwargs` are the same as
    :py:func:`osl_dynamics.config_api.wrappers.train_hive`.
    :code:`embeddings_dim` must be passed. The only :code:`spatial_method`
    is :code:`"mle"`.
    """

    name = "hive"
    session_specific = True
    default_config_kwargs = {
        "sequence_length": 200,
        "spatial_embeddings_dim": 2,
        "dev_n_layers": 5,
        "dev_n_units": 32,
        "dev_activation": "tanh",
        "dev_normalization": "layer",
        "dev_regularizer": "l1",
        "dev_regularizer_factor": 10,
        "batch_size": 128,
        "learning_rate": 0.005,
        "lr_decay": 0.1,
        "n_epochs": 30,
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 15,
    }
    default_init_kwargs = {"n_init": 10, "n_epochs": 2}
    spatial_methods = ("mle",)
    default_spatial_method = "mle"

    # Key of the time courses in the temporal parameters
    _time_courses = "gamma"
    # Config option for precision matrices
    _precisions_option = "embed_precisions"
    # Config overrides for step 3
    _temporal_overrides = {"do_kl_annealing": False}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config_kwargs.get("session_labels"):
            raise ValueError(
                f"Only the session_id embedding is supported for {self.name}."
            )

    @property
    def precisions(self) -> bool:
        """Are the session matrices precisions?"""
        return bool(self.config_kwargs.get(self._precisions_option, False))

    def _model(self, n_channels: int, n_sessions: int, data=None, **overrides):
        from osl_dynamics.models import hive

        config = hive.Config(
            **dict(
                self.config_kwargs,
                n_channels=n_channels,
                n_sessions=n_sessions,
                **overrides,
            )
        )
        model = hive.Model(config)
        if data is not None:
            model.set_regularizers(data)
        return model

    def _initialize(self, model, data) -> None:
        if self.n_states == 1:
            # There is no state time course to sample, so start at the
            # pooled covariance
            x = np.concatenate(time_series(data))
            means, covs = weighted_moments(
                x,
                np.ones((x.shape[0], 1)),
                zero_mean=not model.config.learn_means,
                eps=model.config.covariances_epsilon,
            )
            model.set_group_means_covariances(means, covs)
        else:
            model.random_state_time_course_initialization(data, **self.init_kwargs)

    def _params(self, model) -> List[str]:
        """Parameters with session deviations."""
        return [
            param
            for param, learn in [
                ("means", model.config.learn_means),
                ("covs", model.config.learn_covariances),
            ]
            if learn
        ]

    def _channel_layers(self, model) -> List[str]:
        names = ["group_means", "group_covs"]
        for param in self._params(model):
            names += [name.format(param) for name in _DEV_LAYERS]
        return names

    def _set_channel_weights(self, model, weights: Dict[str, list]) -> None:
        for name, value in weights.items():
            model.get_layer(name).set_weights(value)

    def _set_embeddings(self, model, embeddings: np.ndarray) -> None:
        model.get_layer(_EMBEDDINGS_LAYER).embedding_layer.embeddings.assign(
            np.asarray(embeddings, dtype=np.float32)
        )

    def _magnitudes(self, model) -> Dict[str, np.ndarray]:
        """Posterior shape and mean of each session's deviation magnitudes."""
        from osl_dynamics.models import obs_mod

        magnitudes = {}
        for param in self._params(model):
            alpha, beta = obs_mod.get_dev_mag_parameters(model.model, param)
            magnitudes[f"{param}_alpha"] = alpha[..., 0]
            magnitudes[f"{param}_magnitude"] = alpha[..., 0] / beta[..., 0]
        return magnitudes

    def _set_magnitudes(
        self, model, param: str, alpha: np.ndarray, magnitude: np.ndarray
    ) -> None:
        """Set the posterior of each session's deviation magnitudes to a
        gamma distribution with this shape and mean."""
        import tensorflow_probability as tfp

        shape = (model.config.n_sessions, self.n_states)
        alpha = np.broadcast_to(alpha, shape)
        beta = alpha / np.broadcast_to(magnitude, shape)
        for name, value in [("alpha", alpha), ("beta", beta)]:
            model.get_layer(f"{param}_dev_mag_inf_{name}_input").set_weights(
                [tfp.math.softplus_inverse(value[..., None].astype(np.float32)).numpy()]
            )

    def train(self, data, spatial: Optional[Dict] = None):
        if spatial is not None:
            raise NotImplementedError(
                f"Use infer_temporal to train {self.name} with fixed spatial "
                "parameters."
            )
        _add_session_ids(data)
        model = self._model(data.n_channels, data.n_sessions, data)
        model.set_dev_parameters_initializer(data)
        self._initialize(model, data)
        model.fit(data, **self.fit_kwargs)
        return model

    def get_spatial(self, model) -> Dict:
        means, covs = model.get_group_means_covariances()
        return {
            "means": means,
            "covs": covs,
            "weights": {
                name: model.get_layer(name).get_weights()
                for name in self._channel_layers(model)
            },
            "magnitudes": self._magnitudes(model),
        }

    def get_temporal(self, model, data) -> Dict:
        embeddings = model.get_layer(_EMBEDDINGS_LAYER).embedding_layer.embeddings
        return {
            self._time_courses: _as_list(model.get_alpha(data, verbose=0)),
            "embeddings": embeddings.numpy(),
            "magnitudes": self._magnitudes(model),
        }

    def _fixed_time_course_loss(self, model):
        """Sub-model and loss for training the observation model with fixed
        state probabilities.

        This is HIVE's loss with Baum-Welch replaced by the fixed state
        probabilities. The KL and regularization terms are the losses of the
        sub-model.
        """
        import tensorflow as tf

        keras_model = model.model
        keras_model.get_layer("kl_loss").annealing_factor.assign(1.0)
        sub_model = tf.keras.Model(
            inputs=keras_model.inputs,
            outputs=[
                keras_model.get_layer("ll").output,
                keras_model.get_layer("kl_loss").output,
            ],
        )

        def loss(outputs, x, gamma):
            ll, _ = outputs
            return -tf.reduce_mean(tf.reduce_sum(gamma * ll, axis=-1))

        return sub_model, loss

    def infer_spatial(self, data, temporal: Dict) -> Dict:
        import tensorflow as tf

        time_courses = _as_list(temporal[self._time_courses])
        embeddings = temporal["embeddings"]
        ts = time_series(data)
        _check_n_sessions(ts, time_courses)

        _add_session_ids(data)
        model = self._model(data.n_channels, data.n_sessions, data)
        model.set_dev_parameters_initializer(data)

        # Group parameters start at the weighted moments
        config = model.config
        x, tc = _pool(ts, time_courses)
        means, covs = weighted_moments(
            x,
            tc,
            zero_mean=not config.learn_means,
            eps=config.covariances_epsilon,
        )
        model.set_group_means_covariances(means, covs)

        # Fix the training sessions' embeddings so the decoder of these
        # channels uses the same embedding space as the model from step 1
        self._set_embeddings(model, embeddings)
        model.model.get_layer(_EMBEDDINGS_LAYER).trainable = False

        sub_model, loss_function = self._fixed_time_course_loss(model)
        variables = sub_model.trainable_variables

        x_seq, tc_seq, id_seq = _session_sequences(
            ts, time_courses, config.sequence_length
        )
        dataset = (
            tf.data.Dataset.from_tensor_slices(((x_seq, id_seq), tc_seq))
            .shuffle(x_seq.shape[0])
            .batch(config.batch_size)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        @tf.function
        def train_step(inputs, tc_batch):
            with tf.GradientTape() as tape:
                outputs = sub_model(list(inputs), training=True)
                loss = loss_function(outputs, inputs[0], tc_batch)
                if sub_model.losses:
                    loss += tf.add_n(sub_model.losses)
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(
                [(g, v) for g, v in zip(gradients, variables) if g is not None]
            )
            return loss

        for epoch in range(config.n_epochs):
            optimizer.learning_rate.assign(
                config.learning_rate * np.exp(-config.lr_decay * epoch)
            )
            losses = [float(train_step(i, tc)) for i, tc in dataset]
            _logger.info(
                f"Epoch {epoch + 1}/{config.n_epochs}: loss={np.mean(losses):.6f}"
            )

        spatial = self.get_spatial(model)
        spatial["embeddings_mean"] = np.mean(embeddings, axis=0)
        del model, sub_model
        _clear_session()
        return spatial

    def infer_temporal(self, data, spatial: Dict) -> Dict:
        _add_session_ids(data)
        model = self._model(
            data.n_channels, data.n_sessions, data, **self._temporal_overrides
        )
        self._set_channel_weights(model, spatial["weights"])

        # Start each test session at the average training session
        self._set_embeddings(
            model, np.tile(spatial["embeddings_mean"], (data.n_sessions, 1))
        )
        magnitudes = spatial["magnitudes"]
        for param in self._params(model):
            self._set_magnitudes(
                model,
                param,
                magnitudes[f"{param}_alpha"].mean(axis=0),
                magnitudes[f"{param}_magnitude"].mean(axis=0),
            )

        # The initialization methods would overwrite the fixed group
        # parameters, so the temporal model starts at its initial weights
        with model.set_trainable(list(spatial["weights"]), False):
            model.fit(data, **self.fit_kwargs)
        temporal = self.get_temporal(model, data)

        # The training sessions' magnitudes for these channels are needed to
        # predict the magnitudes of the other channels in step 4
        temporal["train_magnitudes"] = magnitudes
        del model
        _clear_session()
        return temporal

    def session_matrices(
        self,
        n_channels: int,
        spatial: Dict,
        embeddings: np.ndarray,
        magnitudes: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Means and matrices of each session, generated from its embedding.

        Parameters
        ----------
        n_channels : int
            Number of channels.
        spatial : dict
            Spatial parameters.
        embeddings : np.ndarray
            Session embeddings. Shape must be (n_sessions, embeddings_dim).
        magnitudes : dict
            Deviation magnitudes with keys :code:`"<param>_magnitude"`,
            each with shape (n_sessions, n_states). If :code:`None`, the
            deviations are zero, i.e. every session has the group
            parameters.

        Returns
        -------
        means : np.ndarray
            Shape is (n_sessions, n_states, n_channels).
        matrices : np.ndarray
            Covariances, or precisions if :code:`precisions=True`. Shape is
            (n_sessions, n_states, n_channels, n_channels).
        """
        from osl_dynamics.models import obs_mod

        n_sessions = embeddings.shape[0]
        model = self._model(n_channels, n_sessions)
        self._set_channel_weights(model, spatial["weights"])
        self._set_embeddings(model, embeddings)
        config = model.config

        params = {}
        for param, learn in [
            ("means", config.learn_means),
            ("covs", config.learn_covariances),
        ]:
            group = obs_mod.get_observation_model_parameter(
                model.model, f"group_{param}"
            )
            if learn and magnitudes is not None:
                dev_map = obs_mod.get_dev_map(model.model, param, config.session_labels)
                dev = magnitudes[f"{param}_magnitude"][..., None] * dev_map
            else:
                size = (
                    n_channels
                    if param == "means"
                    else n_channels * (n_channels + 1) // 2
                )
                dev = np.zeros((n_sessions, self.n_states, size), dtype=np.float32)
            params[param] = model.get_layer(f"session_{param}")([group, dev]).numpy()

        del model
        _clear_session()
        return params["means"], params["covs"]

    def session_log_likelihoods(
        self,
        data,
        time_courses: List[np.ndarray],
        embeddings: np.ndarray,
        spatial: Dict,
        magnitudes: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Log-likelihood of each session given its time courses, embedding
        and magnitudes (see :py:meth:`session_matrices`).

        Returns
        -------
        log_likelihoods : np.ndarray
            Log-likelihood summed over time points and the number of time
            points for each session. Shape is (n_sessions, 2).
        """
        gamma = _as_list(time_courses)
        ts = time_series(data)
        _check_n_sessions(ts, gamma)

        means, matrices = self.session_matrices(
            data.n_channels, spatial, embeddings, magnitudes
        )
        covs = np.linalg.inv(matrices) if self.precisions else matrices
        log_likelihoods = []
        for x, g, m, C in zip(ts, gamma, means, covs):
            x = np.asarray(x[: g.shape[0]], dtype=np.float64)
            ll = np.stack(
                [gaussian_log_likelihood(x, m[k], C[k]) for k in range(g.shape[1])],
                axis=1,
            )
            log_likelihoods.append([np.sum(ll * g), g.shape[0]])
        return np.array(log_likelihoods)

    def _average_log_likelihood(
        self, data, temporal: Dict, spatial: Dict, embeddings, magnitudes
    ) -> float:
        log_likelihoods = self.session_log_likelihoods(
            data, temporal[self._time_courses], embeddings, spatial, magnitudes
        )
        return float(log_likelihoods[:, 0].sum() / log_likelihoods[:, 1].sum())

    def _predicted_magnitudes(self, temporal: Dict, spatial: Dict) -> Dict:
        return _predict_magnitudes(
            temporal["train_magnitudes"], spatial["magnitudes"], temporal["magnitudes"]
        )

    def log_likelihood(self, data, temporal: Dict, spatial: Dict) -> float:
        return self._average_log_likelihood(
            data,
            temporal,
            spatial,
            temporal["embeddings"],
            self._predicted_magnitudes(temporal, spatial),
        )

    def log_likelihood_controls(
        self, data, temporal: Dict, spatial: Dict
    ) -> Dict[str, float]:
        """Log-likelihood with zero deviations (:code:`"log_likelihood_group"`)
        and with the embeddings and magnitudes shuffled between the test
        sessions (:code:`"log_likelihood_shuffled"`, if there is more than
        one test session)."""
        embeddings = temporal["embeddings"]
        controls = {
            "log_likelihood_group": self._average_log_likelihood(
                data, temporal, spatial, embeddings, None
            )
        }
        n_sessions = embeddings.shape[0]
        if n_sessions > 1:
            # Each session gets the embedding and magnitudes of another
            order = np.roll(np.arange(n_sessions), 1)
            magnitudes = {
                key: value[order]
                for key, value in self._predicted_magnitudes(temporal, spatial).items()
            }
            controls["log_likelihood_shuffled"] = self._average_log_likelihood(
                data, temporal, spatial, embeddings[order], magnitudes
            )
        return controls

    def temporal_files(self, temporal: Dict) -> Dict[str, object]:
        return {
            "alp.pkl": temporal[self._time_courses],
            "embeddings.npy": temporal["embeddings"],
        }


class DIVEOps(HIVEOps):
    """Operations for :py:mod:`DIVE <osl_dynamics.models.dive>`.

    The parameters are treated as in :py:class:`HIVEOps`, with the mixing
    coefficients (:code:`alpha`) in place of the state probabilities. The
    inference network reads the data, so it is channel-specific too: step 3
    trains the temporal model (inference and model RNNs) on channels X of
    the test sessions with the observation model fixed, as
    :py:class:`DyNeMoOps` does. Step 3 does not use an initialization method,
    because it would overwrite the fixed group matrices.

    Step 2 maximizes DIVE's loss with :code:`alpha` fixed: the mean and
    covariance (or precision with :code:`mix_precisions=True`) at each time
    point mix the session means and matrices.

    Temporal parameters are a dict with the mixing coefficients
    (:code:`"alpha"`), the session :code:`"embeddings"` and the deviation
    :code:`"magnitudes"`. The controls are the same as for HIVE.

    The default :code:`config_kwargs` and :code:`init_kwargs` are the
    settings used to test bi-cross-validation of DIVE on simulated data.
    :code:`embeddings_dim` defaults to 5.
    """

    name = "dive"
    default_config_kwargs = {
        "sequence_length": 200,
        "inference_n_units": 128,
        "inference_normalization": "layer",
        "model_n_units": 128,
        "model_normalization": "layer",
        "learn_alpha_temperature": True,
        "initial_alpha_temperature": 1.0,
        "embeddings_dim": 5,
        "spatial_embeddings_dim": 2,
        "dev_n_layers": 5,
        "dev_n_units": 32,
        "dev_activation": "tanh",
        "dev_normalization": "layer",
        "dev_regularizer": "l1",
        "dev_regularizer_factor": 0.01,
        "batch_size": 32,
        "learning_rate": 0.005,
        "lr_decay": 0.1,
        "n_epochs": 40,
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 20,
    }
    default_init_kwargs = {"n_init": 5, "n_epochs": 3, "take": 1}

    _time_courses = "alpha"
    _precisions_option = "mix_precisions"
    _temporal_overrides = {}

    def _model(self, n_channels: int, n_sessions: int, data=None, **overrides):
        from osl_dynamics.models import dive

        config = dive.Config(
            **dict(
                self.config_kwargs,
                n_channels=n_channels,
                n_sessions=n_sessions,
                **overrides,
            )
        )
        model = dive.Model(config)
        if data is not None:
            model.set_regularizers(data)
        return model

    def _initialize(self, model, data) -> None:
        model.random_subset_initialization(data, **self.init_kwargs)

    def _fixed_time_course_loss(self, model):
        """Sub-model and loss for training the observation model with fixed
        mixing coefficients.

        This is DIVE's loss with the inference network replaced by the
        fixed mixing coefficients.
        """
        import tensorflow as tf
        import tensorflow_probability as tfp

        from osl_dynamics.inference.layers import precision_log_prob

        keras_model = model.model
        kl_layers = [f"{param}_dev_mag_kl_loss" for param in self._params(model)]
        sub_model = tf.keras.Model(
            inputs=keras_model.inputs,
            outputs=[
                keras_model.get_layer(name).output
                for name in ["session_means", "session_covs", *kl_layers]
            ],
        )
        precisions = self.precisions

        def loss(outputs, x, alpha):
            mu, D, *kl_losses = outputs
            m = tf.einsum("btk,bkm->btm", alpha, mu)
            C = tf.einsum("btk,bkij->btij", alpha, D)
            if precisions:
                ll = precision_log_prob(x, m, C)
            else:
                mvn = tfp.distributions.MultivariateNormalTriL(
                    loc=m,
                    scale_tril=tf.linalg.cholesky(C),
                    allow_nan_stats=False,
                )
                ll = mvn.log_prob(x)
            return -tf.reduce_mean(ll) + sum(tf.reduce_sum(kl) for kl in kl_losses)

        return sub_model, loss

    def session_log_likelihoods(
        self,
        data,
        time_courses: List[np.ndarray],
        embeddings: np.ndarray,
        spatial: Dict,
        magnitudes: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Log-likelihood of each session given its mixing coefficients,
        embedding and magnitudes (see :py:meth:`session_matrices`).

        Returns
        -------
        log_likelihoods : np.ndarray
            Log-likelihood summed over time points and the number of time
            points for each session. Shape is (n_sessions, 2).
        """
        alpha = _as_list(time_courses)
        ts = time_series(data)
        _check_n_sessions(ts, alpha)

        means, matrices = self.session_matrices(
            data.n_channels, spatial, embeddings, magnitudes
        )
        log_likelihood = (
            batched_precision_log_likelihood
            if self.precisions
            else batched_gaussian_log_likelihood
        )
        log_likelihoods = []
        for x, a, mu, D in zip(ts, alpha, means, matrices):
            x = np.asarray(x[: a.shape[0]], dtype=np.float64)
            a = np.asarray(a, dtype=np.float64)
            total = 0.0
            for chunk in _chunks(a.shape[0], x.shape[1]):
                m = a[chunk] @ mu
                M = np.einsum("tk,kij->tij", a[chunk], D)
                total += np.sum(log_likelihood(x[chunk], m, M))
            log_likelihoods.append([total, a.shape[0]])
        return np.array(log_likelihoods)


def _means_layer(n: int, m: int, learn: bool, initial_value: np.ndarray):
    from osl_dynamics.inference.layers import VectorsLayer

    return VectorsLayer(n, m, learn, initial_value.astype(np.float32))


_MODEL_OPS = {
    "hmm": HMMOps,
    "dynemo": DyNeMoOps,
    "mdynemo": MDyNeMoOps,
    "swc": SWCOps,
    "hive": HIVEOps,
    "dive": DIVEOps,
}


def get_model_ops(
    model: str,
    config_kwargs: dict,
    init_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
    spatial_method: Optional[str] = None,
    best_of: int = 1,
) -> ModelOps:
    """Get the operations for a model.

    Parameters
    ----------
    model : str
        Model name. Options are :code:`"hmm"`, :code:`"dynemo"`,
        :code:`"mdynemo"`, :code:`"swc"`, :code:`"hive"` and :code:`"dive"`.
    config_kwargs : dict
        Keyword arguments for the model's :code:`Config`.
    init_kwargs : dict, optional
        Keyword arguments for the model's initialization method.
    fit_kwargs : dict, optional
        Keyword arguments for the model's :code:`fit` method.
    spatial_method : str, optional
        How to estimate the spatial parameters given the time courses.
    best_of : int, optional
        Number of times to train the model in :code:`fit`.

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
    return _MODEL_OPS[model](
        config_kwargs, init_kwargs, fit_kwargs, spatial_method, best_of
    )
