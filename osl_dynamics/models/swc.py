"""Sliding Window Correlation (SWC) model.

SWC estimates a covariance matrix in sliding windows and clusters these
matrices with k-means. Each cluster centroid is a state covariance and each
window is assigned to the nearest centroid. The observation model is a
zero-mean multivariate normal.

Unlike the other models in osl-dynamics, SWC is not trained with TensorFlow.
It provides a similar interface (:code:`Config`, :code:`Model`, :code:`fit`,
:code:`get_alpha`, :code:`dual_estimation`, :code:`save`, :code:`load`) so it
can be used as a baseline for the HMM and DyNeMo, e.g. in
:py:mod:`osl_dynamics.evaluate`.

Note, the state time course of SWC is defined per window rather than per
time point.

See Also
--------
- :doc:`Sliding Window Analysis </tutorials_build/3-1_sliding_window_analysis>`
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import yaml
from scipy.linalg import solve_triangular
from sklearn.cluster import KMeans

import osl_dynamics
from osl_dynamics.utils.misc import NumpyLoader

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config:
    """Settings for SWC.

    Parameters
    ----------
    model_name : str
        Model name.
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    window_length : int
        Number of samples in each window.
    window_offset : int
        Number of samples between the start of consecutive windows.
    learn_means : bool
        Must be :code:`False`. The state means are fixed to
        :code:`initial_means` (zero by default).
    learn_covariances : bool
        Should :code:`fit` estimate the state covariances with k-means?
        If :code:`False`, :code:`fit` assigns windows to the fixed
        :code:`initial_covariances`.
    initial_means : np.ndarray, optional
        State means. Shape must be (n_states, n_channels).
        Defaults to zeros.
    initial_covariances : np.ndarray, optional
        State covariances. Shape must be (n_states, n_channels, n_channels).
        Defaults to identity matrices. Must be passed if
        :code:`learn_covariances=False`.
    kmeans_n_init : int, optional
        Number of k-means initializations. Passed as :code:`n_init` to
        :code:`sklearn.cluster.KMeans`.
    random_state : int, optional
        Seed for k-means.
    """

    model_name: str = "SWC"

    # Dimension parameters
    n_states: int = None
    n_channels: int = None

    # Window parameters
    window_length: int = None
    window_offset: int = None

    # Observation model parameters
    learn_means: bool = False
    learn_covariances: bool = True
    initial_means: Optional[np.ndarray] = None
    initial_covariances: Optional[np.ndarray] = None

    # K-means parameters
    kmeans_n_init: int = 10
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        self.validate_dimension_parameters()
        self.validate_window_parameters()
        self.validate_observation_model_parameters()

    def validate_dimension_parameters(self) -> None:
        if self.n_states is None:
            raise ValueError("n_states must be passed.")
        elif self.n_states < 1:
            raise ValueError("n_states must be one or greater.")

        if self.n_channels is None:
            raise ValueError("n_channels must be passed.")
        elif self.n_channels < 1:
            raise ValueError("n_channels must be one or greater.")

    def validate_window_parameters(self) -> None:
        if self.window_length is None:
            raise ValueError("window_length must be passed.")
        elif self.window_length < 2:
            raise ValueError("window_length must be two or greater.")

        if self.window_offset is None:
            raise ValueError("window_offset must be passed.")
        elif self.window_offset < 1:
            raise ValueError("window_offset must be one or greater.")

    def validate_observation_model_parameters(self) -> None:
        if self.learn_means:
            raise ValueError("SWC does not learn the means, use learn_means=False.")

        if not self.learn_covariances and self.initial_covariances is None:
            raise ValueError(
                "initial_covariances must be passed if learn_covariances=False."
            )

        if self.initial_means is not None:
            self.initial_means = np.array(self.initial_means, dtype=np.float64)
            if self.initial_means.shape != (self.n_states, self.n_channels):
                raise ValueError(
                    "initial_means must have shape (n_states, n_channels)."
                )

        if self.initial_covariances is not None:
            self.initial_covariances = np.array(
                self.initial_covariances, dtype=np.float64
            )
            if self.initial_covariances.shape != (
                self.n_states,
                self.n_channels,
                self.n_channels,
            ):
                raise ValueError(
                    "initial_covariances must have shape "
                    "(n_states, n_channels, n_channels)."
                )


class Model:
    """Sliding Window Correlation model.

    Parameters
    ----------
    config : osl_dynamics.models.swc.Config
    """

    config_type = Config

    def __init__(self, config: Config) -> None:
        self.config = config

        if config.initial_means is None:
            self.means = np.zeros([config.n_states, config.n_channels])
        else:
            self.means = config.initial_means.copy()

        if config.initial_covariances is None:
            self.covariances = np.stack([np.eye(config.n_channels)] * config.n_states)
        else:
            self.covariances = config.initial_covariances.copy()

    def fit(self, data) -> List[np.ndarray]:
        """Fit the model.

        If :code:`config.learn_covariances=True`, the sliding window
        covariances are clustered with k-means and the cluster centroids are
        used as the state covariances. Otherwise, the windows are assigned to
        the current state covariances.

        Parameters
        ----------
        data : osl_dynamics.data.Data or list of np.ndarray or np.ndarray
            Training data. If a :code:`Data` object is passed, only the
            sessions in :code:`data.keep` are used.

        Returns
        -------
        alpha : list of np.ndarray
            State assigned to each window. Each array has shape (n_windows,).
        """
        if not self.config.learn_covariances:
            return self.get_alpha(data, concatenate=False)

        n_channels = self.config.n_channels
        swc = self._sliding_window_covariances(data)

        # Cluster the upper triangle (including the diagonal) of each window
        i, j = np.triu_indices(n_channels)
        vectors = np.concatenate(swc, axis=0)[:, i, j]
        kmeans = KMeans(
            n_clusters=self.config.n_states,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.random_state,
        ).fit(vectors)

        # Convert the centroids into covariance matrices
        covariances = np.empty([self.config.n_states, n_channels, n_channels])
        covariances[:, i, j] = kmeans.cluster_centers_
        covariances[:, j, i] = kmeans.cluster_centers_
        self.covariances = covariances

        n_windows = [s.shape[0] for s in swc]
        return np.split(kmeans.labels_, np.cumsum(n_windows)[:-1])

    def get_alpha(
        self, data, concatenate: bool = False
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Assign each window to the nearest state covariance.

        The distance is the Euclidean distance between the upper triangle
        (including the diagonal) of the window covariance and the state
        covariance.

        Parameters
        ----------
        data : osl_dynamics.data.Data or list of np.ndarray or np.ndarray
            Data to assign states to.
        concatenate : bool, optional
            Should we concatenate the state time courses of each session?

        Returns
        -------
        alpha : list of np.ndarray or np.ndarray
            State assigned to each window. Shape is (n_sessions, n_windows)
            or (n_windows,).
        """
        i, j = np.triu_indices(self.config.n_channels)
        centroids = self.covariances[:, i, j]

        alpha = []
        for x in self._get_time_series(data):
            labels = []
            for covs in self._iterate_window_covariances(x):
                vectors = covs[:, i, j]
                distances = np.sum(
                    (vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
                )
                labels.append(np.argmin(distances, axis=1))
            alpha.append(np.concatenate(labels))

        if concatenate or len(alpha) == 1:
            alpha = np.concatenate(alpha)

        return alpha

    def dual_estimation(
        self, data, alpha: Union[List[np.ndarray], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate the state covariances given the state time course.

        Each state covariance is the average of the window covariances
        assigned to the state. The windows of all sessions are pooled.

        Parameters
        ----------
        data : osl_dynamics.data.Data or list of np.ndarray or np.ndarray
            Data to estimate the state covariances from.
        alpha : list of np.ndarray or np.ndarray
            State assigned to each window. Shape must be
            (n_sessions, n_windows) or (n_windows,).

        Returns
        -------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        swc = self._sliding_window_covariances(data)
        alpha = self._validate_alpha(alpha, [s.shape[0] for s in swc])

        swc = np.concatenate(swc, axis=0)
        alpha = np.concatenate(alpha)

        n_channels = self.config.n_channels
        covariances = np.empty([self.config.n_states, n_channels, n_channels])
        for state in range(self.config.n_states):
            windows = swc[alpha == state]
            if windows.shape[0] == 0:
                _logger.warning(
                    f"No windows assigned to state {state}. "
                    "Using an identity covariance."
                )
                covariances[state] = np.eye(n_channels)
            else:
                covariances[state] = np.mean(windows, axis=0)

        return self.means.copy(), covariances

    def log_likelihood(self, data, alpha: Union[List[np.ndarray], np.ndarray]) -> float:
        """Average log-likelihood of the data given the state time course.

        For each window, we calculate the average log-likelihood of its time
        points under the multivariate normal of the state assigned to the
        window. We return the average over all windows.

        Parameters
        ----------
        data : osl_dynamics.data.Data or list of np.ndarray or np.ndarray
            Data to calculate the log-likelihood for.
        alpha : list of np.ndarray or np.ndarray
            State assigned to each window. Shape must be
            (n_sessions, n_windows) or (n_windows,).

        Returns
        -------
        log_likelihood : float
            Average log-likelihood per window.
        """
        ts = self._get_time_series(data)
        window_length = self.config.window_length
        window_offset = self.config.window_offset
        alpha = self._validate_alpha(alpha, [self._n_windows(x) for x in ts])

        # Pre-compute the Cholesky factors of the state covariances
        cholesky = np.linalg.cholesky(self.covariances)
        log_det = 2 * np.sum(np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1)
        constant = self.config.n_channels * np.log(2 * np.pi)

        total = 0.0
        n_windows = 0
        for x, a in zip(ts, alpha):
            # Log-likelihood of each time point under each state
            ll = np.empty([x.shape[0], self.config.n_states])
            for state in range(self.config.n_states):
                z = solve_triangular(
                    cholesky[state], (x - self.means[state]).T, lower=True
                )
                ll[:, state] = -0.5 * (constant + log_det[state] + np.sum(z**2, axis=0))

            # Sum over the time points in each window
            cumsum = np.concatenate([np.zeros([1, ll.shape[1]]), np.cumsum(ll, axis=0)])
            starts = np.arange(a.shape[0]) * window_offset
            window_sums = cumsum[starts + window_length, a] - cumsum[starts, a]

            total += np.sum(window_sums / window_length)
            n_windows += a.shape[0]

        return float(total / n_windows)

    def get_means(self) -> np.ndarray:
        """Get the state means.

        Returns
        -------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        """
        return self.means.copy()

    def get_covariances(self) -> np.ndarray:
        """Get the state covariances.

        Returns
        -------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        return self.covariances.copy()

    def get_means_covariances(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the state means and covariances.

        Returns
        -------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        return self.get_means(), self.get_covariances()

    def set_means(self, means: np.ndarray) -> None:
        """Set the state means.

        Parameters
        ----------
        means : np.ndarray
            State means. Shape must be (n_states, n_channels).
        """
        means = np.array(means, dtype=np.float64)
        if means.shape != self.means.shape:
            raise ValueError(f"means must have shape {self.means.shape}.")
        self.means = means

    def set_covariances(self, covariances: np.ndarray) -> None:
        """Set the state covariances.

        Parameters
        ----------
        covariances : np.ndarray
            State covariances. Shape must be (n_states, n_channels, n_channels).
        """
        covariances = np.array(covariances, dtype=np.float64)
        if covariances.shape != self.covariances.shape:
            raise ValueError(f"covariances must have shape {self.covariances.shape}.")
        self.covariances = covariances

    def set_means_covariances(self, means: np.ndarray, covariances: np.ndarray) -> None:
        """Set the state means and covariances.

        Parameters
        ----------
        means : np.ndarray
            State means. Shape must be (n_states, n_channels).
        covariances : np.ndarray
            State covariances. Shape must be (n_states, n_channels, n_channels).
        """
        self.set_means(means)
        self.set_covariances(covariances)

    def save(self, dirname: str) -> None:
        """Save the config and the state means and covariances.

        Parameters
        ----------
        dirname : str
            Directory to save :code:`config.yml`, :code:`means.npy` and
            :code:`covs.npy` to.
        """
        os.makedirs(dirname, exist_ok=True)

        config_dict = asdict(self.config)
        config_dict["initial_means"] = None
        config_dict["initial_covariances"] = None

        with open(f"{dirname}/config.yml", "w") as file:
            file.write(f"# osl-dynamics version: {osl_dynamics.__version__}\n")
            yaml.dump(config_dict, file)

        np.save(f"{dirname}/means.npy", self.means)
        np.save(f"{dirname}/covs.npy", self.covariances)

    @classmethod
    def load(cls, dirname: str, single_gpu: bool = True) -> "Model":
        """Load a model saved with :code:`save`.

        Parameters
        ----------
        dirname : str
            Directory containing :code:`config.yml`, :code:`means.npy` and
            :code:`covs.npy`.
        single_gpu : bool, optional
            Not used. Accepted for compatibility with
            :code:`osl_dynamics.models.load`.

        Returns
        -------
        model : osl_dynamics.models.swc.Model
            Model object.
        """
        _logger.info(f"Loading model: {dirname}")
        with open(f"{dirname}/config.yml", "r") as file:
            config_dict = yaml.load(file, NumpyLoader)
        config_dict["initial_means"] = np.load(f"{dirname}/means.npy")
        config_dict["initial_covariances"] = np.load(f"{dirname}/covs.npy")
        return cls(cls.config_type(**config_dict))

    def _n_windows(self, x: np.ndarray) -> int:
        """Number of windows in a time series."""
        return (x.shape[0] - self.config.window_length) // self.config.window_offset + 1

    def _get_time_series(self, data) -> List[np.ndarray]:
        """Get a list of time series (one per session) from the data."""
        if hasattr(data, "time_series") and hasattr(data, "keep"):
            # osl_dynamics.data.Data object
            ts = data.time_series(prepared=True, concatenate=False)
            if isinstance(ts, np.ndarray):
                ts = [ts]
            ts = [ts[i] for i in data.keep]
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            ts = [data]
        else:
            ts = list(data)

        for x in ts:
            if x.shape[-1] != self.config.n_channels:
                raise ValueError(
                    f"data has {x.shape[-1]} channels, but the model has "
                    f"n_channels={self.config.n_channels}."
                )
            if x.shape[0] < self.config.window_length:
                raise ValueError("Each session must be at least one window long.")

        return ts

    def _sliding_window_covariances(self, data) -> List[np.ndarray]:
        """Sliding window covariances for each session."""
        from osl_dynamics.analysis.connectivity import sliding_window_connectivity

        ts = self._get_time_series(data)
        swc = sliding_window_connectivity(
            ts,
            window_length=self.config.window_length,
            step_size=self.config.window_offset,
            conn_type="cov",
        )
        if isinstance(swc, np.ndarray):
            swc = [swc]
        return swc

    def _iterate_window_covariances(self, x: np.ndarray, chunk_size: int = 1000):
        """Yield the covariance of each window in chunks of windows."""
        window_length = self.config.window_length
        window_offset = self.config.window_offset
        n_windows = self._n_windows(x)
        x = np.asarray(x, dtype=np.float64)

        for start in range(0, n_windows, chunk_size):
            stop = min(start + chunk_size, n_windows)
            windows = np.stack(
                [
                    x[w * window_offset : w * window_offset + window_length]
                    for w in range(start, stop)
                ]
            )
            windows = windows - np.mean(windows, axis=1, keepdims=True)
            yield np.einsum("wti,wtj->wij", windows, windows) / (window_length - 1)

    def _validate_alpha(
        self, alpha: Union[List[np.ndarray], np.ndarray], n_windows: List[int]
    ) -> List[np.ndarray]:
        """Check alpha has one label per window for each session."""
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            alpha = [alpha]
        alpha = [np.asarray(a, dtype=int) for a in alpha]

        if len(alpha) != len(n_windows):
            raise ValueError(
                f"alpha has {len(alpha)} sessions, but the data has "
                f"{len(n_windows)} sessions."
            )
        for i, (a, n) in enumerate(zip(alpha, n_windows)):
            if a.shape[0] != n:
                raise ValueError(
                    f"alpha[{i}] has {a.shape[0]} windows, but the data has {n}."
                )

        return alpha
