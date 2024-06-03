"""Sliding Window Correlation (SWC) with a Multivariate Normal observation model."""

import logging
import os
import os.path as op
import sys
import warnings
from dataclasses import dataclass
from typing import Union
from pathlib import Path

import numba
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend, layers, utils
from numba.core.errors import NumbaWarning
from scipy.special import logsumexp, xlogy
from tqdm.auto import trange
from pqdm.threads import pqdm

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference import initializers
from osl_dynamics.inference.layers import (
    CategoricalLogLikelihoodLossLayer,
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    VectorsLayer,
    StaticLossScalingFactorLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig, ModelBase
from osl_dynamics.simulation import HMM
from osl_dynamics.utils.misc import set_logging_level
from osl_dynamics.inference.modes import argmax_time_courses
from osl_dynamics.analysis import connectivity


_logger = logging.getLogger("osl-dynamics")

warnings.filterwarnings("ignore", category=NumbaWarning)

EPS = sys.float_info.epsilon

@dataclass
class Config(BaseModelConfig):
    """Settings for SWC.

    Parameters
    ----------
    model_name : str
        Model name.
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    window_length: int
        Length of the sliding window
    window_offset: int
        Offset of window each time
    window_type: str
        The type of window. Either 'rectangular' or 'tapered'

    learn_means : bool
        Should we make the mean vectors for each state trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each staet trainable?
    diagonal_covariances : bool
        Should we learn diagonal covariances?
    covariances_epsilon : float
        Error added to state covariances for numerical stability.
    """

    model_name: str = "SWC"

    # Observation model parameters
    window_type: str = 'rectanglular'
    learn_means: bool = None
    learn_covariances: bool = None
    diagonal_covariances: bool = False
    covariances_epsilon: float = None


    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_dimension_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0

        if self.window_type not in ['rectangular','taperped']:
            raise ValueError('Invalid window type!')

    def validate_dimension_parameters(self):
        if self.n_modes is None and self.n_states is None:
            raise ValueError("Either n_modes or n_states must be passed.")

        if self.n_modes is not None:
            if self.n_modes < 1:
                raise ValueError("n_modes must be one or greater.")

        if self.n_states is not None:
            if self.n_states < 1:
                raise ValueError("n_states must be one or greater.")

        if self.n_channels is None:
            raise ValueError("n_channels must be passed.")
        elif self.n_channels < 1:
            raise ValueError("n_channels must be one or greater.")

class Model(ModelBase):
    """SWC class.

    Parameters
    ----------
    config : osl_dynamics.models.swc.Config
    """

    config_type = Config

    def build_model(self):
        pass

    def fit(self, dataset, verbose=1, **kwargs):
        ts = dataset.time_series()
        ts = np.array([ts[i] for i in dataset.keep])
        swc = connectivity.sliding_window_connectivity(ts, window_length=self.config.window_length, step_size=self.config.window_offset,
                                                       conn_type="corr")

        kmeans = KMeans(n_clusters=self.config.n_states, verbose=0)

        # Get indices that correspond to an upper triangle of a matrix
        # (not including the diagonal)
        i, j = np.triu_indices(self.config.n_channels, k=1)

        # Now let's convert the sliding window connectivity matrices to a series of vectors
        swc_vectors = swc[:, i, j]

        # Check the shape of swc vectors
        print(f'SWC vectors shape: {swc_vectors.shape}')

        # Fitting
        kmeans.fit(swc_vectors)  # should take a few seconds

        centroids = kmeans.cluster_centers_