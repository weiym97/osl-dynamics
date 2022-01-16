"""Data class for model settings.

"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy


@dataclass
class Config:
    """Settings for a model in DyNeMo.

    Dimension Parameters
    --------------------
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    Inference Network Parameters
    ----------------------------
    inference_rnn : str
        RNN to use, either 'gru' or 'lstm'.
    inference_n_layers : int
        Number of layers.
    inference_n_units : int
        Number of units.
    inference_normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    inference_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. 'relu', 'elu', etc.
    inference_dropout_rate : float
        Dropout rate.

    Model Network Parameters
    ------------------------
    model_rnn : str
        RNN to use, either 'gru' or 'lstm'.
    model_n_layers : int
        Number of layers.
    model_n_units : int
        Number of units.
    model_normalization : str
        Type of normalization to use. Either None, 'batch' or 'layer'.
    model_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. 'relu', 'elu', etc.
    model_dropout_rate : float
        Dropout rate.

    Multi-Time-Scale Model Parameters
    ---------------------------------
    multiple_scales: bool
        Should we use the multi-scale model?
    fix_std: bool
        Should we have constant std across modes and time?
    tie_mean_std: bool
        Should we tie up the time courses of mean and std?
    learn_stds: bool
        Should we make the standard deviation for each mode trainable?
    learn_fcs: bool
        Should we make the functional connectivity for each mode trainable?
    initial_stds: np.ndarray
        Initialisation for mode standard deviations.
    initial_fcs: np.ndarray
        Initialisation for mode functional connectivity matrices.
    regularize_fcs : bool
        Should we regularize the cholesky factor of the functional connectivity
        matrices? An L2 norm is used. Optional.

    Alpha Parameters
    ----------------
    theta_normalization : str
        Type of normalization to apply to the posterior samples, theta.
        Either 'layer', 'batch' or None.
    alpha_xform : str
        Functional form of alpha. Either 'gumbel-softmax', 'softmax' or 'softplus'.
    initial_alpha_temperature : float
        Initial value for the alpha temperature if it is being learnt or if
        we are performing alpha temperature annealing
    learn_alpha_temperature : bool
        Should we learn the alpha temperature when alpha_xform = 'softmax' or
        'gumbel-softmax'?
    do_alpha_temperature_annealing : bool
        Should we perform alpha temperature annealing. Can be used when
        alpha_xform = 'softmax' or 'gumbel-softmax'.
    final_alpha_temperature : bool
        Final value for the alpha temperature if we are annealing.
    n_alpha_temperature_annealing_epochs : int
        Number of alpha temperature annealing epochs.

    Observation Model Parameters
    ----------------------------
    observation_model : str
        Type of observation model. Either 'multivariate_normal' or 'wavenet'.
    wavenet_n_filters : int
        Number of filters in the each convolutional layer.
    wavenet_n_layers : int
        Number of dilated causal convolution layers.
    learn_means : bool
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    learn_alpha_scaling : bool
        Should we learn a scaling for alpha?
    normalize_covariances : bool
        Should we trace normalize the mode covariances?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for mode covariances.
    diag_covs : bool
        Should we learn diagonal covariances?
        Only used if observation_model='wavenet'.

    KL Annealing Parameters
    -----------------------
    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either 'linear' or 'tanh'.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        kl_annealing_curve='tanh'.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.
    n_kl_annealing_cycles : int
        Number of times to perform KL annealing within n_kl_annealing_epochs.

    Initialization Parameters
    -------------------------
    n_init : int
        Number of initializations.
    n_init_epochs : int
        Number of epochs to train each initialization.

    Training Parameters
    -------------------
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    gradient_clip : float
        Value to clip gradients by. This is the clipnorm argument passed to
        the Keras optimizer. Cannot be used if multi_gpu=True.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tensorflow.keras.optimizers.Optimizer
        Optimizer to use. 'adam' is recommended.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    """

    # Dimension parameters
    n_modes: int = None
    n_channels: int = None
    sequence_length: int = None

    # Inference network parameters
    inference_rnn: Literal["gru", "lstm"] = None
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: Literal[None, "batch", "layer"] = None
    inference_activation: str = None
    inference_dropout_rate: float = 0.0

    # Model network parameters
    model_rnn: Literal["gru", "lstm"] = None
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: Literal[None, "batch", "layer"] = None
    model_activation: str = None
    model_dropout_rate: float = 0.0

    # Multi-time-scale model parameters
    multiple_scales: bool = False
    fix_std: bool = False
    tie_mean_std: bool = False
    learn_stds: bool = None
    learn_fcs: bool = None
    initial_stds: np.ndarray = None
    initial_fcs: np.ndarray = None
    regularize_fcs: bool = False

    # Alpha parameters
    theta_normalization: Literal[None, "batch", "layer"] = None
    alpha_xform: Literal["gumbel-softmax", "softmax", "softplus"] = None
    learn_alpha_temperature: bool = None
    initial_alpha_temperature: float = None
    do_alpha_temperature_annealing: bool = None
    final_alpha_temperature: float = None
    n_alpha_temperature_annealing_epochs: int = None

    # Observation model parameters
    observation_model: Literal["multivariate_normal", "wavenet"] = "multivariate_normal"
    wavenet_n_filters: int = None
    wavenet_n_layers: int = None
    learn_means: bool = None
    learn_covariances: bool = None
    learn_alpha_scaling: bool = False
    normalize_covariances: bool = False
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    diag_covs: bool = False

    # KL annealing parameters
    do_kl_annealing: bool = None
    kl_annealing_curve: Literal["linear", "tanh"] = None
    kl_annealing_sharpness: float = None
    n_kl_annealing_epochs: int = None
    n_kl_annealing_cycles: int = 1

    # Initialisation parameters
    n_init: int = None
    n_init_epochs: int = None

    # Training parameters
    batch_size: int = None
    learning_rate: float = None
    gradient_clip: float = None
    n_epochs: int = None
    optimizer: tensorflow.keras.optimizers.Optimizer = "adam"
    multi_gpu: bool = False
    strategy: str = None

    def __post_init__(self):
        self.validate_model_choice_parameters()
        self.validate_dimension_parameters()
        self.validate_rnn_parameters()
        self.validate_alpha_parameters()
        self.validate_observation_model_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_initialization_parameters()
        self.validate_training_parameters()

    def validate_model_choice_parameters(self):

        if self.observation_model not in ["multivariate_normal", "wavenet"]:
            raise ValueError(
                "observation_model must be 'multivariate_normal' or 'wavenet'."
            )

    def validate_rnn_parameters(self):

        if self.inference_rnn is None and (
            self.model_rnn is not None
            or self.alpha_xform is not None
            or self.do_kl_annealing is not None
        ):
            raise ValueError("Please pass inference_rnn.")

        if self.model_rnn is None and (
            self.inference_rnn is not None
            or self.alpha_xform is not None
            or self.do_kl_annealing is not None
        ):
            raise ValueError("Please pass model_rnn.")

    def validate_alpha_parameters(self):
        if self.inference_rnn is None:
            return

        if self.alpha_xform not in ["gumbel-softmax", "softmax", "softplus"]:
            raise ValueError(
                "alpha_xform must be 'gumbel-softmax', 'softmax' or 'softplus'."
            )

        if "softmax" in self.alpha_xform:
            if (
                self.learn_alpha_temperature is None
                and self.do_alpha_temperature_annealing is None
            ):
                raise ValueError(
                    "Either learn_alpha_temperature or do_alpha_temperature_annealing "
                    + "must be passed."
                )

            if self.initial_alpha_temperature is None:
                raise ValueError("initial_alpha_temperature must be passed.")

            if self.initial_alpha_temperature <= 0:
                raise ValueError("initial_alpha_temperature must be greater than zero.")

            if self.learn_alpha_temperature:
                self.do_alpha_temperature_annealing = False

            if self.do_alpha_temperature_annealing:
                self.learn_alpha_temperature = False

                if self.final_alpha_temperature is None:
                    raise ValueError(
                        "If we are performing alpha temperature annealing, "
                        + "final_alpha_temperature must be passed."
                    )

                if self.final_alpha_temperature <= 0:
                    raise ValueError(
                        "final_alpha_temperature must be greater than zero."
                    )

                if self.n_alpha_temperature_annealing_epochs is None:
                    raise ValueError(
                        "If we are performing alpha temperature annealing, "
                        + "n_alpha_temperature_annealing_epochs must be passed."
                    )

                if self.n_alpha_temperature_annealing_epochs < 1:
                    raise ValueError(
                        "n_alpha_temperature_annealing_epochs must be one or above."
                    )

        elif self.alpha_xform == "softplus":
            self.initial_alpha_temperature = 1.0  # not used in the model
            self.learn_alpha_temperature = False
            self.do_alpha_temperature_annealing = False

    def validate_dimension_parameters(self):

        if self.sequence_length is None:
            raise ValueError("sequence_length must be passed.")

        if self.n_modes is not None:
            if self.n_modes < 1:
                raise ValueError("n_modes must be one or greater.")

        if self.n_channels is not None:
            if self.n_channels < 1:
                raise ValueError("n_channels must be one or greater.")

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be one or greater.")

    def validate_kl_annealing_parameters(self):
        if self.inference_rnn is None:
            return

        if self.do_kl_annealing is None:
            raise ValueError("do_kl_annealing must be passed.")

        if self.do_kl_annealing:
            if self.kl_annealing_curve is None:
                raise ValueError(
                    "If we are performing KL annealing, "
                    "kl_annealing_curve must be passed."
                )

            if self.kl_annealing_curve not in ["linear", "tanh"]:
                raise ValueError("KL annealing curve must be 'linear' or 'tanh'.")

            if self.kl_annealing_curve == "tanh":
                if self.kl_annealing_sharpness is None:
                    raise ValueError(
                        "kl_annealing_sharpness must be passed if "
                        + "kl_annealing_curve='tanh'."
                    )

                if self.kl_annealing_sharpness < 0:
                    raise ValueError("KL annealing sharpness must be positive.")

            if self.n_kl_annealing_epochs is None:
                raise ValueError(
                    "If we are performing KL annealing, n_kl_annealing_epochs must be "
                    + "passed."
                )

            if self.n_kl_annealing_epochs < 1:
                raise ValueError(
                    "Number of KL annealing epochs must be greater than zero."
                )

            if self.n_kl_annealing_cycles < 1:
                raise ValueError("n_kl_annealing_cycles must be one or greater.")

    def validate_observation_model_parameters(self):

        if self.observation_model == "multivariate_normal":
            if self.multiple_scales:
                if (
                    self.learn_means is None
                    or self.learn_stds is None
                    or self.learn_fcs is None
                ):
                    raise ValueError(
                        "learn_means, learn_stds and learn_fcs must be passed."
                    )

            else:
                if self.learn_means is None or self.learn_covariances is None:
                    raise ValueError(
                        "learn_means and learn_covariances must be passed."
                    )

        if self.observation_model == "wavenet":
            if self.wavenet_n_filters is None or self.wavenet_n_layers is None:
                raise ValueError(
                    "wavenet_n_filters and wavenet_n_layers must be passed."
                )
            if self.learn_covariances is None:
                self.learn_covariances = True

    def validate_initialization_parameters(self):

        if self.n_init is not None:
            if self.n_init < 1:
                raise ValueError("n_init must be one or greater.")

            if self.n_init_epochs is None:
                raise ValueError(
                    "If we are initializing the model, n_init_epochs "
                    + "must be passed."
                )

            if self.n_init_epochs < 1:
                raise ValueError("n_init_epochs must be one or greater.")

    def validate_training_parameters(self):

        if self.batch_size is None:
            raise ValueError("batch_size must be passed.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be one or greater.")

        if self.n_epochs is None:
            raise ValueError("n_epochs must be passed.")

        if self.n_epochs < 1:
            raise ValueError("n_epochs must be one or greater.")

        if self.learning_rate is None:
            raise ValueError("learning_rate must be passed.")

        if self.learning_rate < 0:
            raise ValueError("learning_rate must be greater than zero.")

        if self.optimizer not in ["adam", "Adam", "rmsprop", "RMSprop"]:
            raise NotImplementedError("Please use optimizer='adam' or 'rmsprop'.")

        # Strategy for distributed learning
        if self.multi_gpu:
            self.strategy = MirroredStrategy()
        elif self.strategy is None:
            self.strategy = get_strategy()
