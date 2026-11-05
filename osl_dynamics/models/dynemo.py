"""Dynamic Network Modes (DyNeMo).

See the `documentation <https://osl-dynamics.readthedocs.io/en/latest/models\
/dynemo.html>`_ for a description of this model.
"""

import os
import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tqdm.auto import trange

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.layers import (
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    InferenceRNNLayer,
    KLDivergenceLayer,
    KLLossLayer,
    LogLikelihoodLossLayer,
    MixMatricesLayer,
    MixVectorsLayer,
    ModelRNNLayer,
    SampleNormalDistributionLayer,
    SoftmaxLayer,
    VectorsLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.inf_mod_base import (
    VariationalInferenceModelBase,
    VariationalInferenceModelConfig,
)
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.utils.misc import set_logging_level

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, VariationalInferenceModelConfig):
    """Settings for DyNeMo.

    Parameters
    ----------
    model_name : str
        Model name.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    inference_rnn : str
        RNN to use, either :code:`'gru'` or :code:`'lstm'`.
    inference_n_layers : int
        Number of layers.
    inference_n_units : int
        Number of units.
    inference_normalization : str
        Type of normalization to use. Either :code:`None`, :code:`'batch'` or
        :code:`'layer'`.
    inference_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. :code:`'relu'`, :code:`'elu'`, etc.
    inference_dropout : float
        Dropout rate.
    inference_regularizer : str
        Regularizer.

    model_rnn : str
        RNN to use, either :code:`'gru'` or :code:`'lstm'`.
    model_n_layers : int
        Number of layers.
    model_n_units : int
        Number of units.
    model_normalization : str
        Type of normalization to use. Either None, :code:`'batch'` or
        :code:`'layer'`.
    model_activation : str
        Type of activation to use after normalization and before dropout.
        E.g. :code:`'relu'`, :code:`'elu'`, etc.
    model_dropout : float
        Dropout rate.
    model_regularizer : str
        Regularizer.

    learn_alpha_temperature : bool
        Should we learn :code:`alpha_temperature`?
    initial_alpha_temperature : float
        Initial value for :code:`alpha_temperature`.

    learn_means : bool
        Should we make the mean vectors for each mode trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each mode trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for mode covariances.
        If :code:`diagonal_covariances=True` and full matrices are passed,
        the diagonal is extracted.
    covariances_epsilon : float
        Error added to mode covariances for numerical stability.
    diagonal_covariances : bool
        Should we learn diagonal mode covariances?
    mix_precisions : bool
        Should we mix the mode precision matrices instead of the covariances?
        If :code:`True`, the precision matrix at each time point is
        :code:`P_t = Sum_j alpha_jt P_j`, where :code:`P_j` is the inverse
        of the covariance of mode :code:`j`. Mode covariances are still passed
        and returned (e.g. :code:`initial_covariances`,
        :code:`get_covariances`), but :code:`covariances_epsilon` is added to
        the precisions. Defaults to :code:`False`.
    second_order_alpha_gradient : bool
        PROFUMO only (:code:`fit_and_get_alpha` with :code:`sigmas`): should
        the second-order term :code:`0.5 tr(P_t Sigma_t)` also train the
        inference network through alpha? If :code:`False` (default), alpha
        is held fixed in it and only the mode matrices learn from it.
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for mean vectors.
    covariances_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for covariance matrices.

    do_kl_annealing : bool
        Should we use KL annealing during training?
    kl_annealing_curve : str
        Type of KL annealing curve. Either :code:`'linear'` or :code:`'tanh'`.
    kl_annealing_sharpness : float
        Parameter to control the shape of the annealing curve if
        :code:`kl_annealing_curve='tanh'`.
    n_kl_annealing_epochs : int
        Number of epochs to perform KL annealing.

    init_method : str
        Initialization method to use. Defaults to 'random_subset'.
    n_init : int
        Number of initializations. Defaults to 5.
    n_init_epochs : int
        Number of epochs for each initialization. Defaults to 2.
    init_take : float
        Fraction of dataset to use in the initialization.
        Defaults to 1.0.

    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate.
    lr_decay : float
        Decay for learning rate. Default is 0.1. We use
        :code:`lr = learning_rate * exp(-lr_decay * epoch)`.
    gradient_clip : float
        Value to clip gradients by. This is the :code:`clipnorm` argument
        passed to the Keras optimizer. Cannot be used if :code:`multi_gpu=True`.
    n_epochs : int
        Number of training epochs.
    optimizer : str or tf.keras.optimizers.Optimizer
        Optimizer to use. :code:`'adam'` is recommended.
    loss_calc : str
        How should we collapse the time dimension in the loss?
        Either :code:`'mean'` or :code:`'sum'`.
    multi_gpu : bool
        Should be use multiple GPUs for training?
    strategy : str
        Strategy for distributed learning.
    best_of : int
        Number of full training runs to perform. A single run includes
        its own initialization and fitting from scratch.
    """

    model_name: str = "DyNeMo"

    # Inference network parameters
    inference_rnn: str = "lstm"
    inference_n_layers: int = 1
    inference_n_units: int = None
    inference_normalization: str = None
    inference_activation: str = None
    inference_dropout: float = 0.0
    inference_regularizer: str = None

    # Model network parameters
    model_rnn: str = "lstm"
    model_n_layers: int = 1
    model_n_units: int = None
    model_normalization: str = None
    model_activation: str = None
    model_dropout: float = 0.0
    model_regularizer: str = None

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    diagonal_covariances: bool = False
    mix_precisions: bool = False
    second_order_alpha_gradient: bool = False
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    # Initialization
    init_method: str = "random_subset"
    n_init: int = 5
    n_init_epochs: int = 2
    init_take: float = 1.0

    def __post_init__(self):
        self.validate_rnn_parameters()
        self.validate_observation_model_parameters()
        self.validate_alpha_parameters()
        self.validate_kl_annealing_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_rnn_parameters(self):
        if self.inference_n_units is None:
            raise ValueError("Please pass inference_n_units.")

        if self.model_n_units is None:
            raise ValueError("Please pass model_n_units.")

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0


class Model(VariationalInferenceModelBase):
    """DyNeMo model class.

    Parameters
    ----------
    config : osl_dynamics.models.dynemo.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""

        config = self.config

        # ---------- Define layers ---------- #
        data_drop_layer = tf.keras.layers.Dropout(
            config.inference_dropout, name="data_drop"
        )
        inf_rnn_layer = InferenceRNNLayer(
            config.inference_rnn,
            config.inference_normalization,
            config.inference_activation,
            config.inference_n_layers,
            config.inference_n_units,
            config.inference_dropout,
            config.inference_regularizer,
            name="inf_rnn",
        )
        inf_mu_layer = tf.keras.layers.Dense(config.n_modes, name="inf_mu")
        inf_sigma_layer = tf.keras.layers.Dense(
            config.n_modes, activation="softplus", name="inf_sigma"
        )
        theta_layer = SampleNormalDistributionLayer(
            config.theta_std_epsilon,
            name="theta",
        )
        alpha_layer = SoftmaxLayer(
            config.initial_alpha_temperature,
            config.learn_alpha_temperature,
            name="alpha",
        )
        means_layer = VectorsLayer(
            config.n_modes,
            config.n_channels,
            config.learn_means,
            config.initial_means,
            config.means_regularizer,
            name="means",
        )
        # If we mix precisions, the layer learns the mode precisions
        if config.mix_precisions:
            covs_name = "precs"
            initial_covs = obs_mod.invert_matrices(
                config.initial_covariances, config.diagonal_covariances
            )
        else:
            covs_name = "covs"
            initial_covs = config.initial_covariances
        if config.diagonal_covariances:
            covs_layer = DiagonalMatricesLayer(
                config.n_modes,
                config.n_channels,
                config.learn_covariances,
                initial_covs,
                config.covariances_epsilon,
                config.covariances_regularizer,
                name=covs_name,
            )
        else:
            covs_layer = CovarianceMatricesLayer(
                config.n_modes,
                config.n_channels,
                config.learn_covariances,
                initial_covs,
                config.covariances_epsilon,
                config.covariances_regularizer,
                name=covs_name,
            )
        mix_means_layer = MixVectorsLayer(name="mix_means")
        mix_covs_layer = MixMatricesLayer(name="mix_" + covs_name)
        ll_loss_layer = LogLikelihoodLossLayer(
            config.loss_calc, config.mix_precisions, name="ll_loss"
        )
        theta_drop_layer = tf.keras.layers.Dropout(
            config.model_dropout,
            name="theta_drop",
        )
        mod_rnn_layer = ModelRNNLayer(
            config.model_rnn,
            config.model_normalization,
            config.model_activation,
            config.model_n_layers,
            config.model_n_units,
            config.model_dropout,
            config.model_regularizer,
            name="mod_rnn",
        )
        mod_mu_layer = tf.keras.layers.Dense(config.n_modes, name="mod_mu")
        mod_sigma_layer = tf.keras.layers.Dense(
            config.n_modes, activation="softplus", name="mod_sigma"
        )
        kl_div_layer = KLDivergenceLayer(
            config.theta_std_epsilon, config.loss_calc, name="kl_div"
        )
        kl_loss_layer = KLLossLayer(config.do_kl_annealing, name="kl_loss")

        # ---------- Forward pass ---------- #

        # Encoder
        data = tf.keras.layers.Input(
            shape=(config.sequence_length, config.n_channels), name="data"
        )
        data_drop = data_drop_layer(data)
        inf_rnn = inf_rnn_layer(data_drop)
        inf_mu = inf_mu_layer(inf_rnn)
        inf_sigma = inf_sigma_layer(inf_rnn)
        theta = theta_layer([inf_mu, inf_sigma])
        alpha = alpha_layer(theta)

        # Observation model
        mu = means_layer(data)
        D = covs_layer(data)
        m = mix_means_layer([alpha, mu])
        C = mix_covs_layer([alpha, D])
        ll_loss = ll_loss_layer([data, m, C])

        # Temporal prior
        theta_drop = theta_drop_layer(theta)
        mod_rnn = mod_rnn_layer(theta_drop)
        mod_mu = mod_mu_layer(mod_rnn)
        mod_sigma = mod_sigma_layer(mod_rnn)
        kl_div = kl_div_layer([inf_mu, inf_sigma, mod_mu, mod_sigma])
        kl_loss = kl_loss_layer(kl_div)

        # ---------- Create model ---------- #
        inputs = {"data": data}
        outputs = {"ll_loss": ll_loss, "kl_loss": kl_loss, "theta": theta}
        name = config.model_name
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    def get_means(self):
        """Get the mode means.

        Returns
        -------
        means : np.ndarary
            Mode means.
        """
        return obs_mod.get_observation_model_parameter(self.model, "means")

    def get_covariances(self):
        """Get the mode covariances.

        Returns
        -------
        covariances : np.ndarary
            Mode covariances.
        """
        if self.config.mix_precisions:
            return obs_mod.invert_matrices(self.get_precisions())
        return obs_mod.get_observation_model_parameter(self.model, "covs")

    def get_precisions(self):
        """Get the mode precisions (inverse covariances).

        Returns
        -------
        precisions : np.ndarary
            Mode precisions.
        """
        if self.config.mix_precisions:
            return obs_mod.get_observation_model_parameter(self.model, "precs")
        return obs_mod.invert_matrices(self.get_covariances())

    def get_means_covariances(self):
        """Get the mode means and covariances.

        This is a wrapper for :code:`get_means` and :code:`get_covariances`.

        Returns
        -------
        means : np.ndarary
            Mode means.
        covariances : np.ndarray
            Mode covariances.
        """
        return self.get_means(), self.get_covariances()

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_means_covariances`."""
        return self.get_means_covariances()

    def set_means(self, means, update_initializer=True):
        """Set the mode means.

        Parameters
        ----------
        means : np.ndarray
            Mode means. Shape is (n_modes, n_channels).
        update_initializer : bool
            Do we want to use the passed means when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            means,
            layer_name="means",
            update_initializer=update_initializer,
        )

    def set_covariances(self, covariances, update_initializer=True):
        """Set the mode covariances.

        Parameters
        ----------
        covariances : np.ndarray
            Mode covariances. Shape is (n_modes, n_channels, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed covariances when we re-initialize
            the model?
        """
        if self.config.mix_precisions:
            covariances = obs_mod.invert_matrices(
                covariances, self.config.diagonal_covariances
            )
            layer_name = "precs"
        else:
            layer_name = "covs"
        obs_mod.set_observation_model_parameter(
            self.model,
            covariances,
            layer_name=layer_name,
            update_initializer=update_initializer,
            diagonal_covariances=self.config.diagonal_covariances,
        )

    def set_means_covariances(
        self,
        means,
        covariances,
        update_initializer=True,
    ):
        """This is a wrapper for :code:`set_means` and :code:`set_covariances`."""
        self.set_means(
            means,
            update_initializer=update_initializer,
        )
        self.set_covariances(
            covariances,
            update_initializer=update_initializer,
        )

    def set_observation_model_parameters(
        self, observation_model_parameters, update_initializer=True
    ):
        """Wrapper for :code:`set_means_covariances`."""
        self.set_means_covariances(
            observation_model_parameters[0],
            observation_model_parameters[1],
            update_initializer=update_initializer,
        )

    def set_regularizers(self, training_dataset):
        """Set the means and covariances regularizer based on the training data.

        A multivariate normal prior is applied to the mean vectors with
        :code:`mu=0`, :code:`sigma=diag((range/2)**2)`. If
        :code:`config.diagonal_covariances=True`, a log normal prior is
        applied to the diagonal of the covariances matrices with :code:`mu=0`,
        :code:`sigma=sqrt(log(2*range))`, otherwise an inverse Wishart prior
        is applied to the covariances matrices with :code:`nu=n_channels-1+0.1`
        and :code:`psi=diag(1/range)`.

        Parameters
        ----------
        training_dataset : tf.data.Dataset or osl_dynamics.data.Data
            Training dataset.
        """
        _logger.info("Setting regularizers")

        training_dataset = self.make_dataset(
            training_dataset, shuffle=False, concatenate=True
        )
        n_sequences, range_ = dtf.get_n_sequences_and_range(training_dataset)
        scale_factor = self.get_static_loss_scaling_factor(n_sequences)

        if self.config.learn_means:
            obs_mod.set_means_regularizer(self.model, range_, scale_factor)

        if self.config.learn_covariances and self.config.mix_precisions:
            _logger.warning("No regularizer is set for the mode precisions.")
        elif self.config.learn_covariances:
            obs_mod.set_covariances_regularizer(
                self.model,
                range_,
                self.config.covariances_epsilon,
                scale_factor,
                self.config.diagonal_covariances,
            )

    def fit_and_get_alpha(self, dataset, sigmas=None, epochs=None, verbose=1,
                          do_kl_annealing=None, n_kl_annealing_epochs=None,
                          kl_factor=None, freeze_covariances=False,
                          freeze_means=False, regress_means=False,
                          coordinated_dropout=0.0):
        """Fit DyNeMo and return per-session mode mixing coefficients.

        Runs a custom GradientTape loop training all model parameters
        (inference RNN, model RNN, observation model) and captures alpha
        from the last epoch directly, avoiding a second forward pass.

        KL annealing and learning-rate decay match the behaviour of the
        standard :code:`VariationalInferenceModelBase.fit()` + callbacks.

        Parameters
        ----------
        dataset : list of np.ndarray
            Per-session data arrays, each shaped ``(T_i, n_channels)``.
        sigmas : list of np.ndarray, optional
            Per-session posterior covariance cubes from DMvN, each shaped
            ``(T_i, n_channels, n_channels)`` (float32).  When provided,
            the fully-Bayesian second-order emission correction

            .. math::

                0.5 \\sum_t \\operatorname{tr}(\\Lambda_t \\Sigma_t), \\quad
                \\Lambda_t = \\Bigl(\\sum_k \\alpha_{t,k} D_k\\Bigr)^{-1}

            is added to the ELBO after the inference network has produced
            :math:`\\alpha_t` from the first-order data.  :math:`\\Lambda_t`
            is the mixed precision: the inverse of the blended covariance, or
            with ``mix_precisions`` the blended precision
            :math:`\\sum_k \\alpha_{t,k} P_k` (no inverse).  Unless
            ``config.second_order_alpha_gradient``, :math:`\\alpha_t` is held
            fixed in this term (``tf.stop_gradient``), so only the mode
            matrices receive its gradient.  Pass ``None`` (default) for the
            original first-order behaviour.
        epochs : int, optional
            Number of training epochs.  Defaults to ``config.n_epochs``.
        verbose : int, optional
            Verbosity (0 = silent, 1 = per-epoch summary).
        do_kl_annealing : bool, optional
            Override ``config.do_kl_annealing`` for this call only.
            ``None`` (default) uses the config value.
        n_kl_annealing_epochs : int, optional
            Override ``config.n_kl_annealing_epochs`` for this call only.
            ``None`` (default) uses the config value.
        kl_factor : float, optional
            Fixed KL weight for the whole call (no annealing within it), for
            a KL annealing schedule across calls. ``None`` (default): as set
            by ``do_kl_annealing``.
        freeze_covariances, freeze_means : bool, optional
            Hold the mode matrices / means fixed in this call (their gradients
            are zeroed; exact if they have not been trained before).
        regress_means : bool, optional
            Before training, set the means by ridge regression of the data on
            the alpha the inference network gives for it.
        coordinated_dropout : float, optional
            Rate of conditional coordinated dropout: per sequence, each channel
            is hidden from the inference network with this probability (at
            least one hidden and one shown), and the log-likelihood (with its
            second-order term) is that of the hidden channels given the shown
            ones, rescaled by n_channels / n_hidden. The returned alpha is then
            inferred from all channels after training (:code:`infer_alpha`).

        Returns
        -------
        alphas : list of np.ndarray
            Per-session mode mixing coefficients, each shaped
            ``(T_i, n_modes)``.
        """

        print('KL annealing is', 'ON' if (do_kl_annealing if do_kl_annealing is not None else self.config.do_kl_annealing) else 'OFF')
        print('KL annealing epochs:', n_kl_annealing_epochs if n_kl_annealing_epochs is not None else self.config.n_kl_annealing_epochs)
        if kl_factor is not None:
            print('KL factor (fixed for this call):', kl_factor)
        print('##########################################################')
        if epochs is None:
            epochs = self.config.n_epochs

        seq_len    = self.config.sequence_length
        n_channels = self.config.n_channels
        n_modes    = self.config.n_modes

        # ------------------------------------------------------------------
        # 1. Slice sessions into (seq_len,) windows and build one TF dataset.
        #    Record per-session metadata for later reassembly.
        # ------------------------------------------------------------------
        session_lengths = [x.shape[0] for x in dataset]
        session_n_seqs  = [T // seq_len for T in session_lengths]

        def _to_seqs(x):
            n = x.shape[0] // seq_len
            return x[:n * seq_len].reshape(n, seq_len, n_channels).astype(
                np.float32)

        x_all = np.concatenate([_to_seqs(x) for x in dataset], axis=0)

        if x_all.shape[0] == 0:
            return [
                np.full((T, n_modes), 1.0 / n_modes, dtype=np.float32)
                for T in session_lengths
            ]

        if sigmas is not None:
            def _sigma_to_seqs(sigma):
                n = sigma.shape[0] // seq_len
                return sigma[:n * seq_len].reshape(
                    n, seq_len, n_channels, n_channels).astype(np.float32)
            sigma_all = np.concatenate(
                [_sigma_to_seqs(s) for s in sigmas], axis=0)
            tf_dataset = (
                tf.data.Dataset
                .from_tensor_slices({"data": x_all, "sigma": sigma_all})
                .batch(self.config.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            tf_dataset = (
                tf.data.Dataset
                .from_tensor_slices({"data": x_all})
                .batch(self.config.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        # ------------------------------------------------------------------
        # 2. Build optimizer and locate layers used in the loop.
        # ------------------------------------------------------------------
        opt_cfg = {
            "class_name": self.config.optimizer.lower(),
            "config": {"learning_rate": float(self.config.learning_rate)},
        }
        if getattr(self.config, "gradient_clip", None) is not None:
            opt_cfg["config"]["clipnorm"] = self.config.gradient_clip

        if sigmas is not None:
            # Cache the optimizer across VB iterations to preserve Adam
            # momentum state, matching the HMMWrapper/HMMGroup design.
            if not hasattr(self, "_profumo_sigma_optimizer"):
                self._profumo_sigma_optimizer = tf.keras.optimizers.get(opt_cfg)
            optimizer = self._profumo_sigma_optimizer
        else:
            # Cache optimizer so Adam momentum accumulates across VB iterations,
            # matching the behaviour of the sigma path.
            if not hasattr(self, "_profumo_optimizer"):
                self._profumo_optimizer = tf.keras.optimizers.get(opt_cfg)
            optimizer = self._profumo_optimizer

        all_vars      = self.model.trainable_variables
        kl_loss_layer = self.model.get_layer("kl_loss")
        alpha_layer   = self.model.get_layer("alpha")

        lr_decay      = float(getattr(self.config, "lr_decay", 0.0))
        do_kl         = (do_kl_annealing if do_kl_annealing is not None
                         else self.config.do_kl_annealing)
        n_kl          = (n_kl_annealing_epochs if n_kl_annealing_epochs is not None
                         else (self.config.n_kl_annealing_epochs if do_kl else 0))
        decay_start   = n_kl if do_kl else 0

        # Reset KL factor to 0 at the start of each fit call, or hold the
        # given factor for the whole call.
        if kl_factor is not None:
            do_kl, n_kl, decay_start = False, 0, 0
            kl_loss_layer.annealing_factor.assign(float(kl_factor))
        elif do_kl:
            kl_loss_layer.annealing_factor.assign(0.0)

        # ------------------------------------------------------------------
        # 3a. Compiled training step (built once per variant, cached on self).
        #
        #     The inference RNN sees the first-order data x_t = E[A_t] only.
        #     With sigmas, the expected log-likelihood under q(A) adds
        #
        #         0.5 * tr(Lambda_t Sigma_t)
        #
        #     to the loss, where Lambda_t is the mixed precision: with
        #     mix_precisions, Lambda_t = sum_k alpha_{t,k} P_k, so the term is
        #     0.5 * sum_k alpha_{t,k} tr(P_k Sigma_t) and needs no inverse;
        #     otherwise Lambda_t = (sum_k alpha_{t,k} D_k)^{-1}. The term
        #     depends on alpha as well as on the mode matrices. With
        #     config.second_order_alpha_gradient the encoder gets its gradient
        #     (the exact ELBO, as the HMM folds the term into its E-step);
        #     otherwise alpha is held fixed in it (stop_gradient) and only the
        #     mode matrices learn from it.
        #
        #     kl_loss_layer.annealing_factor and optimizer.learning_rate are
        #     TF Variables, so per-epoch Python assignments outside @tf.function
        #     are visible inside the compiled graph without retracing.
        # ------------------------------------------------------------------
        use_sigma  = sigmas is not None
        alpha_grad = bool(self.config.second_order_alpha_gradient)
        if not hasattr(self, "_profumo_train_steps"):
            self._profumo_train_steps = {}
        cd_rate    = float(coordinated_dropout)
        step_key = (use_sigma, alpha_grad, cd_rate)
        if step_key not in self._profumo_train_steps:
            self._profumo_train_steps[step_key] = self._make_profumo_train_step(
                optimizer, use_sigma, alpha_grad, cd_rate)
        train_step = self._profumo_train_steps[step_key]
        self._profumo_grad_scales["mats"].assign(0.0 if freeze_covariances else 1.0)
        self._profumo_grad_scales["means"].assign(0.0 if freeze_means else 1.0)
        if regress_means:
            self._profumo_regress_means(dataset)

        # ------------------------------------------------------------------
        # 3b. Custom GradientTape training loop.
        # ------------------------------------------------------------------
        epoch_losses = []

        for epoch in range(epochs):
            # ---- KL annealing factor (matches KLAnnealingCallback) --------
            # on_epoch_end(e-1) sets the factor used by epoch e.
            # For linear curve that gives factor = e / n_kl (0 at epoch 0).
            if do_kl:
                if epoch < n_kl:
                    curve = self.config.kl_annealing_curve
                    if curve == "tanh":
                        kl_factor = float(
                            0.5 * np.tanh(
                                self.config.kl_annealing_sharpness
                                * (epoch - 0.5 * n_kl) / n_kl
                            ) + 0.5
                        )
                    else:  # linear
                        kl_factor = epoch / n_kl
                else:
                    kl_factor = 1.0
                kl_loss_layer.annealing_factor.assign(kl_factor)

            # ---- Learning-rate decay ------------------------------------
            cur_lr = float(
                self.config.learning_rate
                if epoch < decay_start
                else self.config.learning_rate
                     * np.exp(-lr_decay * (epoch - decay_start + 1))
            )
            optimizer.learning_rate = cur_lr

            inf_mu_batches = []
            inf_sigma_batches = []
            epoch_loss     = 0.0
            n_batches      = 0

            for batch in tf_dataset:
                sigma_batch = (tf.cast(batch["sigma"], tf.float32)
                               if use_sigma else None)
                loss, inf_mu_out, inf_sigma_out = train_step(batch["data"], sigma_batch)

                epoch_loss += float(loss)
                n_batches  += 1

                if epoch == epochs - 1:
                    inf_mu_batches.append(inf_mu_out.numpy())  # (B, seq_len, K)
                    inf_sigma_batches.append(inf_sigma_out.numpy())

            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            if verbose:
                print(f"[fit_and_get_alpha] epoch {epoch + 1}/{epochs}  "
                      f"loss={avg_loss:.4f}")

        # Store history so get_free_energy() can read it.
        self.history = {"loss": epoch_losses}
        if cd_rate > 0:
            # The last epoch's alpha saw masked inputs
            return self.infer_alpha(dataset)

        # ------------------------------------------------------------------
        # 4. Recover alpha from the last epoch: its posterior expectation
        #    E_q[alpha] = E[softmax(theta)], theta ~ N(inf_mu, inf_sigma),
        #    which is what PROFUMO's update needs (Lambda_t is linear in
        #    alpha); not softmax(inf_mu) (sharper) nor a single sample.
        # ------------------------------------------------------------------
        last_alpha_seqs = self._expected_alpha(
            np.concatenate(inf_mu_batches, axis=0),
            np.concatenate(inf_sigma_batches, axis=0))            # (N, S, K)

        # ------------------------------------------------------------------
        # 5. Reassemble per-session alpha with remainder padding.
        # ------------------------------------------------------------------
        alphas     = []
        seq_offset = 0
        for T, n_seqs in zip(session_lengths, session_n_seqs):
            if n_seqs == 0:
                alphas.append(
                    np.full((T, n_modes), 1.0 / n_modes, dtype=np.float32))
                continue

            a = last_alpha_seqs[seq_offset: seq_offset + n_seqs]  # (n_seqs, S, K)
            a = a.reshape(n_seqs * seq_len, n_modes)

            remainder = T - n_seqs * seq_len
            if remainder > 0:
                a = np.concatenate(
                    [a, np.tile(a[-1:], (remainder, 1))], axis=0)

            alphas.append(a.astype(np.float32))
            seq_offset += n_seqs

        return alphas

    def infer_alpha(self, dataset, batch_size=256):
        """Per-session alpha from the inference network alone (no training).

        PROFUMO's E-step without an M-step (subject-only updates, e.g. in
        bigData batches): alpha is its posterior expectation
        :code:`E[softmax(theta)]` (see :code:`_expected_alpha`), with sessions
        cut into :code:`sequence_length` windows and the remainder padded as
        in :code:`fit_and_get_alpha`.

        Parameters
        ----------
        dataset : list of np.ndarray
            Per-session data arrays, each shaped ``(T_i, n_channels)``.
        batch_size : int, optional
            Number of windows per forward pass.

        Returns
        -------
        alphas : list of np.ndarray
            Per-session mode mixing coefficients, each ``(T_i, n_modes)``.
        """
        seq_len    = self.config.sequence_length
        n_channels = self.config.n_channels
        n_modes    = self.config.n_modes

        if not hasattr(self, "_profumo_encoder"):
            L = {name: self.model.get_layer(name) for name in
                 ["data_drop", "inf_rnn", "inf_mu", "inf_sigma"]}

            @tf.function
            def encoder(x):
                h = L["inf_rnn"](L["data_drop"](x, training=False), training=False)
                return (L["inf_mu"](h, training=False),
                        L["inf_sigma"](h, training=False))

            self._profumo_encoder = encoder

        lengths = [x.shape[0] for x in dataset]
        n_seqs  = [T // seq_len for T in lengths]
        seqs = [x[:n * seq_len].reshape(n, seq_len, n_channels).astype(np.float32)
                for x, n in zip(dataset, n_seqs) if n > 0]
        alpha_seqs = None
        if seqs:
            x_all = np.concatenate(seqs, axis=0)
            outs = [self._profumo_encoder(x_all[i:i + batch_size])
                    for i in range(0, x_all.shape[0], batch_size)]
            alpha_seqs = self._expected_alpha(
                np.concatenate([o[0].numpy() for o in outs], axis=0),
                np.concatenate([o[1].numpy() for o in outs], axis=0))

        alphas, offset = [], 0
        for T, n in zip(lengths, n_seqs):
            if n == 0:
                alphas.append(np.full((T, n_modes), 1.0 / n_modes, dtype=np.float32))
                continue
            a = alpha_seqs[offset: offset + n].reshape(n * seq_len, n_modes)
            if T > n * seq_len:
                a = np.concatenate([a, np.tile(a[-1:], (T - n * seq_len, 1))], axis=0)
            alphas.append(a.astype(np.float32))
            offset += n
        return alphas

    def _expected_alpha(self, inf_mu, inf_sigma, n_samples=20, seed=0):
        """Posterior expectation of alpha, E[softmax(theta)].

        Monte Carlo over :code:`n_samples` draws of
        :code:`theta ~ N(inf_mu, inf_sigma)` through the alpha layer (with its
        temperature). Arrays are ``(..., n_modes)``. PROFUMO can override
        :code:`n_samples` with :code:`self._profumo_alpha_samples`; 0 gives
        softmax(inf_mu) instead (sharper than E[alpha]).
        """
        alpha_layer = self.model.get_layer("alpha")
        n_samples = getattr(self, "_profumo_alpha_samples", n_samples)
        if n_samples <= 0:
            return alpha_layer(inf_mu).numpy()
        rng = np.random.default_rng(seed)
        total = np.zeros(inf_mu.shape, dtype=np.float64)
        for _ in range(n_samples):
            theta = inf_mu + inf_sigma * rng.standard_normal(inf_mu.shape).astype(np.float32)
            total += alpha_layer(theta).numpy()
        return (total / n_samples).astype(np.float32)

    def profumo_nfindr_initialization(self, dataset, n_init=0, n_init_epochs=2,
                                      window_length=None, n_restarts=10,
                                      init_means=True):
        """N-FINDR initialization for PROFUMO's DyNeMo.

        The mode matrices are set to the N-FINDR vertices of the windows of
        :code:`dataset` (:func:`osl_dynamics.models.nfindr.nfindr_covariances`;
        window precisions with a maximum likelihood refit if the model mixes
        precisions) and, if the means are learnt and :code:`init_means`, the
        means to the N-FINDR vertices of the window means. With means, each
        window is centred before its second moment is taken. The model is
        then trained from there for :code:`n_init_epochs` epochs on
        :code:`n_init` random subsets, keeping the best (none if
        :code:`n_init` is 0).

        Parameters
        ----------
        dataset : list of np.ndarray
            Per-session data, each ``(T_i, n_channels)``.
        n_init, n_init_epochs : int, optional
            Short random-subset fits from the N-FINDR start.
        window_length : int, optional
            Window length. Defaults to max(20, n_channels) for precision
            mixing and 20 otherwise, at most :code:`sequence_length`.
        n_restarts : int, optional
            Random starts of N-FINDR.
        init_means : bool, optional
            Also set the means with N-FINDR (if they are learnt).
        """
        from osl_dynamics.models.nfindr import nfindr_covariances, nfindr_means

        precisions = self.config.mix_precisions
        n_channels = self.config.n_channels
        if window_length is None:
            window_length = max(20, n_channels) if precisions else 20
        window_length = min(window_length, self.config.sequence_length)
        x = [np.asarray(d, dtype=np.float64) for d in dataset]
        learn_means = self.config.learn_means
        if learn_means:
            # Centre each window, so the mode means do not enter its second moment
            windows = []
            for d in x:
                n = d.shape[0] // window_length
                w = d[: n * window_length].reshape(n, window_length, n_channels)
                windows.extend(w - w.mean(axis=1, keepdims=True))
            cov_input = windows
        else:
            cov_input = x
        seed = np.random.randint(2**31 - 1)
        covariances = nfindr_covariances(
            cov_input, self.config.n_modes, window_length=window_length,
            n_restarts=n_restarts, precisions=precisions, random_seed=seed)
        self.set_covariances(covariances, update_initializer=True)
        print(f"[profumo_nfindr_initialization] mode matrices from N-FINDR "
              f"(window {window_length}, {'precisions' if precisions else 'covariances'})")
        if learn_means and init_means:
            means = nfindr_means(x, self.config.n_modes, random_seed=seed)
            self.set_means(means, update_initializer=True)
            print("[profumo_nfindr_initialization] means from N-FINDR")
        if n_init > 0:
            from osl_dynamics.data import Data
            self.random_subset_initialization(Data(dataset), n_epochs=n_init_epochs,
                                              n_init=n_init)

    def profumo_hmm_initialization(self, dataset, n_init=0, n_init_epochs=2,
                                   hmm_epochs=20, init_means=True, sigma_mean=None):
        """HMM initialization for PROFUMO's DyNeMo.

        A hidden Markov model with :code:`n_modes` states (full covariances,
        and means if the model learns them and :code:`init_means`) is fitted
        to :code:`dataset`, and DyNeMo's mode covariances (and means) are set
        to its state covariances (and means). When the modes are often close
        to pure, the HMM states sit near the mode matrices, and they are
        estimated from every sample rather than from short windows (as in
        N-FINDR). The model is then trained from there for
        :code:`n_init_epochs` epochs on :code:`n_init` random subsets,
        keeping the best (none if :code:`n_init` is 0).

        Parameters
        ----------
        dataset : list of np.ndarray
            Per-session data, each ``(T_i, n_channels)``.
        n_init, n_init_epochs : int, optional
            Short random-subset fits from the HMM start.
        hmm_epochs : int, optional
            Training epochs of the HMM.
        init_means : bool, optional
            Also set the means from the HMM (if they are learnt).
        sigma_mean : np.ndarray, optional
            Average posterior covariance of the data, ``(n_channels,
            n_channels)``. If given, it is added to the HMM state
            covariances: with uncertain data the M-step needs
            :code:`E[x x'] = E[x] E[x]' + Sigma`, and the HMM only saw
            :code:`E[x]`.
        """
        from osl_dynamics.data import Data
        from osl_dynamics.models import hmm

        means = bool(self.config.learn_means and init_means)
        hmm_config = hmm.Config(
            n_states=self.config.n_modes,
            n_channels=self.config.n_channels,
            sequence_length=self.config.sequence_length,
            learn_means=means,
            learn_covariances=True,
            batch_size=self.config.batch_size,
            learning_rate=0.01,
            n_epochs=hmm_epochs,
        )
        hmm_model = hmm.Model(hmm_config)
        data = Data([np.asarray(d, dtype=np.float32) for d in dataset])
        hmm_model.random_state_time_course_initialization(data, n_epochs=2, n_init=3)
        hmm_model.fit(data, verbose=0)
        covariances = hmm_model.get_covariances()
        if sigma_mean is not None:
            covariances = covariances + np.asarray(sigma_mean, dtype=np.float32)[None]
        self.set_covariances(covariances, update_initializer=True)
        print(f"[profumo_hmm_initialization] mode covariances from a {hmm_epochs}-epoch HMM"
              + (" + average posterior covariance" if sigma_mean is not None else ""))
        if means:
            self.set_means(hmm_model.get_means(), update_initializer=True)
            print("[profumo_hmm_initialization] means from the HMM")
        del hmm_model
        if n_init > 0:
            self.random_subset_initialization(data, n_epochs=n_init_epochs, n_init=n_init)

    def _profumo_regress_means(self, dataset, ridge=1e-3):
        """Set the means by ridge regression of the data on the inferred alpha.

        Under :code:`m_t = sum_k alpha_tk mu_k`, the least-squares means given
        alpha solve :code:`(A'A + ridge T I) mu = A'X` (pooled over sessions).
        """
        alphas = self.infer_alpha(dataset)
        a = np.concatenate(alphas).astype(np.float64)
        x = np.concatenate([np.asarray(d, dtype=np.float64) for d in dataset])
        gram = a.T @ a + ridge * len(a) * np.eye(a.shape[1])
        means = np.linalg.solve(gram, a.T @ x)
        self.set_means(means.astype(np.float32), update_initializer=False)
        print("[_profumo_regress_means] means regressed on alpha")

    def _make_profumo_train_step(self, optimizer, use_sigma, alpha_grad, cd_rate=0.0):
        """Compiled training step for :code:`fit_and_get_alpha`.

        The layers are called one by one (rather than :code:`self.model`) so
        that the second-order term can reuse alpha and the mode matrices.
        Returns a function of :code:`(x_batch, sigma_batch)` giving the loss
        and the inference network's posterior mean of theta.
        """
        model     = self.model
        all_vars  = model.trainable_variables
        mats_name = "precs" if self.config.mix_precisions else "covs"
        names     = ["data_drop", "inf_rnn", "inf_mu", "inf_sigma", "theta",
                     "alpha", "means", mats_name, "mix_means",
                     "mix_" + mats_name, "ll_loss", "theta_drop", "mod_rnn",
                     "mod_mu", "mod_sigma", "kl_div", "kl_loss"]
        L              = {name: model.get_layer(name) for name in names}
        mix_precisions = self.config.mix_precisions
        loss_calc      = self.config.loss_calc

        # Gradient multipliers (0 = frozen) for the mode matrices and means,
        # shared by every compiled variant and set per call
        if not hasattr(self, "_profumo_grad_scales"):
            self._profumo_grad_scales = {
                "mats": tf.Variable(1.0, trainable=False),
                "means": tf.Variable(1.0, trainable=False),
            }
        mats_ids  = {id(v) for v in L[mats_name].trainable_variables}
        means_ids = {id(v) for v in L["means"].trainable_variables}
        mults = [self._profumo_grad_scales["mats"] if id(v) in mats_ids
                 else self._profumo_grad_scales["means"] if id(v) in means_ids
                 else None for v in all_vars]

        # Conditional coordinated dropout helpers. With the hidden-channel mask
        # h (B, 1, M), the shown channels' covariance is embedded in an M x M
        # matrix with the identity on the hidden channels, so every sequence
        # has the same shapes whatever its mask
        log2pi = np.log(2 * np.pi)

        def reduce_t(v):
            # (B, S) -> scalar, as LogLikelihoodLossLayer
            if loss_calc == "sum":
                return tf.reduce_mean(tf.reduce_sum(v, axis=1))
            return tf.reduce_mean(v)

        def shown_covariance(Lam, h_mask):
            v_mask = 1.0 - h_mask
            vv = v_mask[..., :, None] * v_mask[..., None, :]
            cov = tf.linalg.inv(Lam)
            eye_h = tf.linalg.diag(tf.broadcast_to(h_mask, tf.shape(cov)[:-1]))
            return cov * vv + eye_h, v_mask, vv

        def conditional_ll(x, m, Lam, h_mask):
            # log p(x_hidden | x_shown) = log p(x) - log p(x_shown), (B, S)
            d = x - m
            Lc = tf.linalg.cholesky(Lam)
            z = tf.linalg.matvec(Lc, d, adjoint_a=True)
            ll_full = (tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lc)), axis=-1)
                       - 0.5 * tf.reduce_sum(tf.square(z), axis=-1))
            cov_v, v_mask, _ = shown_covariance(Lam, h_mask)
            Lv = tf.linalg.cholesky(cov_v)
            zv = tf.linalg.triangular_solve(Lv, (d * v_mask)[..., None], lower=True)[..., 0]
            ll_shown = (-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lv)), axis=-1)
                        - 0.5 * tf.reduce_sum(tf.square(zv), axis=-1))
            n_hidden = tf.reduce_sum(h_mask, axis=-1)            # (B, 1)
            return ll_full - ll_shown - 0.5 * n_hidden * log2pi

        @tf.function
        def train_step(x_batch, sigma_batch):
            with tf.GradientTape() as tape:
                # Encoder: first-order data only
                if cd_rate > 0:
                    u = tf.random.uniform(
                        tf.stack([tf.shape(x_batch)[0], 1, tf.shape(x_batch)[-1]]))
                    hidden = tf.logical_or(
                        u < cd_rate, u <= tf.reduce_min(u, axis=-1, keepdims=True))
                    hidden = tf.logical_and(
                        hidden, u < tf.reduce_max(u, axis=-1, keepdims=True))
                    h_mask = tf.cast(hidden, x_batch.dtype)       # (B, 1, M)
                    cd_scale = (tf.cast(tf.shape(x_batch)[-1], x_batch.dtype)
                                / tf.reduce_sum(h_mask, axis=-1))  # (B, 1)
                    enc_in = x_batch * (1.0 - h_mask) / (1.0 - cd_rate)
                else:
                    enc_in = x_batch
                data_drop = L["data_drop"](enc_in, training=True)
                inf_rnn   = L["inf_rnn"](data_drop, training=True)
                inf_mu    = L["inf_mu"](inf_rnn, training=True)
                inf_sigma = L["inf_sigma"](inf_rnn, training=True)
                theta     = L["theta"]([inf_mu, inf_sigma], training=True)
                alpha     = L["alpha"](theta, training=True)

                # Observation model (D: mode covariances or precisions)
                mu  = L["means"](x_batch, training=True)
                D   = L[mats_name](x_batch, training=True)        # (K, M, M)
                m   = L["mix_means"]([alpha, mu], training=True)
                C   = L["mix_" + mats_name]([alpha, D], training=True)
                if cd_rate > 0:
                    Lam = C if mix_precisions else tf.linalg.inv(C)
                    nll = -reduce_t(cd_scale * conditional_ll(x_batch, m, Lam, h_mask))
                else:
                    nll = tf.squeeze(L["ll_loss"]([x_batch, m, C], training=True))

                # Temporal prior (model RNN)
                theta_drop = L["theta_drop"](theta, training=True)
                mod_rnn    = L["mod_rnn"](theta_drop, training=True)
                mod_mu     = L["mod_mu"](mod_rnn, training=True)
                mod_sigma  = L["mod_sigma"](mod_rnn, training=True)
                kl_div     = L["kl_div"](
                    [inf_mu, inf_sigma, mod_mu, mod_sigma], training=True)
                kl = tf.squeeze(L["kl_loss"](kl_div, training=True))

                loss = nll + kl
                reg = L["means"].losses + L[mats_name].losses
                if reg:
                    loss = loss + tf.add_n(reg)

                # Second-order term 0.5 * tr(Lambda_t Sigma_t)
                if use_sigma:
                    a = alpha if alpha_grad else tf.stop_gradient(alpha)
                    if mix_precisions:
                        tr_k = tf.einsum("kmn,bsmn->bsk", D, sigma_batch)
                        corr = tf.reduce_sum(a * tr_k, axis=-1)       # (B, S)
                    else:
                        C_a  = L["mix_" + mats_name]([a, D], training=True)
                        corr = tf.einsum("bsmn,bsmn->bs",
                                         tf.linalg.inv(C_a), sigma_batch)
                    if cd_rate > 0:
                        # E_q log p(x_h | x_v): minus 0.5 tr(C_vv^-1 Sigma_vv) too
                        if mix_precisions:
                            Lam_a = L["mix_" + mats_name]([a, D], training=True)
                        else:
                            Lam_a = tf.linalg.inv(L["mix_" + mats_name]([a, D], training=True))
                        cov_v_a, _, vv = shown_covariance(Lam_a, h_mask)
                        corr_v = tf.einsum("bsmn,bsmn->bs", tf.linalg.inv(cov_v_a),
                                           sigma_batch * vv)
                        corr = cd_scale * (corr - corr_v)
                    loss = loss + 0.5 * reduce_t(corr)

            grads = tape.gradient(loss, all_vars)
            grads = [g if (m is None or g is None) else g * m
                     for g, m in zip(grads, mults)]
            optimizer.apply_gradients(zip(grads, all_vars))
            return loss, inf_mu, inf_sigma

        return train_step

    def sample_alpha(self, n_samples, theta=None):
        """Uses the model RNN to sample mode mixing factors, :code:`alpha`.

        Parameters
        ----------
        n_samples : int
            Number of samples to take.
        theta : np.ndarray, optional
            Normalized logits to initialise the sampling with.
            Shape must be (sequence_length, n_modes).

        Returns
        -------
        alpha : np.ndarray
            Sampled alpha.
        """
        # Get layers
        model_rnn_layer = self.model.get_layer("mod_rnn")
        mod_mu_layer = self.model.get_layer("mod_mu")
        mod_sigma_layer = self.model.get_layer("mod_sigma")
        alpha_layer = self.model.get_layer("alpha")

        # Normally distributed random numbers used to sample the logits theta
        epsilon = np.random.normal(0, 1, [n_samples + 1, self.config.n_modes]).astype(
            np.float32
        )

        if theta is None:
            # Sequence of the underlying logits theta
            theta = np.zeros(
                [self.config.sequence_length, self.config.n_modes],
                dtype=np.float32,
            )

            # Randomly sample the first time step
            theta[-1] = np.random.normal(size=self.config.n_modes)

        # Sample the mode fixing factors
        alpha = np.empty([n_samples, self.config.n_modes], dtype=np.float32)
        for i in trange(n_samples, desc="Sampling mode time course"):
            # If there are leading zeros we trim theta so that we don't pass
            # the zeros
            trimmed_theta = theta[~np.all(theta == 0, axis=1)][np.newaxis, :, :]

            # Predict the probability distribution function for theta one time
            # step in the future, p(theta|theta_<t) ~ N(mod_mu, sigma_theta_jt)
            model_rnn = model_rnn_layer(trimmed_theta)
            mod_mu = mod_mu_layer(model_rnn)[0, -1]
            mod_sigma = mod_sigma_layer(model_rnn)[0, -1]

            # Shift theta one time step to the left
            theta = np.roll(theta, -1, axis=0)

            # Sample from the probability distribution function
            theta[-1] = mod_mu + mod_sigma * epsilon[i]

            # Calculate the mode mixing factors
            alpha[i] = alpha_layer(mod_mu[np.newaxis, np.newaxis, :])[0, 0]

        return alpha

    def fine_tuning(
        self, training_data, n_epochs=None, learning_rate=None, store_dir="tmp"
    ):
        """Fine tuning the model for each session.

        Here, we train the inference RNN and observation model with the model
        RNN fixed held fixed at the group-level.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data
            Training dataset.
        n_epochs : int, optional
            Number of epochs to train for. Defaults to the value in the
            :code:`config` used to create the model.
        learning_rate : float, optional
            Learning rate. Defaults to the value in the :code:`config` used
            to create the model.
        store_dir : str, optional
            Directory to temporarily store the model in.

        Returns
        -------
        alpha : list of np.ndarray
            Session-specific mixing coefficients.
            Each element has shape (n_samples, n_modes).
        means : np.ndarray
            Session-specific means. Shape is (n_sessions, n_modes, n_channels).
        covariances : np.ndarray
            Session-specific covariances.
            Shape is (n_sessions, n_modes, n_channels, n_channels).
        """
        # Save the group level model
        os.makedirs(store_dir, exist_ok=True)
        self.save_weights(f"{store_dir}/model.weights.h5")

        # Save original training hyperparameters
        original_n_epochs = self.config.n_epochs
        original_learning_rate = self.config.learning_rate
        original_do_kl_annealing = self.config.do_kl_annealing
        self.config.n_epochs = n_epochs or self.config.n_epochs
        self.config.learning_rate = learning_rate or self.config.learning_rate
        self.config.do_kl_annealing = False

        # Layers to fix (i.e. make non-trainable)
        fixed_layers = ["mod_rnn", "mod_mu", "mod_sigma"]

        # Fine tune on sessions
        alpha = []
        means = []
        covariances = []
        with self.set_trainable(fixed_layers, False), set_logging_level(
            _logger, logging.WARNING
        ):
            for i in trange(training_data.n_sessions, desc="Fine tuning"):
                # Train on this session
                with training_data.set_keep(i):
                    self.fit(training_data, verbose=0)
                    a = self.get_alpha(
                        training_data,
                        concatenate=True,
                        verbose=0,
                    )

                # Get inferred parameters
                m, c = self.get_means_covariances()
                alpha.append(a)
                means.append(m)
                covariances.append(c)

                # Reset back to group-level model parameters
                self.load_weights(f"{store_dir}/model.weights.h5")
                self.compile()

        # Reset hyperparameters
        self.config.n_epochs = original_n_epochs
        self.config.learning_rate = original_learning_rate
        self.config.do_kl_annealing = original_do_kl_annealing

        return alpha, np.array(means), np.array(covariances)

    def dual_estimation(
        self,
        training_data,
        n_epochs=None,
        learning_rate=None,
        store_dir="tmp",
    ):
        """Dual estimation to get the session-specific observation model parameters.

        Here, we train the observation model parameters (mode means and
        covariances) with the inference RNN and model RNN held fixed at
        the group-level.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data or list of tf.data.Dataset
            Training dataset.
        n_epochs : int, optional
            Number of epochs to train for. Defaults to the value in the
            :code:`config` used to create the model.
        learning_rate : float, optional
            Learning rate. Defaults to the value in the :code:`config` used
            to create the model.
        store_dir : str, optional
            Directory to temporarily store the model in.

        Returns
        -------
        means : np.ndarray
            Session-specific means. Shape is (n_sessions, n_modes, n_channels).
        covariances : np.ndarray
            Session-specific covariances.
            Shape is (n_sessions, n_modes, n_channels, n_channels).
        """
        # Save the group level model
        os.makedirs(store_dir, exist_ok=True)
        self.save_weights(f"{store_dir}/model_weights.weights.h5")

        # Save original training hyperparameters
        original_n_epochs = self.config.n_epochs
        original_learning_rate = self.config.learning_rate
        self.config.n_epochs = n_epochs or self.config.n_epochs
        self.config.learning_rate = learning_rate or self.config.learning_rate

        # Layers to fix (i.e. make non-trainable)
        fixed_layers = [
            "mod_rnn",
            "mod_mu",
            "mod_sigma",
            "inf_rnn",
            "inf_mu",
            "inf_sigma",
            "theta",
            "alpha",
        ]

        # Dual estimation on sessions
        means = []
        covariances = []
        with self.set_trainable(fixed_layers, False):
            if isinstance(training_data, list):
                n_sessions = len(training_data)
            else:
                n_sessions = training_data.n_sessions

            for i in trange(n_sessions, desc="Dual estimation"):
                # Train on this session
                if isinstance(training_data, list):
                    self.fit(training_data[i], verbose=0)
                else:
                    with training_data.set_keep(i):
                        self.fit(training_data, verbose=0)

                # Get inferred parameters
                m, c = self.get_means_covariances()
                means.append(m)
                covariances.append(c)

                # Reset back to group-level model parameters
                self.load_weights(f"{store_dir}/model_weights.weights.h5")
                self.compile()

        # Reset hyperparameters
        self.config.n_epochs = original_n_epochs
        self.config.learning_rate = original_learning_rate

        return np.array(means), np.array(covariances)
