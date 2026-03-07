"""Hidden Markov Model (HMM) with a Multivariate Normal observation model.

See the `documentation <https://osl-dynamics.readthedocs.io/en/latest/models\
/hmm.html>`_ for a description of this model.

See Also
--------
- D. Vidaurre, et al., "Spectrally resolved fast transient brain states in
  electrophysiological data". `Neuroimage 126, 81-95 (2016)
  <https://www.sciencedirect.com/science/article/pii/S1053811915010691>`_.
- D. Vidaurre, et al., "Discovering dynamic brain networks from big data in
  rest and task". `Neuroimage 180, 646-656 (2018)
  <https://www.sciencedirect.com/science/article/pii/S1053811917305487>`_.
- `MATLAB HMM-MAR Toolbox <https://github.com/OHBA-analysis/HMM-MAR>`_.
"""

import os
import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm.auto import trange

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference.layers import (
    VectorsLayer,
    CovarianceMatricesLayer,
    DiagonalMatricesLayer,
    SeparateLogLikelihoodLayer,
    HiddenMarkovStateInferenceLayer,
    SumLogLikelihoodLossLayer,
)
from osl_dynamics.models import obs_mod
from osl_dynamics.models.mod_base import BaseModelConfig
from osl_dynamics.models.inf_mod_base import (
    MarkovStateInferenceModelConfig,
    MarkovStateInferenceModelBase,
)
from osl_dynamics.analysis.post_hoc import hmm_dual_estimation
from osl_dynamics.utils.misc import set_logging_level

_logger = logging.getLogger("osl-dynamics")


@dataclass
class Config(BaseModelConfig, MarkovStateInferenceModelConfig):
    """Settings for the HMM.

    Parameters
    ----------
    model_name : str
        Model name.
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.

    learn_means : bool
        Should we make the mean vectors for each state trainable?
    learn_covariances : bool
        Should we make the covariance matrix for each state trainable?
    initial_means : np.ndarray
        Initialisation for mean vectors.
    initial_covariances : np.ndarray
        Initialisation for state covariances.
        If :code:`diagonal_covariances=True` and full matrices are passed,
        the diagonal is extracted.
    covariances_epsilon : float
        Error added to state covariances for numerical stability.
    diagonal_covariances : bool
        Should we learn diagonal state covariances?
    means_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for mean vectors.
    covariances_regularizer : tf.keras.regularizers.Regularizer
        Regularizer for covariance matrices.

    initial_trans_prob : np.ndarray
        Initialisation for the transition probability matrix.
    learn_trans_prob : bool
        Should we make the transition probability matrix trainable?
    trans_prob_prior : np.ndarray
        Dirichlet prior for the transition probability matrix.
        Each row is the alpha parameters of the Dirichlet distribution.
    trans_prob_update_delay : float
        We update the transition probability matrix as
        :code:`trans_prob = (1-rho) * trans_prob + rho * trans_prob_update`,
        where :code:`rho = (100 * epoch / n_epochs + 1 +
        trans_prob_update_delay) ** -trans_prob_update_forget`.
        This is the delay parameter.
    trans_prob_update_forget : float
        We update the transition probability matrix as
        :code:`trans_prob = (1-rho) * trans_prob + rho * trans_prob_update`,
        where :code:`rho = (100 * epoch / n_epochs + 1 +
        trans_prob_update_delay) ** -trans_prob_update_forget`.
        This is the forget parameter.
    initial_state_probs : np.ndarray
        State probabilities at :code:`time=0`.
    learn_initial_state_probs : bool
        Should we make the initial state probabilities trainable?
    baum_welch_implementation : str
        Which implementation of the Baum-Welch algorithm should we use?
        Either :code:`'log'` (default) or :code:`'rescale'`.

    init_method : str
        Initialization method. Defaults to 'random_state_time_course'.
    n_init : int
        Number of initializations. Defaults to 3.
    n_init_epochs : int
        Number of epochs for each initialization. Defaults to 1.
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
    n_epochs : int
        Number of training epochs.
    optimizer : str or tf.keras.optimizers.Optimizer
        Optimizer to use.
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

    model_name: str = "HMM"

    # Observation model parameters
    learn_means: bool = None
    learn_covariances: bool = None
    initial_means: np.ndarray = None
    initial_covariances: np.ndarray = None
    diagonal_covariances: bool = False
    covariances_epsilon: float = None
    means_regularizer: tf.keras.regularizers.Regularizer = None
    covariances_regularizer: tf.keras.regularizers.Regularizer = None

    # Initialization
    init_method: str = "random_state_time_course"
    n_init: int = 3
    n_init_epochs: int = 1
    init_take: float = 1.0

    def __post_init__(self):
        self.validate_observation_model_parameters()
        self.validate_hmm_parameters()
        self.validate_dimension_parameters()
        self.validate_training_parameters()

    def validate_observation_model_parameters(self):
        if self.learn_means is None or self.learn_covariances is None:
            raise ValueError("learn_means and learn_covariances must be passed.")

        if self.covariances_epsilon is None:
            if self.learn_covariances:
                self.covariances_epsilon = 1e-6
            else:
                self.covariances_epsilon = 0.0


class Model(MarkovStateInferenceModelBase):
    """HMM class.

    Parameters
    ----------
    config : osl_dynamics.models.hmm.Config
    """

    config_type = Config

    def build_model(self):
        """Builds a keras model."""

        config = self.config

        # Inputs
        data = layers.Input(
            shape=(config.sequence_length, config.n_channels),
            name="data",
        )

        # Observation model
        means_layer = VectorsLayer(
            config.n_states,
            config.n_channels,
            config.learn_means,
            config.initial_means,
            config.means_regularizer,
            name="means",
        )
        if config.diagonal_covariances:
            covs_layer = DiagonalMatricesLayer(
                config.n_states,
                config.n_channels,
                config.learn_covariances,
                config.initial_covariances,
                config.covariances_epsilon,
                config.covariances_regularizer,
                name="covs",
            )
        else:
            covs_layer = CovarianceMatricesLayer(
                config.n_states,
                config.n_channels,
                config.learn_covariances,
                config.initial_covariances,
                config.covariances_epsilon,
                config.covariances_regularizer,
                name="covs",
            )
        mu = means_layer(data)  # data not used
        D = covs_layer(data)  # data not used

        # Log-likelihood
        ll_layer = SeparateLogLikelihoodLayer(config.n_states, name="ll")
        ll = ll_layer([data, mu, D])

        # Hidden state inference
        hidden_state_inference_layer = HiddenMarkovStateInferenceLayer(
            config.n_states,
            config.sequence_length,
            config.initial_trans_prob,
            config.trans_prob_prior,
            config.initial_state_probs,
            config.learn_trans_prob,
            config.learn_initial_state_probs,
            implementation=config.baum_welch_implementation,
            dtype="float64",
            name="hid_state_inf",
        )
        gamma, xi = hidden_state_inference_layer(ll)

        # Loss
        ll_loss_layer = SumLogLikelihoodLossLayer(config.loss_calc, name="ll_loss")
        ll_loss = ll_loss_layer([ll, gamma])

        # Create model
        inputs = {"data": data}
        outputs = {"ll_loss": ll_loss, "gamma": gamma, "xi": xi}
        name = config.model_name
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    # -------------------------------------------------------------------------
    # PROFUMO integration: M-step with pre-computed state time courses
    # -------------------------------------------------------------------------

    def get_alpha(self, dataset, concatenate=False, remove_edge_effects=False,
                  **kwargs):
        """Get state probabilities.

        Extends the base-class implementation to also accept a plain list of
        ``(T_i, n_channels)`` numpy arrays, which is the format passed by
        ``HMMWrapper::get_state_timecourses`` from the C++ side.

        When a list of numpy arrays is detected, each session is manually
        sliced into non-overlapping ``sequence_length`` windows, passed
        through the model's Baum-Welch layer, and the resulting gamma
        sequences are concatenated back to ``(T_i, n_states)`` per session.

        All other input types (``osl_dynamics.data.Data``,
        ``tf.data.Dataset``) are forwarded to the base-class implementation
        unchanged.

        Parameters
        ----------
        dataset : list of np.ndarray, tf.data.Dataset, or osl_dynamics.data.Data
            If a list of ``(T_i, n_channels)`` float32 arrays, each element
            is one session.
        concatenate : bool, optional
            Concatenate across sessions into a single array.
        remove_edge_effects : bool, optional
            Passed through to the base-class implementation for non-list inputs.

        Returns
        -------
        alpha : list of np.ndarray or np.ndarray
            State probabilities, each shaped ``(T_i, n_states)``, or
            concatenated to ``(T_total, n_states)`` if ``concatenate=True``.
        """
        # Fast-path: list of numpy arrays (HMMWrapper call convention).
        if (isinstance(dataset, list) and len(dataset) > 0
                and isinstance(dataset[0], np.ndarray)):
            seq_len    = self.config.sequence_length
            n_states   = self.config.n_states
            n_channels = self.config.n_channels

            alpha = []
            for x_sess in dataset:
                T = x_sess.shape[0]
                n_seq = T // seq_len
                if n_seq == 0:
                    # Session shorter than one window — pad with zeros.
                    alpha.append(np.zeros((T, n_states), dtype=np.float32))
                    continue

                # Slice into (n_seq, seq_len, n_channels) batches.
                x_seqs = (x_sess[: n_seq * seq_len]
                          .reshape(n_seq, seq_len, n_channels)
                          .astype(np.float32))

                tf_ds = (tf.data.Dataset
                         .from_tensor_slices({"data": x_seqs})
                         .batch(self.config.batch_size)
                         .prefetch(tf.data.AUTOTUNE))

                # Collect gamma over all batches, then flatten sequences.
                gamma_seqs = []
                for batch in tf_ds:
                    pred = self.predict(batch, **kwargs)
                    gamma_seqs.append(pred["gamma"])   # (B, seq_len, K)
                # gamma_seqs → (n_seq, seq_len, K) → (n_seq*seq_len, K)
                gamma_full = np.concatenate(gamma_seqs, axis=0).reshape(
                    n_seq * seq_len, n_states)

                # Pad remainder rows with the last valid gamma row.
                remainder = T - n_seq * seq_len
                if remainder > 0:
                    pad = np.tile(gamma_full[-1:], (remainder, 1))
                    gamma_full = np.concatenate([gamma_full, pad], axis=0)

                alpha.append(gamma_full)

            if concatenate or len(alpha) == 1:
                return np.concatenate(alpha, axis=0)
            return alpha

        # Default path: Data object or tf.data.Dataset.
        return super().get_alpha(dataset, concatenate=concatenate,
                                 remove_edge_effects=remove_edge_effects,
                                 **kwargs)

    def fit_with_gamma(self, dataset, gammas, epochs=None, verbose=1):
        """Fit the group model M-step using pre-computed state time courses.

        Called by PROFUMO's HMMGroup during the VB update loop. Instead of
        running the full Baum-Welch E-step (which is embedded inside
        ``HiddenMarkovStateInferenceLayer`` as a custom gradient), we accept
        pre-computed gamma arrays from the per-run HMMRun children and:

        1. Update the observation model (means / covariances) via gradient
           descent on the log-likelihood weighted by the supplied gammas.
        2. Update the transition probability matrix (TPM) via a manual EMA
           M-step using the xi sufficient statistic computed from the gammas.

        The TPM update mirrors the EMA performed by
        ``MarkovStateModelOptimizer`` / ``ExponentialMovingAverage`` during a
        normal ``fit()`` call, using the same decay schedule as
        ``EMADecayCallback``.

        Parameters
        ----------
        dataset : list of np.ndarray
            Per-session data arrays, each shaped ``(T_i, n_channels)``.
        gammas : list of np.ndarray
            Per-session state probability arrays, each shaped
            ``(T_i, n_states)``. Must be in the same order as ``dataset``.
        epochs : int, optional
            Number of gradient-descent epochs over the observation model.
            Defaults to ``config.n_epochs``.
        verbose : int, optional
            Verbosity level (0 = silent, 1 = per-epoch summary).

        Returns
        -------
        history : dict
            ``{"loss": [float, ...]}`` — per-epoch observation model loss.
        """
        if epochs is None:
            epochs = self.config.n_epochs

        seq_len    = self.config.sequence_length
        n_states   = self.config.n_states
        n_channels = self.config.n_channels

        # ------------------------------------------------------------------
        # 1. Slice sessions into (seq_len,) windows and build a TF dataset
        #    that pairs (data_window, gamma_window) batches.
        #    We discard the trailing remainder — same convention as the
        #    base class make_dataset().
        # ------------------------------------------------------------------
        def _to_sequences(x, g):
            n = x.shape[0] // seq_len
            x = x[: n * seq_len].reshape(n, seq_len, n_channels)
            g = g[: n * seq_len].reshape(n, seq_len, n_states)
            return x.astype(np.float32), g.astype(np.float32)

        x_parts, g_parts = [], []
        for x_sess, g_sess in zip(dataset, gammas):
            xw, gw = _to_sequences(x_sess, g_sess)
            x_parts.append(xw)
            g_parts.append(gw)

        x_all = np.concatenate(x_parts, axis=0)   # (N, seq_len, M)
        g_all = np.concatenate(g_parts, axis=0)   # (N, seq_len, K)

        tf_dataset = (
            tf.data.Dataset
            .from_tensor_slices({"data": x_all, "gamma": g_all})
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # ------------------------------------------------------------------
        # 2. Observation model sub-model.
        #    Reuse the already-built layer objects from self.model so that
        #    gradient updates are applied to the same shared weights.
        #    gamma is injected as an input, bypassing HiddenMarkovStateInferenceLayer.
        # ------------------------------------------------------------------
        means_layer   = self.model.get_layer("means")
        covs_layer    = self.model.get_layer("covs")
        ll_layer      = self.model.get_layer("ll")
        ll_loss_layer = self.model.get_layer("ll_loss")

        data_in  = tf.keras.layers.Input(
            shape=(seq_len, n_channels), dtype=tf.float32, name="data")
        gamma_in = tf.keras.layers.Input(
            shape=(seq_len, n_states), dtype=tf.float32, name="gamma")

        mu      = means_layer(data_in)
        D       = covs_layer(data_in)
        ll      = ll_layer([data_in, mu, D])
        ll_loss = ll_loss_layer([ll, gamma_in])

        obs_model = tf.keras.Model(
            inputs={"data": data_in, "gamma": gamma_in},
            outputs={"ll_loss": ll_loss},
        )
        obs_model.compile(
            optimizer=tf.keras.optimizers.get({
                "class_name": self.config.optimizer.lower(),
                "config": {"learning_rate": self.config.learning_rate},
            })
        )

        # ------------------------------------------------------------------
        # 3. Gradient-descent M-step over the observation model.
        #    LR follows the same exponential decay as MarkovStateInferenceModelBase.fit().
        # ------------------------------------------------------------------
        lr_decay = getattr(self.config, "lr_decay", 0.0)
        history  = {"loss": []}

        for epoch in range(epochs):
            lr = float(self.config.learning_rate * np.exp(-lr_decay * epoch))
            # Keras 3: learning_rate is a float property, not a tf.Variable.
            # Assign via the property setter (works for Keras 2 and 3).
            obs_model.optimizer.learning_rate = lr

            h = obs_model.fit(tf_dataset, epochs=1, verbose=0)
            loss = h.history["loss"][0]
            history["loss"].append(loss)

            if verbose:
                print(f"[fit_with_gamma] epoch {epoch + 1}/{epochs}  "
                      f"loss={loss:.4f}")

        # ------------------------------------------------------------------
        # 4. TPM M-step via manual EMA update.
        #
        #    During a normal fit(), the custom gradient on
        #    HiddenMarkovStateInferenceLayer returns phi_interim (the
        #    row-normalised xi/gamma sufficient statistic) as the "gradient",
        #    and ExponentialMovingAverage applies:
        #        trans_prob = (1 - rho) * trans_prob + rho * phi_interim
        #    We replicate this exactly using the gammas already computed by
        #    HMMRun's Baum-Welch, without needing another forward pass.
        #
        #    rho follows the EMADecayCallback schedule evaluated at the last
        #    epoch: rho = (100 * (epochs-1) / n_epochs + 1 + delay)^{-forget}
        # ------------------------------------------------------------------
        if self.config.learn_trans_prob:
            # Concatenate all sessions for a single aggregate update,
            # consistent with PROFUMO's one-TPM-update-per-VB-iteration.
            g_concat = np.concatenate(
                [g.reshape(-1, n_states) for g in gammas], axis=0
            ).astype(np.float64)                              # (T_total, K)

            # xi sufficient statistic: sum_{t} gamma_t outer gamma_{t+1}
            # Shape: (K, K), then row-normalise to get phi_interim.
            xi_sum = g_concat[:-1].T @ g_concat[1:]          # (K, K)
            row_sums = xi_sum.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            phi_interim = xi_sum / row_sums                   # (K, K), rows sum to 1

            # EMA decay at the final epoch of this call.
            n_epochs_total = self.config.n_epochs
            last_epoch = epochs - 1
            rho = (
                100 * last_epoch / n_epochs_total + 1
                + self.config.trans_prob_update_delay
            ) ** -self.config.trans_prob_update_forget

            current_tp = self.get_trans_prob().astype(np.float64)
            new_tp = (1.0 - rho) * current_tp + rho * phi_interim

            # Renormalise rows to sum to 1 (guards against numerical drift).
            new_tp = new_tp / new_tp.sum(axis=1, keepdims=True)

            self.set_trans_prob(new_tp.astype(np.float32))

        return history

    def get_means(self):
        """Get the state means.

        Returns
        -------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "means")

    def get_covariances(self):
        """Get the state covariances.

        Returns
        -------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        """
        return obs_mod.get_observation_model_parameter(self.model, "covs")

    def get_means_covariances(self):
        """Get the state means and covariances.

        This is a wrapper for :code:`get_means` and :code:`get_covariances`.

        Returns
        -------
        means : np.ndarray
            State means.
        covariances : np.ndarray
            State covariances.
        """
        return self.get_means(), self.get_covariances()

    def get_observation_model_parameters(self):
        """Wrapper for :code:`get_means_covariances`."""
        return self.get_means_covariances()

    def set_means(self, means, update_initializer=True):
        """Set the state means.

        Parameters
        ----------
        means : np.ndarray
            State means. Shape is (n_states, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed means when we re-initialize the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            means,
            layer_name="means",
            update_initializer=update_initializer,
        )

    def set_covariances(self, covariances, update_initializer=True):
        """Set the state covariances.

        Parameters
        ----------
        covariances : np.ndarray
            State covariances. Shape is (n_states, n_channels, n_channels).
        update_initializer : bool, optional
            Do we want to use the passed covariances when we re-initialize
            the model?
        """
        obs_mod.set_observation_model_parameter(
            self.model,
            covariances,
            layer_name="covs",
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
        self.set_means(means, update_initializer=update_initializer)
        self.set_covariances(covariances, update_initializer=update_initializer)

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
        :code:`config.diagonal_covariances=True`, a log normal prior is applied
        to the diagonal of the covariances matrices with :code:`mu=0`,
        :code:`sigma=sqrt(log(2*range))`, otherwise an inverse Wishart prior is
        applied to the covariances matrices with :code:`nu=n_channels-1+0.1`
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

        if self.config.learn_covariances:
            obs_mod.set_covariances_regularizer(
                self.model,
                range_,
                self.config.covariances_epsilon,
                scale_factor,
                self.config.diagonal_covariances,
            )

    def dual_estimation(self, training_data, alpha=None, concatenate=False, n_jobs=1):
        """Dual estimation to get session-specific observation model parameters.

        This function is the wrapper for the :code:`hmm_dual_estimation` function.

        Here, we estimate the state means and covariances for sessions
        with the posterior distribution of the states held fixed.

        Parameters
        ----------
        training_data : osl_dynamics.data.Data or list of tf.data.Dataset
            Prepared training data object.
        alpha : list of np.ndarray, optional
            Posterior distribution of the states. Shape is
            (n_sessions, n_samples, n_states).
        concatenate : bool, optional
            Should we concatenate the data across sessions?
        n_jobs : int, optional
            Number of jobs to run in parallel.

        Returns
        -------
        means : np.ndarray
            Session-specific means. Shape is (n_sessions, n_states, n_channels).
        covariances : np.ndarray
            Session-specific covariances.
            Shape is (n_sessions, n_states, n_channels, n_channels).
            When ``config.diagonal_covariances=True``, the matrices are
            diagonal (zeros off-diagonal) and encode per-channel variances only.
        """
        if alpha is None:
            # Get the posterior
            alpha = self.get_alpha(training_data, concatenate=concatenate)

        if isinstance(alpha, np.ndarray):
            alpha = [alpha]

        # Get the session-specific data
        if isinstance(training_data, list):
            data = []
            for d in training_data:
                subject_data = []
                for batch in d:
                    subject_data.append(np.concatenate(batch["data"]))
                data.append(np.concatenate(subject_data))
        else:
            data = training_data.time_series(prepared=True, concatenate=concatenate)

        if isinstance(data, np.ndarray):
            data = [data]

        # Make sure the data and alpha have the same number of samples
        data = [d[: a.shape[0]] for d, a in zip(data, alpha)]

        # Estimate session-specific observation model parameters
        means, covariances = hmm_dual_estimation(
            data,
            alpha,
            zero_mean=(not self.config.learn_means),
            diagonal_covariances=self.config.diagonal_covariances,
            eps=self.config.covariances_epsilon,
            n_jobs=n_jobs,
        )

        return means, covariances

    def fine_tuning(
        self,
        training_data,
        n_epochs=None,
        learning_rate=None,
        store_dir="tmp",
    ):
        """Fine tuning the model for each session.

        Here, we estimate the posterior distribution (state probabilities)
        and observation model using the data from a single session with the
        group-level transition probability matrix held fixed.

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
            Session-specific state probabilities.
            Each element has shape (n_samples, n_states).
        means : np.ndarray
            Session-specific means. Shape is (n_sessions, n_states, n_channels).
        covariances : np.ndarray
            Session-specific covariances.
            Shape is (n_sessions, n_states, n_channels, n_channels).
        """
        # Save group-level model parameters
        os.makedirs(store_dir, exist_ok=True)
        self.save_weights(f"{store_dir}/model.weights.h5")

        # Temporarily change hyperparameters
        original_n_epochs = self.config.n_epochs
        original_learning_rate = self.config.learning_rate
        self.config.n_epochs = n_epochs or self.config.n_epochs
        self.config.learning_rate = learning_rate or self.config.learning_rate

        # Layers to fix (i.e. make non-trainable)
        fixed_layers = ["hid_state_inf"]

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

                # Get the inferred parameters
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

        return alpha, np.array(means), np.array(covariances)