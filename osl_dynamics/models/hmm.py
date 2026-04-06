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
                  zscore=False, **kwargs):
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
        zscore : bool, optional
            If True, z-score each session (zero mean, unit variance per
            channel) before inference.  Must match the flag used during
            training so that the data is in the same space as the learned
            state covariances.  Passed through from
            ``HMMWrapper::get_state_timecourses`` on the C++ side.

        Returns
        -------
        alpha : list of np.ndarray or np.ndarray
            State probabilities, each shaped ``(T_i, n_states)``, or
            concatenated to ``(T_total, n_states)`` if ``concatenate=True``.
        """
        # Fast-path: list of numpy arrays (HMMWrapper call convention).
        if (isinstance(dataset, list) and len(dataset) > 0
                and isinstance(dataset[0], np.ndarray)):
            # Apply the same per-session z-scoring used during training
            # (fit_and_get_alpha).  Skipping this when the model was trained
            # with zscore=True would present data in a different scale than
            # the learned covariances, producing miscalibrated state gammas.
            if zscore:
                def _zscore(x):
                    mu  = x.mean(axis=0, keepdims=True)
                    std = x.std(axis=0, keepdims=True)
                    std = np.where(std < 1e-8, 1.0, std)
                    return (x - mu) / std
                dataset = [_zscore(x) for x in dataset]

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
                # Use self.model() directly (synchronous forward pass) instead
                # of self.model.predict(), which spawns its own background
                # data-pipeline threads and deadlocks against the prefetch
                # threads already running on tf_ds.
                gamma_seqs = []
                for batch in tf_ds:
                    pred = self.model(batch, training=False)
                    gamma_seqs.append(pred["gamma"].numpy())   # (B, seq_len, K)
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
        #    Reuse all four layers from self.model directly — they were
        #    already traced correctly in build_model() with the right shapes,
        #    and their weights are shared so gradient updates target the main
        #    model. gamma is injected as a new Input, bypassing
        #    HiddenMarkovStateInferenceLayer entirely.
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
    
    def fit_and_get_alpha(self, dataset, sigmas=None, epochs=None, zscore=False, verbose=1):
        """Fit the observation model and return state probabilities in one pass.

        Runs a single custom training loop so that TF/Keras is invoked only
        once per PROFUMO group-level VB update.  The full model (including
        ``HiddenMarkovStateInferenceLayer``) is executed on every forward
        pass, so Baum-Welch runs inside each batch.  Gamma outputs from the
        **last epoch** are captured directly from ``model(batch)["gamma"]``
        and returned to the caller — no second forward pass is needed.

        The observation model (means / covariances) is updated via
        ``GradientTape``; the TPM is updated at the end via the same manual
        EMA as ``fit_with_gamma``.

        Parameters
        ----------
        dataset : list of np.ndarray
            Per-session data arrays, each shaped ``(T_i, n_channels)``.
        sigmas : list of np.ndarray, optional
            Per-session posterior covariance cubes from DMvN, each shaped
            ``(T_i, n_channels, n_channels)`` (float32).  When provided,
            the fully-Bayesian second-order emission correction
            ``0.5 * sum_{t,k} gamma_{t,k} * tr(Lambda_k * Sigma_t)``
            is added to the loss, preventing state-covariance collapse.
            Pass ``None`` (default) for the original first-order behaviour.
        epochs : int, optional
            Number of training epochs.  Defaults to ``config.n_epochs``.
        zscore : bool, optional
            If True, z-score each session to zero mean / unit variance per
            channel before training.  Default False.
        verbose : int, optional
            Verbosity (0 = silent, 1 = per-epoch summary).

        Returns
        -------
        gammas : list of np.ndarray
            Per-session state probabilities, each shaped
            ``(T_i, n_states)``.
        """
        if epochs is None:
            epochs = self.config.n_epochs

        seq_len    = self.config.sequence_length
        n_channels = self.config.n_channels
        n_states   = self.config.n_states

        print('sequence length: ', seq_len)
        print('n channels: ', n_channels)
        print('n states: ', n_states)

        # ------------------------------------------------------------------
        # 1. Slice sessions into (seq_len,) windows and build one TF dataset
        #    covering all sessions concatenated.  Record per-session metadata
        #    so we can reassemble per-session gammas afterwards.
        # ------------------------------------------------------------------
        if zscore:
            def _zscore(x):
                mu  = x.mean(axis=0, keepdims=True)
                std = x.std(axis=0, keepdims=True)
                std = np.where(std < 1e-8, 1.0, std)
                return (x - mu) / std
            dataset = [_zscore(x) for x in dataset]

        session_lengths = [x.shape[0] for x in dataset]
        session_n_seqs  = [T // seq_len for T in session_lengths]

        def _to_seqs(x):
            n = x.shape[0] // seq_len
            return x[:n * seq_len].reshape(n, seq_len, n_channels).astype(np.float32)

        def _sigma_to_seqs(sigma):
            n = sigma.shape[0] // seq_len
            return sigma[:n * seq_len].reshape(
                n, seq_len, n_channels, n_channels).astype(np.float32)

        x_all = np.concatenate([_to_seqs(x) for x in dataset], axis=0)

        if sigmas is not None:
            print('Sigmas provided')
            sigma_all = np.concatenate(
                [_sigma_to_seqs(s) for s in sigmas], axis=0)  # (N, seq_len, M, M)
            print('x_all shape: ',x_all.shape)
            print('sigma_all shape: ',sigma_all.shape)
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
        
        if sigmas is not None:
            print('Start: sanity check for debugging: ')
            _seg_labels = [
                (slice(0, 100),   'state1 seg1'),
                (slice(100, 200), 'state2 seg1'),
                (slice(200, 300), 'state1 seg2'),
                (slice(300, 400), 'state2 seg2'),
            ]
            for _sl, _label in _seg_labels:
                _ea_segs  = [_sess_ea[_sl]  for _sess_ea  in dataset]
                _cov_segs = [_sess_cov[_sl] for _sess_cov in sigmas]
                _ea_k  = np.concatenate(_ea_segs,  axis=0)  # (100*n_sessions, M)
                _cov_k = np.concatenate(_cov_segs, axis=0)  # (100*n_sessions, M, M)
                _signal_cov = np.cov(_ea_k.T)
                _noise_cov  = _cov_k.mean(axis=0)
                _map_cov    = _signal_cov + _noise_cov
                print(f'{_label}: signal cov =\n', _signal_cov)
                print(f'{_label}: avg noise cov =\n', _noise_cov)
                print(f'{_label}: MAP cov =\n', _map_cov)

            print('Look at TPM: ',self.get_trans_prob())
            print('End: sanity check for debugging.')
        
        # ------------------------------------------------------------------
        # 2. Observation-model trainable variables only (means + covs).
        #    The TPM is handled separately via manual EMA below.
        # ------------------------------------------------------------------
        obs_vars = (
            self.model.get_layer("means").trainable_variables +
            self.model.get_layer("covs").trainable_variables
        )

        lr_decay  = getattr(self.config, "lr_decay", 0.0)
        # Create the observation-model optimizer once and reuse across VB
        # iterations.  A fresh optimizer per call would retrace apply_gradients
        # (TF recompile) and discard Adam momentum state.
        # Named with a leading underscore so it does not clash with anything in
        # ModelBase (self.config, self.model) or MarkovStateInferenceModelBase
        # (self.model.optimizer is the MarkovStateModelOptimizer used by fit()).
        if not hasattr(self, "_profumo_obs_optimizer"):
            self._profumo_obs_optimizer = tf.keras.optimizers.get({
                "class_name": self.config.optimizer.lower(),
                "config": {"learning_rate": self.config.learning_rate},
            })
        optimizer = self._profumo_obs_optimizer

        # ------------------------------------------------------------------
        # 3. Custom GradientTape training loop.
        #    The full model runs on every batch: data → ll → Baum-Welch →
        #    gamma → loss.  We collect gamma from the last epoch's batches.
        #
        #    TPM update: replicates the per-batch EMA of standard fit().
        #    rho is computed once per epoch from the local epoch index so
        #    the decay schedule restarts each PROFUMO VB call, matching
        #    EMADecayCallback in standard fit().
        # ------------------------------------------------------------------
        last_gamma_seqs = None   # (N_total_seqs, seq_len, K), set on last epoch

        for epoch in range(epochs):
            # LR and rho both use the local epoch so the schedule restarts
            # each VB call rather than accumulating across iterations.
            optimizer.learning_rate = float(
                self.config.learning_rate * np.exp(-lr_decay * epoch))

            # Compute rho and prior once per epoch (constant within the
            # epoch, matching EMADecayCallback.on_epoch_end semantics).
            if self.config.learn_trans_prob:
                prior_np = self.model.get_layer("hid_state_inf").trans_prob_prior.numpy(
                ).astype(np.float64)                                    # (K, K)
                prior_row_sums = prior_np.sum(axis=-1, keepdims=True)   # (K, 1)
                rho = (
                    100 * epoch / epochs + 1
                    + self.config.trans_prob_update_delay
                ) ** -self.config.trans_prob_update_forget

            gamma_batches = []
            xi_batches    = []
            epoch_loss    = 0.0
            n_batches     = 0

            for batch in tf_dataset:
                x_batch = batch["data"]

                with tf.GradientTape() as tape:
                    if sigmas is None:
                        # --------------------------------------------------
                        # Standard first-order path: run the full Keras model
                        # (obs layers → Baum-Welch → loss) in one call.
                        # --------------------------------------------------
                        outputs = self.model({"data": x_batch}, training=True)
                        loss    = tf.reduce_mean(outputs["ll_loss"])
                        if self.model.losses:
                            loss = loss + tf.add_n(self.model.losses)
                        gamma_out = outputs["gamma"]
                        xi_out    = outputs["xi"]

                    else:
                        # --------------------------------------------------
                        # Second-order (fully-Bayesian) path.
                        #
                        # E-step and M-step must use the SAME corrected
                        # emission log-likelihood:
                        #
                        #   log p̃(x_t | k) = log p(x_t | k)
                        #                    − 0.5 · tr(Λ_k · Σ_t)
                        #
                        # Using uncorrected ll in Baum-Welch but corrected
                        # ll only in the gradient (old approach) makes the
                        # E-step and M-step inconsistent: inflated covariances
                        # broaden the emission Gaussians, causing state
                        # assignments to leak across states and driving
                        # covariances further above C_k^sample + C_k^sigma.
                        #
                        # By folding −0.5·tr(Λ_k·Σ_t) into the emission LL
                        # before Baum-Welch, the E-step accounts for the
                        # posterior uncertainty of the observations.  The
                        # gradient of the corrected loss w.r.t. Λ_k then
                        # automatically contains the sigma term — no separate
                        # correction term is needed, and the EM fixed point is
                        # self-consistent at C_k = C_k^sample + C_k^sigma.
                        # --------------------------------------------------
                        sigma_batch = tf.cast(batch["sigma"], tf.float32)

                        # Compute emission LL via the obs-model layers.
                        covs_layer    = self.model.get_layer("covs")
                        means_layer   = self.model.get_layer("means")
                        ll_layer      = self.model.get_layer("ll")
                        cov_matrices  = covs_layer(x_batch, training=True)   # (K,M,M)
                        mu            = means_layer(x_batch, training=True)  # (K,M)
                        ll_raw        = ll_layer(
                            [x_batch, mu, cov_matrices], training=True)      # (B,S,K)

                        # Sigma correction to emission LL:
                        #   correction[b,s,k] = 0.5 * tr(Λ_k * Σ_{b,s})
                        #                     = 0.5 * einsum(kmn,bsmn->bsk)
                        prec_matrices = tf.linalg.inv(cov_matrices)          # (K,M,M)
                        ll_sigma_corr = 0.5 * tf.einsum(
                            "kmn,bsmn->bsk", prec_matrices, sigma_batch)
                        ll_corrected  = ll_raw - ll_sigma_corr               # (B,S,K)

                        # E-step: Baum-Welch on corrected emission LL.
                        hid_state_inf = self.model.get_layer("hid_state_inf")
                        gamma_out, xi_out = hid_state_inf(ll_corrected)

                        # M-step loss: −E_γ[log p̃(x_t|k)] (includes sigma).
                        ll_loss_layer = self.model.get_layer("ll_loss")
                        ll_loss       = ll_loss_layer([ll_corrected, gamma_out])
                        loss          = tf.reduce_mean(ll_loss)
                        if self.model.losses:
                            loss = loss + tf.add_n(self.model.losses)

                grads = tape.gradient(loss, obs_vars)
                optimizer.apply_gradients(zip(grads, obs_vars))

                gamma_batches.append(gamma_out.numpy())  # (B, seq_len, K)
                xi_batches.append(xi_out.numpy())         # (B, seq_len-1, K, K)
                epoch_loss += float(loss)
                n_batches  += 1

                # Per-batch TPM EMA update — mirrors ExponentialMovingAverage
                # in standard fit(): A ← (1−ρ)·A + ρ·φ_batch, where φ_batch
                # is HiddenMarkovStateInferenceLayer._trans_prob_update(γ, ξ).
                # rho is fixed for the epoch (computed above), matching the
                # EMADecayCallback.on_epoch_end convention.
                if self.config.learn_trans_prob:
                    xi_b = xi_out.numpy().astype(np.float64)    # (B, S-1, K, K)
                    g_b  = gamma_out.numpy().astype(np.float64) # (B, S,   K)
                    phi_batch = (
                        xi_b.mean(axis=(0, 1)) + prior_np
                    ) / (
                        g_b[:, :-1].mean(axis=(0, 1))[:, np.newaxis] + prior_row_sums
                    )
                    current_tp = self.get_trans_prob().astype(np.float64)
                    new_tp     = (1.0 - rho) * current_tp + rho * phi_batch
                    new_tp     = new_tp / new_tp.sum(axis=1, keepdims=True)
                    self.set_trans_prob(new_tp.astype(np.float32))

            if verbose:
                print(f"[fit_and_get_alpha] epoch {epoch + 1}/{epochs}  "
                      f"loss={epoch_loss / max(n_batches, 1):.4f}")

            # Gamma summary: avg per 100-tp segment across sessions
            if sigmas is not None:
                _all_gamma = np.concatenate(gamma_batches, axis=0)  # (N_seqs, seq_len, K)
                _sess_gammas, _seq_off = [], 0
                for _n_seqs in session_n_seqs:
                    if _n_seqs > 0:
                        _g = _all_gamma[_seq_off:_seq_off + _n_seqs].reshape(
                            _n_seqs * seq_len, n_states)
                        _sess_gammas.append(_g)
                    _seq_off += _n_seqs
                _seg_labels = [
                    '  t=0:100   (GT state1)', '  t=100:200 (GT state2)',
                    '  t=200:300 (GT state1)', '  t=300:400 (GT state2)',
                ]
                print(f'Epoch {epoch + 1} gamma summary (avg across sessions):')
                for _i, _label in enumerate(_seg_labels):
                    _sl = slice(_i * 100, (_i + 1) * 100)
                    _segs = [_g[_sl] for _g in _sess_gammas
                             if _g.shape[0] >= (_i + 1) * 100]
                    if _segs:
                        _mean = np.stack(_segs).mean(axis=(0, 1))
                        print(f'{_label}: {np.array2string(_mean, precision=3)}')

            if epoch == epochs - 1:
                last_gamma_seqs = np.concatenate(
                    gamma_batches, axis=0)           # (N_total_seqs, seq_len, K)

        # ------------------------------------------------------------------
        # 4. Reassemble per-session gamma arrays.
        #    Trailing remainder rows (T % seq_len) are padded by repeating
        #    the last valid row — same convention as get_alpha.
        # ------------------------------------------------------------------
        gammas     = []
        seq_offset = 0
        for T, n_seqs in zip(session_lengths, session_n_seqs):
            if n_seqs == 0:
                gammas.append(np.zeros((T, n_states), dtype=np.float32))
                continue

            g = last_gamma_seqs[seq_offset: seq_offset + n_seqs]  # (n_seqs, seq_len, K)
            g = g.reshape(n_seqs * seq_len, n_states)

            remainder = T - n_seqs * seq_len
            if remainder > 0:
                g = np.concatenate(
                    [g, np.tile(g[-1:], (remainder, 1))], axis=0)

            gammas.append(g.astype(np.float32))
            seq_offset += n_seqs

        return gammas

    def initialize_from_sessions(self, sessions, n_init=3, n_init_epochs=1,
                                 zscore=False):
        """Data-driven initialization of state means/covariances from sessions.

        Called by PROFUMO's HMMWrapper on the very first group-level VB
        iteration, when actual mode timecourses (A) are first available.
        All K states start with near-identical covariances (≈ I), so
        gradient descent alone cannot break the symmetry.  This method
        samples random state time courses, computes per-state empirical
        covariances, sets them as the observation model, and optionally
        trains for a few epochs.  Repeats ``n_init`` times and keeps the
        trial with the lowest loss.

        Parameters
        ----------
        sessions : list of np.ndarray
            Per-session data arrays, each shaped ``(T_i, n_channels)``.
        n_init : int, optional
            Number of random initialization trials.  Default 3.
        n_init_epochs : int, optional
            Number of training epochs per trial.  Default 1.
        zscore : bool, optional
            If True, z-score each session before initialization.

        Returns
        -------
        gammas : list of np.ndarray
            Per-session state probabilities from the best trial,
            each shaped ``(T_i, n_states)``.
        """
        n_states   = self.config.n_states
        n_channels = self.config.n_channels

        # Optional per-session z-scoring (must match fit_and_get_alpha).
        if zscore:
            def _zscore(x):
                mu  = x.mean(axis=0, keepdims=True)
                std = x.std(axis=0, keepdims=True)
                std = np.where(std < 1e-8, 1.0, std)
                return (x - mu) / std
            sessions = [_zscore(x) for x in sessions]

        data_all = np.concatenate(sessions, axis=0)   # (T_total, M)
        T_total  = data_all.shape[0]

        # We need the TPM for sampling.  If it is still diagonal (untrained),
        # build a simple high-self-transition TPM so we get sticky segments.
        try:
            trans_prob = self.get_trans_prob()
            if np.allclose(trans_prob, np.eye(n_states)):
                raise ValueError("diagonal")
        except (ValueError, RuntimeError):
            trans_prob = (
                np.ones((n_states, n_states)) * 0.1 / max(n_states - 1, 1)
            )
            np.fill_diagonal(trans_prob, 0.9)
            self.set_trans_prob(trans_prob.astype(np.float32))

        best_loss    = np.inf
        best_weights = None
        best_gammas  = None

        for trial in range(n_init):
            _logger.info(f"[initialize_from_sessions] trial {trial + 1}/{n_init}")

            # ----- sample a random state time course from the TPM -----
            from osl_dynamics.simulation.hmm import HMM as SimHMM
            sim = SimHMM(trans_prob)
            stc = sim.generate_states(T_total)          # (T_total, K) one-hot

            # Make sure every state activates with enough data points.
            for retry in range(100):
                counts = stc.sum(axis=0)
                non_active = counts < 2 * n_channels
                if not np.any(non_active):
                    break
                new_stc = sim.generate_states(T_total)
                for k in range(n_states):
                    if non_active[k] and new_stc[:, k].sum() > 0:
                        stc[:, k] = new_stc[:, k]
            else:
                # If we still have empty states, fall back to random partition.
                _logger.warning(
                    "Could not activate all states via TPM sampling; "
                    "falling back to random partition."
                )
                indices = np.random.randint(0, n_states, size=T_total)
                stc = np.zeros((T_total, n_states), dtype=np.float32)
                stc[np.arange(T_total), indices] = 1.0

            # ----- compute per-state empirical mean & covariance -----
            means       = np.zeros((n_states, n_channels), dtype=np.float32)
            covariances = np.zeros(
                (n_states, n_channels, n_channels), dtype=np.float32
            )
            for k in range(n_states):
                mask = stc[:, k] == 1
                x_k  = data_all[mask]
                if x_k.shape[0] < 2 * n_channels:
                    # Not enough points — use identity as fallback.
                    covariances[k] = np.eye(n_channels, dtype=np.float32)
                    continue
                means[k] = x_k.mean(axis=0)
                if n_channels == 1:
                    covariances[k] = np.var(x_k).reshape(1, 1)
                else:
                    covariances[k] = np.cov(x_k, rowvar=False)

            # ----- inject into the Keras model -----
            if self.config.learn_means:
                self.set_means(means, update_initializer=True)
            if self.config.learn_covariances:
                self.set_covariances(covariances, update_initializer=True)

            # ----- short training run to refine & evaluate -----
            if n_init_epochs > 0:
                gammas = self.fit_and_get_alpha(
                    sessions, epochs=n_init_epochs, zscore=False, verbose=0,
                )
            else:
                gammas = self.fit_and_get_alpha(
                    sessions, epochs=1, zscore=False, verbose=0,
                )

            # Evaluate loss: use the last-epoch loss from the training history
            # stored internally.  fit_and_get_alpha doesn't return it, so we
            # compute it via a quick forward pass on a small sample.
            try:
                seq_len = self.config.sequence_length
                x_sample = data_all[:max(T_total // seq_len, 1) * seq_len]
                x_sample = x_sample.reshape(-1, seq_len, n_channels).astype(
                    np.float32
                )
                tf_sample = tf.data.Dataset.from_tensor_slices(
                    {"data": x_sample}
                ).batch(self.config.batch_size)
                total_loss = 0.0
                n_batches  = 0
                for batch in tf_sample:
                    out = self.model(batch, training=False)
                    total_loss += float(tf.reduce_mean(out["ll_loss"]))
                    n_batches  += 1
                loss = total_loss / max(n_batches, 1)
            except Exception:
                loss = np.inf

            _logger.info(
                f"[initialize_from_sessions] trial {trial + 1} loss = {loss:.4f}"
            )
            if loss < best_loss:
                best_loss    = loss
                best_weights = self.get_weights()
                best_gammas  = gammas

        # Restore best trial.
        if best_weights is not None:
            self.set_weights(best_weights)

        _logger.info(
            f"[initialize_from_sessions] best loss = {best_loss:.4f}"
        )
        return best_gammas

    def set_covariance_regularizer(self, n_sequences, c=10.0):
        """Set an Inverse-Wishart prior on state covariances.

        Mirrors PROFUMO's ``GROUP_PRECISION_MATRIX`` Wishart prior with the
        same fixed template covariance gCM = 0.9*I + 0.1*ones::

            Λ ~ W(a=c·M, B=c·M·gCM)  →  C ~ IW(ν=c·M, Ψ=c·M·gCM)

        MAP mode ≈ gCM for large c, preventing covariance vanishing/explosion.

        Parameters
        ----------
        n_sequences : int
            Total non-overlapping sequence_length windows across all sessions.
        c : float, optional
            Concentration multiplier (default 10, matching PROFUMO).
        """
        from osl_dynamics.inference import regularizers as osld_reg

        M    = self.config.n_channels
        gCM  = 0.9 * np.eye(M, dtype=np.float32) + 0.1 * np.ones((M, M), dtype=np.float32)
        nu   = c * float(M)
        psi  = nu * gCM  # float32, matches the float32 Cholesky weights in __call__

        scale_factor = 1.0 / float(n_sequences)
        if self.config.loss_calc == "mean":
            scale_factor /= float(self.config.sequence_length)

        regularizer = osld_reg.InverseWishart(
            nu       = nu,
            psi      = psi,
            epsilon  = self.config.covariances_epsilon,
            strength = scale_factor,
        )
        self.model.get_layer("covs").layers[0].regularizer = regularizer

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