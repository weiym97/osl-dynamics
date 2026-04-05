# Workspace Context

See the main CLAUDE.md in the **profumo** folder for full workspace context.

## This folder: osl-dynamics

- **Path:** `/well/win-fmrib-analysis/users/uap971/new-osl-dynamics/osl-dynamics`
- **Language:** Python
- **Purpose:** OSL-Dynamics — WIN Oxford package for analysing neuroimaging timeseries (HMM, DyNeMo, etc.)
- **Package root:** `osl_dynamics/`

## Sibling projects

- **profumo** (`/gpfs3/well/win-fmrib-analysis/users/uap971/profumo`): C++ sPROFUMO inference engine
- **pfm_simulation** (`/well/win-fmrib-analysis/users/uap971/pfm_simulation`): PFM simulation and analysis scripts

## Data Policy (CRITICAL)

Never read data files: `.nii`, `.nii.gz`, `.dscalar.nii`, `.dlabel.nii`, `.npy`, `.npz`, `.mat`, or `.txt` files containing numerical data.

## Key File: `osl_dynamics/models/hmm.py`

This file is the primary integration point between osl-dynamics and PROFUMO (C++ side). Always read it carefully before making changes — it has PROFUMO-specific methods added on top of the base class.

### `Config` dataclass
All HMM hyperparameters: `n_states`, `n_channels`, `sequence_length`, `learn_means`, `learn_covariances`, `batch_size`, `learning_rate`, `lr_decay`, `n_epochs`, `learn_trans_prob`, `trans_prob_update_delay`, `trans_prob_update_forget`, `baum_welch_implementation`, etc. Validated in `__post_init__`.

### `Model` class (extends `MarkovStateInferenceModelBase`)
Keras model architecture built in `build_model()`:
- `VectorsLayer` ("means") + `CovarianceMatricesLayer` / `DiagonalMatricesLayer` ("covs") → `SeparateLogLikelihoodLayer` ("ll") → `HiddenMarkovStateInferenceLayer` ("hid_state_inf") → `SumLogLikelihoodLossLayer` ("ll_loss")
- Outputs: `ll_loss`, `gamma` (T×K state probs), `xi` (T×K×K transition sufficient statistic)

### PROFUMO integration methods (do not remove or rename)

**`fit_and_get_alpha(dataset, sigmas=None, epochs=None, zscore=False)`**
Main entry point called by `HMMWrapper::fit_and_get_gammas()`. Custom `GradientTape` loop: each batch runs the full model (Baum-Welch inside `HiddenMarkovStateInferenceLayer`), updates observation-model weights, captures gamma from the **last epoch** and returns it — no second forward pass. TPM updated via manual EMA at the end. When `sigmas` (list of `(T_i, M, M)` arrays) is provided, adds the fully-Bayesian correction `0.5 * sum_{t,k} gamma_{t,k} * tr(Lambda_k * Sigma_t)` to the loss — E-step and M-step both use the corrected emission ll for consistency.

**`get_alpha(dataset, concatenate=False, zscore=False, ...)`**
Post-training inference; called by `HMMWrapper::get_state_timecourses()`. Accepts a plain list of `(T_i, M)` numpy arrays, slices into `sequence_length` windows, runs `self.model()` directly (not `model.predict()` — avoids deadlock with prefetch threads), pads remainder rows with last valid gamma row.

**`fit_with_gamma(dataset, gammas, epochs=None)`**
Alternative M-step where pre-computed gammas are supplied instead of running Baum-Welch. Updates observation model via gradient descent and TPM via EMA, without an E-step.

## General Preferences

- Keep responses concise and direct
- Prefer editing existing files over creating new ones
- Do not add unnecessary comments, docstrings, or type annotations to untouched code
- Read existing `osl_dynamics/` code before adding new code — follow its conventions
