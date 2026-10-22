# DyNeMo-with-means collapse: simulation study

Goal: a simulation in which DyNeMo with learned means fails the way it does on
HCP ICA_50_mean, and a comparison of fixes. Simulated data only; no real data.

## Mechanism being tested

The bidirectional inference RNN sees x_t, and the mean m_t = sum alpha mu is linear
in alpha. So alpha can follow the channels' private (channel-specific) noise, and
the covariance shrinks to the leftover. With slow private noise (fMRI-like), the
learned prior makes this cheap.

## Files

- `simulate.py`: x = m(alpha) + N(0, sum alpha D) + private AR(1) noise (phi = 0.9 smooth, 0 white)
- `metrics.py`: scores against the truth.
  - lam_min / n<0.5: generalised eigenvalues of the model's time-averaged covariance against the true conditional covariance
  - KL_truth: KL(true conditional || model)
  - R2 of the model mean on the true private noise, i.e. how much it copies
  - matched alpha correlation
- `run.py <sim_dir> <variant> <out_dir>`: trains one variant; `submit.sh`: slurm array over a jobs file
- `summarize.py <runs_dir>`: results table
- `output/` (gitignored): sims, runs, logs, job lists

## Code changes (osl_dynamics)

- `inference/layers.py`: `CoordinatedDropoutLayer`; `LogLikelihoodLossLayer` accepts an optional
  channel mask (marginal LL of masked channels, rescaled by n/n_masked; checked against scipy)
- `models/dynemo.py`: config `coordinated_dropout` (default 0); `private_variance` /
  `learn_private_variance` (a diagonal Psi added to every C_t); `get_private_variance()`

## Reference points (sim_smooth, metrics.py sanity check)

- oracle (true params and alpha): lam_min 1.00, KL 0.00, R2 mean~private 0.002
- static K=1: lam_min 0.96, KL 2.61

## Log

- 2026-10-23 00:14: started. Sims made: sim_smooth, sim_white (200 sessions x 1200, 20 channels, 4 modes;
  variance 23% mean / 34% FC / 43% private; static |r| median 0.11).
- Phase 1 (job 33892097): baseline vs coordinated dropout vs controls. Results: see below.
