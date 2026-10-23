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
- Phase 1 (job 33892097, K=4, 30 epochs):
  - Mechanism confirmed. Baseline DyNeMo+means on sim_smooth: lam_min 0.57-0.61, R2 mean~private 0.09,
    KL 1.4-1.5. The same model on sim_white: lam 0.98, R2 0.01, KL 0.36.
  - cd50 (3 seeds): lam 0.94-0.95, R2 0.01-0.02, KL 0.98-1.36. But seeds 1 and 2 have a dead mode
    (FO 0.00/0.01) plus a merged mode, giving alpha corr 0.62/0.69 against 0.91 for the baseline.
  - HMM: lam 0.86, KL 1.7. DyNeMo without means: lam 0.95, KL 1.9 (it can't model the mean).
- Phase 2 (jobs 33892473-6):
  - Benchmark setting: the baseline collapse grows with training. sim_smooth K=4 at 100 epochs gives
    lam 0.13, 3 dirs < 0.5, R2 0.24, KL 12.4 (static = 2.6). K=2 at 30 epochs: lam 0.21, KL 4.6.
    K=8: lam 0.39, KL 3.0. phi = 0.95 is slightly worse (lam 0.10, KL 16.6 at 100 epochs).
    => Benchmark = sim_smooth, K=4, 100 epochs.
  - cd50 holds everywhere: lam 0.93-0.96, R2 0.01 at K = 2/4/8 and 100 epochs (KL 0.79). The issue is
    dead modes on some seeds.
  - Psi floor (30 epochs): learned Psi shrinks to 0.14 against a true 0.43 (not identifiable), so no fix
    (lam 0.59). Fixed Psi works: oracle gives lam 0.98, KL 0.79, alpha 0.92; FA(8) (estimate 0.56) gives
    lam 1.01, KL 0.98, alpha 0.92. cd50 + learned Psi: Psi shrinks, and there are dead modes again.
- Phase 3 (jobs 33892834, 33892835, all 100 epochs): seed robustness of the baseline failure, the
  white-noise control at 100 epochs, cd25/cd50 seeds, n_init=10 for cd50 (dead modes), fixed-Psi
  oracle/FA(4, 8, 12) at 100 epochs, HMM at 100 epochs.

- Phase 3 results (benchmark: sim_smooth, K=4, 100 epochs; oracle KL 0, static 2.6, HMM 1.64):
  - Baseline, 3 seeds: lam 0.13-0.15, R2 0.24, KL 11.8-12.4 (very consistent). White control at
    100 epochs: lam 0.81, KL 0.51.
  - **cd25, 3 seeds: lam 0.94-0.95, R2 0.01, KL 0.68-0.69, alpha 0.91, all healthy.** Best overall.
  - cd50: dead modes with n_init 5 (KL 0.79-1.21); n_init 10 fixes it (2/2 seeds: KL 0.79, alpha 0.90).
  - Fixed Psi at 100 epochs only bounds the damage. The mean still copies (R2 0.08-0.10). Oracle:
    lam 0.81, KL 1.0. FA8: lam 0.89-0.94, KL 0.96-1.31. FA4/FA12: KL 1.21/1.06. Depends on the FA rank.
  - Pattern recovery (`recovery.py`, matched modes): every model recovers mean/FC patterns well,
    collapsed baseline included (mean 0.97, FC 0.96), so the collapse doesn't show in mode maps.
    cd25: mean 1.00, FC 0.96, FC contrast 0.97. Psi floor: FC 0.98-0.99. HMM: FC 0.74.
- Phase 4 (jobs 33893063-5, 100 epochs):
  - K=2: baseline lam 0.048, KL 20.2 (catastrophic, like HCP K=2); cd25 (2 seeds) lam 0.955, KL 1.72.
  - K=8: baseline lam 0.13, 5 dirs < 0.5, KL 20.6; cd25 lam 0.88/0.94, KL 1.04/0.71.
  - smooth95: cd25 KL 0.69-0.70 (baseline 16.6). cd10 on smooth: KL 0.72-0.73, lam 0.94-0.95.
  - sim_white (nothing to fix): cd25 KL 0.55-0.56 against baseline 0.50-0.51, a negligible cost.
    Mean/FC recovery 0.95-1.00 throughout.
- HCP-scale check (jobs 33893283-4): sim_hcp = 500 sessions x 4800, 25 channels, run with the real
  ICA_50_mean training config (batch 64, 30 epochs, n_init 10). Baseline vs cd25 at K=4 (2 seeds)
  and K=2.
  - K=4 baseline: lam 0.11-0.12, 3 dirs < 0.5, R2 0.18-0.19, KL 15.3-16.2. cd25: lam 0.95-0.96,
    R2 0.00, KL 0.71, alpha 0.94. Mean/FC recovery is 0.96-1.00 for both.
  - K=2 baseline: lam 0.089, KL 12.1. cd25: lam 0.96, KL 2.12.
  - The conclusions hold at HCP scale with the real training config (~1 h per run on 4 CPUs).
- 02:45: first work block done (~2.5 h). Next, after a 1 h rest: check whether M-DyNeMo (the MAGE-like
  model) has the same failure on the benchmark, plus a State-DyNeMo (discrete alpha) control.

## Conclusions so far

1. The simulation reproduces the HCP failure, provided the private noise is temporally smooth.
   White private noise doesn't. Collapse grows with training and is worst at small K.
2. Coordinated channel dropout (rate 0.1-0.25) fixes it at every K tested. Copying goes from
   R2 0.24 to 0.01, lam_min from ~0.1 to ~0.95, and KL to truth from 12-20 to 0.7 (K=4). Mode
   mean/FC recovery is kept. Rate 0.5 also works but gives dead modes unless n_init is raised.
3. A fixed diagonal private-noise floor only bounds the damage: the mean keeps copying, and the
   result depends on the FA rank. A learned floor shrinks and does nothing (not identifiable).
4. Collapse doesn't show in mode maps (pattern recovery is good even for the collapsed baseline).
   Only likelihood-type checks catch it, which is consistent with BCV.

## Notes for moving to real data later

- With a Psi floor, the BCV path must include Psi. `dual_estimation` (it rebuilds the observation model
  from the "mix_covs" and "ll_loss" layers) and `get_posterior_expected_log_likelihood` currently ignore it.
- Coordinated dropout needs no downstream changes, since evaluation runs with training=False.
