Model selection
---------------

These examples choose a model and its number of states/modes with `osl_dynamics.evaluate`:

- **Bi-cross-validation** (`bcv`): hold out sessions and channels and calculate the log-likelihood of the held-out data. Higher is better.
- **Naive cross-validation** (`ncv`): hold out sessions and calculate their free energy.
- **Repeated training** (`repeat`): train on all the data several times.
- **Split-half reproducibility** (`split`): train on two halves of the sessions and compare the states. Lower is more reproducible.

The supported models are the HMM (`hmm`), DyNeMo (`dynemo`), M-DyNeMo (`mdynemo`), Sliding Window Correlation (`swc`), HIVE (`hive`) and DIVE (`dive`). HIVE and DIVE do not support naive cross-validation.

### Files

- `config_minimal.yml`: the smallest config.
- `config_full.yml`: every option, with comments.
- `submit_slurm.sh`: template for running the jobs on a SLURM cluster.
- `bcv_hmm_simulation.py`: bi-cross-validation in Python on simulated data.

### Data

Put the data in a directory with one `.npy` file per session. Each file must have shape (n_samples, n_channels) and every session must have the same channels.

### Running on a laptop or workstation

```
osl-dynamics-evaluate config_minimal.yml
```

This runs every job one after another and saves tables and plots in `<save_dir>/summary`. If it is interrupted, running the same command again skips the jobs that have finished. Jobs that fail are listed in `<save_dir>/summary/failed_jobs.txt` (the error is in `error.txt` in the job's directory) and are retried the next time the command is run.

### Running on a cluster

```
osl-dynamics-evaluate config.yml --init               # prints the number of jobs, N
sbatch --array=1-N submit_slurm.sh config.yml         # one job per array task
osl-dynamics-evaluate config.yml --summarise          # when every job has finished
```

`python -m osl_dynamics.evaluate` can be used instead of `osl-dynamics-evaluate`.

### Results

- `<save_dir>/summary/<mode>.csv`: the result of each job.
- `<save_dir>/summary/<mode>_<model>_<metric>.png`: the results against the number of states/modes.
- `<save_dir>/<model>_state_<n>/<mode>_<i>/`: the parameters of each job. For bi-cross-validation:
  - `Y_train/inf_params`: spatial parameters of channels Y.
  - `X_train/dual_estimates`: spatial parameters of channels X.
  - `X_train/inf_params/alp.pkl` and `Y_test/inf_params/alp.pkl`: time courses of the training and test sessions (first `n_temp_save` folds only).
  - `Y_test/metrics.json`: bi-cross-validated log-likelihood.
- `<save_dir>/<mode>_partition/fold_indices_<i>.json`: the sessions and channels of each fold.
