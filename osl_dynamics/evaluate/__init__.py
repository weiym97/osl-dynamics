"""Model selection and evaluation.

This subpackage helps choose a model and its number of states/modes with:

- **Bi-cross-validation** (:code:`"bcv"`): hold out sessions and channels
  and calculate the log-likelihood of held-out data (higher is better).
  See :py:class:`~osl_dynamics.evaluate.cross_validation.BCV`.
- **Naive cross-validation** (:code:`"ncv"`): hold out sessions and
  calculate their free energy. See
  :py:class:`~osl_dynamics.evaluate.cross_validation.NCV`.
- **Repeated training** (:code:`"repeat"`): train on all the data several
  times. See :py:func:`~osl_dynamics.evaluate.reproducibility.repeat`.
- **Split-half reproducibility** (:code:`"split"`): train on two halves of
  the sessions and compare the states (lower is more reproducible). See
  :py:func:`~osl_dynamics.evaluate.reproducibility.split_half`.

The supported models are the HMM, DyNeMo, M-DyNeMo, SWC, HIVE and DIVE.
HIVE and DIVE support bi-cross-validation, repeated training and split-half
reproducibility (of the group covariances), but not naive cross-validation.

Command line usage
------------------
Put the data in a directory with one :code:`.npy` file per session, each
with shape (n_samples, n_channels), and write a config::

    data: path/to/npy_dir
    save_dir: results
    model: hmm
    n_states: [1, 2, 3, 4, 5, 6, 7, 8]
    modes: [bcv]

Then run every job one after another and summarise the results::

    osl-dynamics-evaluate config.yml

The results are in :code:`results/summary`. If the run is interrupted,
running the same command again skips the jobs that have finished.

On a cluster, create the folds, run one job per array task and then
summarise::

    osl-dynamics-evaluate config.yml --init        # prints the number of jobs
    osl-dynamics-evaluate config.yml --index $SLURM_ARRAY_TASK_ID
    osl-dynamics-evaluate config.yml --summarise

:code:`python -m osl_dynamics.evaluate` can be used instead of
:code:`osl-dynamics-evaluate`. See
:py:func:`~osl_dynamics.evaluate.batch.normalize_config` for all the
options in the config, e.g. model hyperparameters and the number of folds.

Python usage
------------
The same in Python::

    from osl_dynamics import evaluate

    summary = evaluate.run(
        {
            "data": "path/to/npy_dir",
            "save_dir": "results",
            "model": "hmm",
            "n_states": range(1, 9),
            "modes": ["bcv"],
        }
    )
    print(summary["bcv"])

:code:`"data"` can also be a prepared :code:`osl_dynamics.data.Data`
object. Individual folds can be run with the classes and functions below,
e.g.::

    from osl_dynamics.evaluate import BCV, CVSplit

    folds = CVSplit(
        split_row={
            "n_samples": data.n_sessions,
            "method_kwargs": {"n_splits": 10, "train_size": 0.8},
        },
        split_column={
            "n_samples": data.n_channels,
            "method_kwargs": {"n_splits": 10, "train_size": 0.5},
        },
        strategy="pairing",
        random_state=0,
    )
    bcv = BCV(
        "hmm",
        config_kwargs={"n_states": 6, "learn_means": False, "learn_covariances": True},
    )
    log_likelihoods = bcv.run(data, folds, output_dir="results/hmm_state_6")

Modules
-------
- :py:mod:`~osl_dynamics.evaluate.batch` — :py:func:`run`, :py:func:`init`
  and :py:func:`summarise` for running from a config.
- :py:mod:`~osl_dynamics.evaluate.cross_validation` — :py:class:`BCV` and
  :py:class:`NCV`.
- :py:mod:`~osl_dynamics.evaluate.reproducibility` — :py:func:`repeat` and
  :py:func:`split_half`.
- :py:mod:`~osl_dynamics.evaluate.splits` — :py:class:`CVSplit` for
  splitting sessions and channels.
- :py:mod:`~osl_dynamics.evaluate.model_ops` — model-specific operations.

Python example scripts
----------------------
- `Model selection <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/evaluate>`_
"""

from osl_dynamics.evaluate.batch import init, run, summarise
from osl_dynamics.evaluate.cross_validation import BCV, NCV
from osl_dynamics.evaluate.reproducibility import repeat, split_half
from osl_dynamics.evaluate.splits import CVSplit
