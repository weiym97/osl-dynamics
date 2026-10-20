"""Run model selection from a config, on a laptop or a cluster.

A config specifies the data, the models, the numbers of states/modes and the
evaluation modes. For example::

    data: path/to/npy_dir
    save_dir: results
    model: hmm
    n_states: [1, 2, 3, 4, 5, 6, 7, 8]
    modes: [bcv]

Each combination of model, number of states and fold (or run) is a job.
Jobs can be run one after another (:code:`run(config)`), or one job per
cluster task (:code:`run(config, index=i)`). :py:func:`summarise` collects
the results into tables and plots.

See :py:func:`normalize_config` for all the options.
"""

import argparse
import copy
import itertools
import json
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import yaml

from osl_dynamics.evaluate.cross_validation import BCV, NCV
from osl_dynamics.evaluate.reproducibility import repeat, split_half
from osl_dynamics.evaluate.splits import CVSplit

_logger = logging.getLogger("osl-dynamics")

#: Evaluation modes in the order their jobs are listed.
MODES = ["bcv", "ncv", "repeat", "split"]

#: Default settings for each mode.
DEFAULT_MODE_SETTINGS = {
    "bcv": {
        "split_row": {
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 10, "train_size": 0.8},
        },
        "split_column": {
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 10, "train_size": 0.5},
        },
        "strategy": "pairing",
        "random_state": 0,
        "n_temp_save": 5,
    },
    "ncv": {
        "split_row": {"method": "KFold", "method_kwargs": {"n_splits": 5}},
    },
    "repeat": {"n_realizations": 3},
    "split": {
        "split_row": {
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 5, "train_size": 0.5},
        },
        "random_state": 0,
    },
}

_MODEL_SPEC_KEYS = ["config_kwargs", "init_kwargs", "fit_kwargs", "spatial_method"]

# Metrics to plot for each mode: column -> (axis label, which value is best)
_SUMMARY_PLOTS = {
    "bcv": {"log_likelihood": ("Bi-cross-validated log-likelihood", "max")},
    "ncv": {
        "free_energy": ("Free energy of held-out sessions", "min"),
        "log_likelihood": ("Log-likelihood of held-out sessions", "max"),
    },
    "repeat": {
        "free_energy": ("Free energy", None),
        "evidence": ("Model evidence", None),
        "log_likelihood": ("Log-likelihood", None),
    },
    "split": {"riemannian_distance": ("Split-half Riemannian distance", None)},
}


def normalize_config(config: Union[str, dict], require_data: bool = True) -> dict:
    """Fill in the defaults of a config.

    Parameters
    ----------
    config : str or dict
        Path to a YAML file, a YAML string or a dict. The keys are:

        - :code:`save_dir` (required): directory to save the results to.
        - :code:`data`: path to the data (e.g. a directory of :code:`.npy`
          files, one per session, with shape (n_samples, n_channels)) or an
          :code:`osl_dynamics.data.Data` object. The data are standardized.
          Use :code:`load_data` instead for other preparation.
        - :code:`load_data`: keyword arguments for
          :py:func:`osl_dynamics.config_api.wrappers.load_data`
          (:code:`inputs`, :code:`kwargs` and :code:`prepare`).
        - :code:`model` (required): a model name (:code:`"hmm"`,
          :code:`"dynemo"`, :code:`"mdynemo"` or :code:`"swc"`), a list of
          names, or a dict of names to settings. The settings can contain
          :code:`config_kwargs`, :code:`init_kwargs`, :code:`fit_kwargs` and
          :code:`spatial_method` (or :code:`infer_spatial`). Other keys are
          treated as :code:`config_kwargs`.
        - :code:`n_states` (required): number of states/modes, or a list.
          This is :code:`n_modes` for DyNeMo and M-DyNeMo.
        - :code:`modes`: list of modes (:code:`"bcv"`, :code:`"ncv"`,
          :code:`"repeat"` and :code:`"split"`) or a dict of modes to
          settings. Defaults to :code:`["bcv"]`. :code:`mode` is also
          accepted. The settings of :code:`"bcv"`, :code:`"ncv"` and
          :code:`"split"` are :code:`split_row`, :code:`split_column`,
          :code:`strategy` and other keyword arguments for
          :py:class:`CVSplit` (e.g. :code:`random_state`); the number of
          samples to split defaults to the number of sessions/channels.
          :code:`"bcv"` also has :code:`n_temp_save` and :code:`"repeat"`
          has :code:`n_realizations`. See :code:`DEFAULT_MODE_SETTINGS`.

        A config returned by this function (or saved as
        :code:`config_root.yaml`) is also accepted.
    require_data : bool, optional
        Should we raise an error if the config has no data?

    Returns
    -------
    config : dict
        Config with keys :code:`save_dir`, :code:`load_data`, :code:`data`,
        :code:`models`, :code:`n_states` and :code:`modes`.
    """
    from osl_dynamics.config_api.pipeline import load_config

    if isinstance(config, dict):
        # Do not copy a Data object
        data = config.get("data")
        config = {key: value for key, value in config.items() if key != "data"}
        config = copy.deepcopy(config)
        if data is not None:
            config["data"] = data
    else:
        config = load_config(config)

    if "save_dir" not in config:
        raise ValueError("save_dir must be passed.")

    normalized = {
        "save_dir": str(config["save_dir"]).rstrip("/"),
        "load_data": None,
        "data": None,
    }

    # Data
    data = config.get("data")
    if config.get("load_data") is not None:
        normalized["load_data"] = config["load_data"]
    elif isinstance(data, (str, list)):
        normalized["load_data"] = {"inputs": data, "prepare": {"standardize": {}}}
    elif data is not None:
        normalized["data"] = data
    elif require_data:
        raise ValueError("data or load_data must be passed.")

    # Models
    models = config.get("models", config.get("model"))
    if models is None:
        raise ValueError("model must be passed.")
    if isinstance(models, str):
        models = {models: {}}
    elif isinstance(models, list):
        models = {model: {} for model in models}
    normalized["models"] = {
        model.lower(): _normalize_model_spec(spec) for model, spec in models.items()
    }

    # Number of states/modes
    if "n_states" not in config:
        raise ValueError("n_states must be passed.")
    n_states = config["n_states"]
    n_states = [n_states] if np.isscalar(n_states) else list(n_states)
    normalized["n_states"] = [int(n) for n in n_states]

    # Modes
    modes = config.get("modes", config.get("mode", ["bcv"]))
    if isinstance(modes, str):
        modes = [modes]
    if isinstance(modes, list):
        modes = {mode: config.get(mode, {}) for mode in modes}
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Options are {MODES}.")
    normalized["modes"] = {
        mode: _deep_update(DEFAULT_MODE_SETTINGS[mode], modes[mode] or {})
        for mode in MODES
        if mode in modes
    }

    bcv = normalized["modes"].get("bcv", {})
    if str(bcv.pop("cv_variant", 1)) != "1":
        raise ValueError("Only cv_variant=1 is supported.")

    return normalized


def get_jobs(config: dict) -> List[Tuple[str, int, str, int]]:
    """List the jobs in a config.

    Parameters
    ----------
    config : dict
        Config from :py:func:`normalize_config`.

    Returns
    -------
    jobs : list of tuple
        :code:`(model, n_states, mode, i)` for each job, where :code:`i`
        is the fold (or run) number, starting at 1.
    """
    mode_jobs = []
    for mode, settings in config["modes"].items():
        n = _n_mode_jobs(mode, settings)
        mode_jobs += [(mode, i) for i in range(1, n + 1)]
    return [
        (model, n_states, mode, i)
        for model, n_states, (mode, i) in itertools.product(
            config["models"], config["n_states"], mode_jobs
        )
    ]


def init(config: Union[str, dict], data=None) -> int:
    """Save the config and the folds of each mode.

    This function creates:

    - :code:`<save_dir>/config_root.yaml`: the config.
    - :code:`<save_dir>/config_list.csv`: the list of jobs.
    - :code:`<save_dir>/<mode>_partition/fold_indices_<i>.json`: the folds
      of each mode.

    Parameters
    ----------
    config : str or dict
        Config (see :py:func:`normalize_config`).
    data : osl_dynamics.data.Data, optional
        Prepared data. Only needed for the number of sessions and channels,
        and loaded from the config if not passed.

    Returns
    -------
    n_jobs : int
        Number of jobs.
    """
    config = normalize_config(config)
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    loaded_data = None
    data = config["data"] if data is None else data
    if _needs_data_dims(config) and data is None:
        data = loaded_data = _load_data(config)
    _write_partitions(config, data)
    if loaded_data is not None:
        loaded_data.delete_dir()

    _write_config(config)
    jobs = get_jobs(config)
    _write_job_list(jobs, f"{save_dir}/config_list.csv")
    _logger.info(f"{len(jobs)} jobs in {save_dir}/config_list.csv")
    return len(jobs)


def run(
    config: Union[str, dict],
    index: Optional[int] = None,
    overwrite: bool = False,
) -> Optional[dict]:
    """Run model selection.

    Parameters
    ----------
    config : str or dict
        Config (see :py:func:`normalize_config`).
    index : int, optional
        Job to run, starting at 1 (e.g. a cluster array task ID). If
        :code:`None`, every job is run one after another and the results
        are summarised.
    overwrite : bool, optional
        Should we re-run jobs that have already finished?

    Returns
    -------
    summary : dict or None
        Output of :py:func:`summarise` if :code:`index=None`.
    """
    config = normalize_config(config)
    jobs = get_jobs(config)
    if index is not None and not 1 <= index <= len(jobs):
        raise ValueError(f"index must be between 1 and {len(jobs)}.")

    os.makedirs(config["save_dir"], exist_ok=True)
    selected = jobs if index is None else [jobs[index - 1]]
    pending = [job for job in selected if overwrite or not _job_finished(config, *job)]
    if not os.path.exists(f"{config['save_dir']}/config_root.yaml"):
        _write_config(config)

    if pending:
        data = config["data"]
        loaded = data is None
        if loaded:
            data = _load_data(config)
        try:
            # A single job may only create folds that every job would create
            # identically, otherwise the folds must be created with init first
            _write_partitions(config, data, require_deterministic=index is not None)
            for job in pending:
                run_job(config, data, *job, overwrite=overwrite)
        finally:
            if loaded:
                data.delete_dir()
    else:
        _logger.info("All jobs have finished.")

    if index is None:
        return summarise(config)


def run_job(
    config: dict,
    data,
    model: str,
    n_states: int,
    mode: str,
    i: int,
    overwrite: bool = False,
) -> None:
    """Run one job.

    Parameters
    ----------
    config : dict
        Config from :py:func:`normalize_config`.
    data : osl_dynamics.data.Data
        Prepared data.
    model : str
        Model name.
    n_states : int
        Number of states/modes.
    mode : str
        Evaluation mode.
    i : int
        Fold (or run) number, starting at 1.
    overwrite : bool, optional
        Should we re-run the job if it has already finished?
    """
    job_dir = _job_dir(config, model, n_states, mode, i)
    if _job_result(job_dir, mode) is not None and not overwrite:
        _logger.info(f"Already finished: {job_dir}")
        return

    _logger.info(f"Running {model}, n_states={n_states}, {mode}_{i}: {job_dir}")
    spec = config["models"][model]
    settings = config["modes"][mode]
    config_kwargs = dict(spec["config_kwargs"])
    config_kwargs.pop("n_channels", None)
    config_kwargs["n_modes" if model in ["dynemo", "mdynemo"] else "n_states"] = (
        n_states
    )
    kwargs = {
        "config_kwargs": config_kwargs,
        "init_kwargs": spec["init_kwargs"],
        "fit_kwargs": spec["fit_kwargs"],
    }

    os.makedirs(job_dir, exist_ok=True)
    with open(f"{job_dir}/batch_config.yaml", "w") as file:
        yaml.safe_dump(
            _to_yaml(
                {"model": model, "mode": f"{mode}_{i}", **kwargs, "settings": settings}
            ),
            file,
            default_flow_style=False,
        )

    fold = f"{config['save_dir']}/{mode}_partition/fold_indices_{i}.json"
    if mode == "bcv":
        n_temp_save = settings.get("n_temp_save")
        BCV(model, spatial_method=spec["spatial_method"], **kwargs).validate(
            data,
            fold,
            job_dir,
            save_temporal=n_temp_save is None or i <= n_temp_save,
        )
    elif mode == "ncv":
        NCV(model, **kwargs).validate(data, fold, job_dir)
    elif mode == "repeat":
        repeat(model, data, job_dir, **kwargs)
    elif mode == "split":
        split_half(model, data, fold, job_dir, **kwargs)


def summarise(config: Union[str, dict]) -> dict:
    """Collect the results of every job into tables and plots.

    This function creates :code:`<save_dir>/summary/<mode>.csv` and a plot
    of the results against the number of states/modes for each model and
    mode, e.g. :code:`<save_dir>/summary/bcv_hmm.png`.

    Parameters
    ----------
    config : str or dict
        Config (see :py:func:`normalize_config`), or the :code:`save_dir`
        of a run that has a :code:`config_root.yaml`.

    Returns
    -------
    summary : dict
        A :code:`pandas.DataFrame` for each mode.
    """
    import pandas as pd

    if isinstance(config, str) and os.path.isdir(config):
        config = f"{config}/config_root.yaml"
    config = normalize_config(config, require_data=False)
    summary_dir = f"{config['save_dir']}/summary"
    os.makedirs(summary_dir, exist_ok=True)

    rows = {mode: [] for mode in config["modes"]}
    n_missing = 0
    for model, n_states, mode, i in get_jobs(config):
        result = _job_result(_job_dir(config, model, n_states, mode, i), mode)
        if result is None:
            if not (mode == "ncv" and n_states == 1):
                n_missing += 1
            continue
        rows[mode].append({"model": model, "n_states": n_states, "run": i, **result})
    if n_missing > 0:
        _logger.warning(f"{n_missing} jobs have not finished.")

    summary = {}
    for mode, mode_rows in rows.items():
        df = pd.DataFrame(mode_rows)
        summary[mode] = df
        df.to_csv(f"{summary_dir}/{mode}.csv", index=False)
        if len(df) == 0:
            continue
        for model, model_df in df.groupby("model"):
            for column, (label, best) in _SUMMARY_PLOTS[mode].items():
                if column in model_df:
                    _plot_summary(
                        model_df,
                        column,
                        label,
                        best,
                        title=f"{model} ({mode})",
                        filename=f"{summary_dir}/{mode}_{model}_{column}.png",
                    )
    _logger.info(f"Summary saved to {summary_dir}")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    """Command line interface for :py:mod:`osl_dynamics.evaluate`."""
    parser = argparse.ArgumentParser(
        description=(
            "Model selection for osl-dynamics: bi-cross-validation (bcv), "
            "naive cross-validation (ncv), repeated training (repeat) and "
            "split-half reproducibility (split). Without --init, --index or "
            "--summarise, every job is run one after another."
        )
    )
    parser.add_argument("config", type=str, help="Path to the YAML config.")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Save the folds and the job list, then print the number of jobs.",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Run one job (starting at 1), e.g. $SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--summarise",
        action="store_true",
        help="Collect the results of finished jobs.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-run finished jobs."
    )
    parser.add_argument("--restrict", type=str, help="GPU to use. Optional.")
    args = parser.parse_args(argv)

    if args.restrict is not None:
        from osl_dynamics.inference import tf_ops

        tf_ops.select_gpu(int(args.restrict))
        tf_ops.gpu_growth()

    if args.init:
        print(init(args.config))
    elif args.summarise:
        summarise(args.config)
    else:
        run(args.config, index=args.index, overwrite=args.overwrite)


def _normalize_model_spec(spec: Optional[dict]) -> dict:
    spec = dict(spec or {})
    if "infer_spatial" in spec:
        spec["spatial_method"] = spec.pop("infer_spatial")
    config_kwargs = dict(spec.pop("config_kwargs", {}) or {})
    normalized = {key: spec.pop(key, None) for key in _MODEL_SPEC_KEYS[1:]}
    # Any other keys are config_kwargs
    config_kwargs.update(spec)
    normalized["config_kwargs"] = config_kwargs
    return normalized


def _deep_update(default: dict, override: dict) -> dict:
    """Recursively update a copy of default with override."""
    updated = copy.deepcopy(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = _deep_update(updated[key], value)
        else:
            updated[key] = copy.deepcopy(value)
    return updated


def _n_mode_jobs(mode: str, settings: dict) -> int:
    """Number of jobs (folds or runs) of a mode."""
    if mode == "repeat":
        return int(settings["n_realizations"])

    n_splits = []
    for key in ["split_row", "split_column"]:
        if settings.get(key):
            n_splits.append(int(settings[key]["method_kwargs"]["n_splits"]))
    if settings.get("strategy", "combination") == "pairing":
        return min(n_splits)
    return int(np.prod(n_splits))


def _needs_data_dims(config: dict) -> bool:
    """Do we need the data to know the number of sessions/channels?"""
    for mode, settings in config["modes"].items():
        for key in ["split_row", "split_column"]:
            if settings.get(key) and "n_samples" not in settings[key]:
                return True
    return False


def _load_data(config: dict):
    from osl_dynamics.config_api.wrappers import load_data

    load_data_kwargs = copy.deepcopy(config["load_data"])
    kwargs = load_data_kwargs.setdefault("kwargs", {}) or {}
    kwargs.setdefault("store_dir", f"{config['save_dir']}/tmp")
    load_data_kwargs["kwargs"] = kwargs
    return load_data(**load_data_kwargs)


def _write_partitions(
    config: dict, data=None, require_deterministic: bool = False
) -> None:
    """Save the folds of each mode if they do not exist."""
    for mode, settings in config["modes"].items():
        if mode == "repeat":
            continue
        partition_dir = f"{config['save_dir']}/{mode}_partition"
        n_folds = _n_mode_jobs(mode, settings)
        if all(
            os.path.exists(f"{partition_dir}/fold_indices_{i}.json")
            for i in range(1, n_folds + 1)
        ):
            continue

        if require_deterministic and not _is_deterministic(settings):
            raise ValueError(
                f"The {mode} folds have not been created and are random "
                "(no random_state). Run with --init (or call init) first."
            )

        splitter_kwargs = copy.deepcopy(settings)
        splitter_kwargs.pop("n_temp_save", None)
        for key, size in [("split_row", "n_sessions"), ("split_column", "n_channels")]:
            if not splitter_kwargs.get(key):
                continue
            actual = None if data is None else getattr(data, size)
            if "n_samples" not in splitter_kwargs[key]:
                if actual is None:
                    raise ValueError(f"The data is needed to split the {size}.")
                splitter_kwargs[key]["n_samples"] = actual
            elif actual is not None and splitter_kwargs[key]["n_samples"] != actual:
                raise ValueError(
                    f"{mode}: {key}.n_samples={splitter_kwargs[key]['n_samples']} "
                    f"but the data has {size}={actual}."
                )
        if mode != "bcv":
            splitter_kwargs.pop("split_column", None)
        CVSplit(**splitter_kwargs).save(partition_dir)


def _is_deterministic(settings: dict) -> bool:
    """Do the splitters of a mode always give the same folds?"""
    if settings.get("random_state") is not None:
        return True
    for key in ["split_row", "split_column"]:
        split = settings.get(key)
        if not split:
            continue
        shuffle = split.get("method_kwargs", {}).get("shuffle", False)
        if split.get("method", "ShuffleSplit") != "KFold" or shuffle:
            return False
    return True


def _write_config(config: dict) -> None:
    config_to_save = {key: value for key, value in config.items() if key != "data"}
    if config["data"] is not None:
        config_to_save["load_data"] = None
    with open(f"{config['save_dir']}/config_root.yaml", "w") as file:
        yaml.safe_dump(
            _to_yaml(config_to_save), file, default_flow_style=False, sort_keys=False
        )


def _to_yaml(obj):
    """Convert numpy arrays and scalars so a config can be saved as YAML."""
    if isinstance(obj, dict):
        return {key: _to_yaml(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _write_job_list(jobs: list, filename: str) -> None:
    import pandas as pd

    pd.DataFrame(
        [(model, n_states, f"{mode}_{i}") for model, n_states, mode, i in jobs],
        columns=["model_index", "n_states", "mode_index"],
    ).to_csv(filename, index=True)


def _job_dir(config: dict, model: str, n_states: int, mode: str, i: int) -> str:
    return f"{config['save_dir']}/{model}_state_{n_states}/{mode}_{i}"


def _job_finished(config: dict, model: str, n_states: int, mode: str, i: int) -> bool:
    if mode == "ncv" and n_states == 1:
        return True  # naive cross-validation is skipped for one state
    return _job_result(_job_dir(config, model, n_states, mode, i), mode) is not None


def _job_result(job_dir: str, mode: str) -> Optional[dict]:
    """Result of a finished job, or None if it has not finished."""
    if mode == "bcv":
        return _read_json(f"{job_dir}/Y_test/metrics.json")
    if mode == "ncv":
        free_energy = _read_json(f"{job_dir}/ncv_free_energy.json")
        if free_energy is not None:
            return {"free_energy": free_energy[0]}
        return _read_json(f"{job_dir}/metrics.json")
    if mode == "repeat":
        metrics = _read_json(f"{job_dir}/metrics/metrics.json")
        if metrics is None and os.path.exists(f"{job_dir}/inf_params/covs.npy"):
            return {}  # static model
        return metrics
    result = _read_json(f"{job_dir}/reproducibility.json")
    if result is not None:
        return {"riemannian_distance": result["riemannian_distance"]}
    return None


def _read_json(filename: str):
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as file:
        return json.load(file)


def _plot_summary(
    df,
    column: str,
    label: str,
    best: Optional[str],
    title: str,
    filename: str,
) -> None:
    """Box plot of a metric against the number of states/modes."""
    import matplotlib.pyplot as plt

    df = df.dropna(subset=[column])
    if len(df) == 0:
        return
    n_states = sorted(df["n_states"].unique())
    values = [df.loc[df["n_states"] == n, column].to_numpy() for n in n_states]
    means = np.array([np.mean(v) for v in values])

    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(n_states) + 2), 4))
    ax.boxplot(values, positions=range(len(n_states)))
    ax.plot(range(len(n_states)), means, "o-", color="tab:blue", label="Mean")
    if best is not None:
        i = int(np.argmax(means) if best == "max" else np.argmin(means))
        ax.plot(i, means[i], "*", color="tab:red", markersize=14, label="Best")
    ax.set_xticks(range(len(n_states)))
    ax.set_xticklabels(n_states)
    ax.set_xlabel("Number of states/modes")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
