"""Set up results_final/simulation/<name>_<suffix>/ for BCV runs with N-FINDR initialisation.

For each simulation this writes, next to the original results folder:
- config_simulation.yaml: the original DyNeMo training settings plus nfindr_kwargs (and any
  training overrides), K = 1..n_states_max, n_bcv BCV folds and n_repeat repeats
- bcv_partition/fold_indices_{1..n_bcv}.json: the original folds are copied (so the comparison with
  the original results is paired fold by fold); if the original has fewer than n_bcv folds the rest
  are drawn with CVSplit, with the original split settings and a fixed random_state
- config_list.csv: the array index -> (model, K, mode) table that IndexParser would make
  (written here so that IndexParser does not draw new random partitions)

With --baseline it writes <name>_baseline/ instead: the same set-up without nfindr_kwargs, for
originals that were run by an older BCV pipeline (dynemo_fair, dynemo_fair_large_act).

The defaults reproduce the first trial (2026-10-23): K 1-8, 5 folds, 3 repeats, window 20.

Usage (from the repository root):
    python examples/simulation/dynemo_mode_recovery/make_nfindr_configs.py [--baseline] [sim ...]
    python examples/simulation/dynemo_mode_recovery/make_nfindr_configs.py dynemo_iid dynemo_iid_real_2 \
        --n_states_max 16 --n_bcv 100 --window dynemo_iid_real_2=40 --n_epochs 60 \
        --n_kl_annealing_epochs 30 --overwrite
"""

import argparse
import glob
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from itertools import product

import pandas as pd
import yaml

SIMULATIONS = [
    "dynemo_fair",
    "dynemo_fair_large_act",
    "dynemo_hrf",
    "dynemo_iid",
    "dynemo_iid_real",
    "dynemo_iid_real_2",
    "dynemo_iid_real_3",
]
ROOT = "./results_final/simulation"


def write_partition(src, dst, bcv, n_bcv, seed=2026):
    """Copy the original folds; draw any missing ones with CVSplit (original settings)."""
    from osl_dynamics.evaluate.cross_validation import CVSplit

    os.makedirs(f"{dst}/bcv_partition")
    n_original = len(glob.glob(f"{src}/bcv_partition/fold_indices_*.json"))
    n_copy = min(n_bcv, n_original)
    for i in range(1, n_copy + 1):
        shutil.copy(f"{src}/bcv_partition/fold_indices_{i}.json", f"{dst}/bcv_partition/")
    n_new = n_bcv - n_copy
    if n_new > 0:
        settings = {
            "split_row": {**bcv["split_row"], "method_kwargs": {**bcv["split_row"]["method_kwargs"], "n_splits": n_new}},
            "split_column": {**bcv["split_column"], "method_kwargs": {**bcv["split_column"]["method_kwargs"], "n_splits": n_new}},
            "strategy": bcv["strategy"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            CVSplit(**settings, random_state=seed).save(tmp)
            for i in range(1, n_new + 1):
                shutil.copy(f"{tmp}/fold_indices_{i}.json", f"{dst}/bcv_partition/fold_indices_{n_copy + i}.json")
    return n_copy, n_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sims", nargs="*", default=SIMULATIONS)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--n_states_max", type=int, default=8)
    parser.add_argument("--n_bcv", type=int, default=5)
    parser.add_argument("--n_repeat", type=int, default=3)
    parser.add_argument("--window", action="append", default=[],
                        help="N-FINDR window, either a number for all sims or sim=number")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--n_kl_annealing_epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an existing destination first (only allowed for *_nfindr folders)")
    args = parser.parse_args()
    suffix = "baseline" if args.baseline else "nfindr"
    n_states = list(range(1, args.n_states_max + 1))

    windows = {}
    for w in args.window:
        if "=" in w:
            sim, value = w.split("=")
            windows[sim] = int(value)
        else:
            windows["*"] = int(w)

    for name in args.sims:
        src = f"{ROOT}/{name}"
        dst = f"{ROOT}/{name}_{suffix}"
        if os.path.exists(dst):
            if not (args.overwrite and dst.endswith("_nfindr")):
                raise FileExistsError(f"{dst} already exists")
            shutil.rmtree(dst)
            print(f"deleted {dst}")
        with open(f"{src}/config_simulation.yaml") as f:
            original = yaml.safe_load(f)

        dynemo = original["model"]["dynemo"]
        config_kwargs = dict(dynemo["config_kwargs"])
        overrides = {k: getattr(args, k) for k in ["n_epochs", "n_kl_annealing_epochs"] if getattr(args, k) is not None}
        config_kwargs.update(overrides)
        window = windows.get(name, windows.get("*", 20))
        bcv = original["mode"]["bcv"]
        note = (f"{'Baseline (no N-FINDR)' if args.baseline else f'N-FINDR initialisation (window {window})'} "
                f"for {name}: DyNeMo only, original training settings"
                + (f" except {overrides}" if overrides else "")
                + f", K 1-{args.n_states_max}, {args.n_bcv} BCV folds, {args.n_repeat} repeats")
        config = {
            "header": {"time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "note": note},
            "load_data": original["load_data"],
            "save_dir": f"{dst}/",
            "spatial_map": original["spatial_map"],
            "model": {"dynemo": {"config_kwargs": config_kwargs, "init_kwargs": dynemo["init_kwargs"]}},
            "n_states": n_states,
            "mode": {
                "bcv": {
                    "split_row": {**bcv["split_row"], "method_kwargs": {**bcv["split_row"]["method_kwargs"], "n_splits": args.n_bcv}},
                    "split_column": {**bcv["split_column"], "method_kwargs": {**bcv["split_column"]["method_kwargs"], "n_splits": args.n_bcv}},
                    "strategy": bcv["strategy"],
                },
                "repeat": {"n_realizations": args.n_repeat},
            },
        }
        if not args.baseline:
            config["model"]["dynemo"]["nfindr_kwargs"] = {"window_length": window, "n_restarts": 10}

        n_copy, n_new = write_partition(src, dst, bcv, args.n_bcv)
        with open(f"{dst}/config_simulation.yaml", "w") as f:
            f.write(f"# Generated by examples/simulation/dynemo_mode_recovery/make_nfindr_configs.py from {src}/config_simulation.yaml\n")
            f.write(f"# BCV folds 1-{n_copy} are copied from {src}/bcv_partition"
                    + (f"; folds {n_copy + 1}-{args.n_bcv} were drawn with CVSplit (random_state 2026)" if n_new else "") + ".\n")
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        modes = [f"bcv_{i}" for i in range(1, args.n_bcv + 1)] + [f"repeat_{i}" for i in range(1, args.n_repeat + 1)]
        df = pd.DataFrame(list(product(["dynemo"], n_states, modes)), columns=["model_index", "n_states", "mode_index"])
        df.to_csv(f"{dst}/config_list.csv", index=True)
        print(f"{dst}: {len(df)} jobs (folds copied {n_copy}, new {n_new}, window {window}, overrides {overrides})")


if __name__ == "__main__":
    main()
