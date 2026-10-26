"""Collect metrics.json files from a runs directory into one table.

Usage: python summarize.py <runs_dir> [<runs_dir> ...]
"""

import glob
import json
import sys

import numpy as np


def fmt(values):
    return " ".join("  - " if v is None or np.isnan(v) else f"{v:4.2f}" for v in values)


def variant(r):
    v = r["init"]
    if r["init"] in ["nfindr", "nfindr_allch", "als"] and r.get("als_window", 20) != 20:
        v += f"-w{r['als_window']}"
    if r.get("nfindr_expand", 1) != 1:
        v += f"-x{r['nfindr_expand']:g}"
    if r.get("sim_config"):
        # Training settings are the simulation's own; only flag what differs from them
        v += " revive" if r.get("revive") else ""
        v += f" split{r['split_rounds']}" if r.get("split_rounds") else ""
        v += f" reset{r['reset_rounds']}" if r.get("reset_rounds") else ""
        v += " prune" if r.get("prune_from_plus1") else ""
        return v + f" ep{r['n_epochs']} dec{r['lr_decay']:g}"
    if r.get("revive"):
        v += " revive"
    if r.get("learning_rate", 0.01) != 0.01:
        v += f" lr{r['learning_rate']:g}"
    if r.get("fixed_temperature"):
        v += " Tfix"
    if r.get("init_epochs", 2) != 2:
        v += f" ie{r['init_epochs']}"
    if r["n_epochs"] != 200:
        v += f" ep{r['n_epochs']}"
    if r.get("init_select", "loss") != "loss":
        v += f" sel-{r['init_select']}"
    if r.get("n_init", 10) != 10:
        v += f" ni{r['n_init']}"
    if r.get("no_kl_annealing"):
        v += " noKLann"
    if r.get("kl_beta0", 1) != 1:
        v += f" b{r['kl_beta0']:g}"
    if r.get("lr_decay", 0.1) != 0.1:
        v += f" dec{r['lr_decay']:g}"
    if r.get("batch_size", 16) != 16:
        v += f" bs{r['batch_size']}"
    if r.get("smem_rounds", 0):
        v += f" smem{r['smem_rounds']}"
    if r.get("reset_rounds", 0):
        v += f" reset{r['reset_rounds']}"
    return v


def main():
    rows = []
    for d in sys.argv[1:]:
        for f in sorted(glob.glob(f"{d}/*/metrics.json")):
            r = json.load(open(f))
            rows.append(r)
    rows.sort(key=lambda r: (r["sim"], r["regime"], r["fold"], r["n_modes"], variant(r), r["free_energy"]))
    print(f"{'sim':<24}{'regime':<7}{'fold':>4}{'K':>3}{'variant':>16}{'seed':>5}"
          f"{'free_energy':>13}  {'alpha r per true mode':<30}{'cov r per true mode':<30}{'Ct err':>7}{'dead':>5}{'min':>5}")
    for r in rows:
        print(f"{r['sim']:<24}{r['regime']:<7}{r['fold']:>4}{r['n_modes']:>3}{variant(r):>16}{r['seed']:>5}"
              f"{r['free_energy']:>13.3f}  {fmt(r['alpha_r']):<30}{fmt(r['cov_r']):<30}"
              f"{r['ct_rel_err']:>7.3f}{r['n_dead']:>5}{r['train_minutes']:>5.0f}")


if __name__ == "__main__":
    main()
