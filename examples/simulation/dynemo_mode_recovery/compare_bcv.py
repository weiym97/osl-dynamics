"""Compare the N-FINDR BCV trial with its baseline, fold by fold.

Baseline: <name>_baseline/ if it exists (same pipeline, no N-FINDR), else the original <name>/
(same folds 1-5, which were copied from it). Reads only pipeline outputs (metrics.json, fitted
alpha/covariances) and the simulation's true parameters; prints summary statistics only.

Usage (from the repository root):
    python examples/simulation/dynemo_mode_recovery/compare_bcv.py [sim ...] [--K_max 8] [--n_folds 5]
"""

import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = "./results_final/simulation"
SIMULATIONS = ["dynemo_fair", "dynemo_fair_large_act", "dynemo_hrf", "dynemo_iid",
               "dynemo_iid_real", "dynemo_iid_real_2", "dynemo_iid_real_3"]
KS = range(1, 9)
FOLDS = range(1, 6)
REPEAT_K = 6


def bcv_ll(folder):
    """{K: {fold: Y_test log-likelihood}}"""
    out = {}
    for K in KS:
        for f in FOLDS:
            path = f"{folder}/dynemo_state_{K}/bcv_{f}/Y_test/metrics.json"
            if os.path.exists(path):
                out.setdefault(K, {})[f] = json.load(open(path))["log_likelihood"]
    return out


def repeat_fe(folder, K):
    fe = []
    for path in sorted(glob.glob(f"{folder}/dynemo_state_{K}/repeat_*/metrics/metrics.json")):
        fe.append(json.load(open(path))["free_energy"])
    return fe


def recovery(folder, name, K=6):
    """Matched alpha r per true mode for each repeat (summary only)."""
    truth_dir = f"./data/node_timeseries/simulation_final/{name}/truth"
    files = sorted(glob.glob(f"{truth_dir}/*_mode_time_course.npy"))
    if not files:
        return []
    out = []
    for rep in sorted(glob.glob(f"{folder}/dynemo_state_{K}/repeat_*/inf_params/alp.pkl")):
        alpha = pickle.load(open(rep, "rb"))
        a_fit = np.concatenate(alpha)
        n = len(alpha[0])
        a_true = np.concatenate([np.load(f)[:n] for f in files[:len(alpha)]])
        k = a_true.shape[1]
        corr = np.nan_to_num(np.corrcoef(a_true.T, a_fit.T)[:k, k:])
        r, c = linear_sum_assignment(-corr)
        out.append(np.sort(corr[r, c])[::-1])
    return out


def main():
    global KS, FOLDS
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("sims", nargs="*", default=SIMULATIONS)
    parser.add_argument("--K_max", type=int, default=8)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()
    KS, FOLDS = range(1, args.K_max + 1), range(1, args.n_folds + 1)
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    for name in args.sims:
        new = f"{ROOT}/{name}_nfindr"
        base = f"{ROOT}/{name}_baseline" if os.path.exists(f"{ROOT}/{name}_baseline") else f"{ROOT}/{name}"
        ll_new, ll_base = bcv_ll(new), bcv_ll(base)
        print(f"\n===== {name}: N-FINDR vs {os.path.basename(base)} (BCV folds 1-{len(FOLDS)}, Y_test LL per sample)")
        print(f"{'K':>3} {'n':>2} {'N-FINDR':>10} {'baseline':>10} {'diff':>8} {'sd':>6} {'better':>7}")
        for K in KS:
            common = sorted(set(ll_new.get(K, {})) & set(ll_base.get(K, {})))
            if not common:
                print(f"{K:>3}  - (N-FINDR folds {len(ll_new.get(K, {}))}, baseline folds {len(ll_base.get(K, {}))})")
                continue
            a = np.array([ll_new[K][f] for f in common])
            b = np.array([ll_base[K][f] for f in common])
            d = a - b
            print(f"{K:>3} {len(common):>2} {a.mean():>10.4f} {b.mean():>10.4f} {d.mean():>+8.4f} {d.std():>6.4f} {np.mean(d > 0):>7.2f}")
        # Curves on the folds done for every K that N-FINDR has (so both curves use the same folds)
        Ks_done = [K for K in KS if len(ll_new.get(K, {})) >= 0.9 * len(FOLDS)]
        folds_all = sorted(set.intersection(*[set(ll_new[K]) & set(ll_base.get(K, {})) for K in Ks_done])) if Ks_done else []
        print(f"  curves over K {Ks_done[0] if Ks_done else '-'}-{Ks_done[-1] if Ks_done else '-'} on the {len(folds_all)} folds done for all of them")
        for label, ll in [("N-FINDR", ll_new), ("baseline", ll_base)]:
            means = {K: np.mean([ll[K][f] for f in folds_all]) for K in Ks_done if folds_all}
            if means:
                ref = max(means, key=means.get)
                gains = " ".join(f"K{K}:{means[K] - means[ref]:+.3f}" for K in sorted(means))
                print(f"  {label:>8} curve (relative to its best K={ref}): {gains}")
        fe_new, fe_base = repeat_fe(new, 6), repeat_fe(base, 6)
        if fe_new or fe_base:
            print(f"  K=6 repeat free energy: N-FINDR {np.round(fe_new, 1)}, baseline {np.round(fe_base, 1)}")
        for label, folder in [("N-FINDR", new), ("baseline", base)]:
            rec = recovery(folder, name)
            if rec:
                print(f"  K=6 repeat alpha r per true mode (sorted), {label}: " + " | ".join(str(r) for r in rec))


if __name__ == "__main__":
    main()
