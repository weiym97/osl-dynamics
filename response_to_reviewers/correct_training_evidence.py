"""Correct the training-set `evidence` stored in metrics/metrics.json (HMM runs) after the evidence() fix (a7768e5f).

Before the fix, HMM.evidence() used the stationary state distribution at t = 1 as if it were a
log-probability. For every HMM run under a batch, this script:
  1. rebuilds the run's training data from its prepared_config.yaml (keep_list, selected channels),
     starting from the batch's prepared data (config_root.yaml load_data);
  2. recomputes evidence() exactly as osl-dynamics does (each sequence of `sequence_length` samples
     is a separate chain, stationary initial distribution, mean over sequences within a batch of
     `batch_size`, then mean over batches), once with the old bug and once fixed;
  3. only if the with-bug value reproduces the stored `evidence` (within TOLERANCE), writes the fixed
     value to `evidence` and keeps the old one as `evidence_buggy`. Runs that do not reproduce are left
     untouched and reported. Already-corrected files are re-checked against `evidence_buggy`.

Run from the osl-dynamics repo root, with the batch path relative to results_final/ and optionally a
list of K values (for cluster array jobs):
    python response_to_reviewers/correct_training_evidence.py simulation/hmm_iid_real [--K 6] [--no-values] [--dry-run]
"""
import argparse
import glob
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import solve_triangular

from osl_dynamics.config_api.wrappers import load_data

RESULTS_DIR = Path("results_final")
TOLERANCE = 0.05  # nats per sequence, for reproducing the stored (float32) value


def log_likelihoods(x, means, covs):
    # log N(x_t; mu_k, Sigma_k) for every sample and state, shape (n_samples, K)
    d = x.shape[1]
    out = np.empty((x.shape[0], len(covs)))
    for k, (mu, cov) in enumerate(zip(means, covs)):
        L = np.linalg.cholesky(cov)
        y = solve_triangular(L, (x - mu).T, lower=True)
        out[:, k] = -0.5 * (d * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + np.sum(y**2, axis=0))
    return out


def stationary_distribution(P):
    # As HMM.get_stationary_distribution
    eigval, eigvec = np.linalg.eig(P.T)
    pi = np.squeeze(eigvec[:, np.isclose(eigval, 1)]).real
    return pi / pi.sum()


def log_evidence(log_b, P, pi0):
    # Scaled forward algorithm for chains of equal length; pi0 enters as given
    shift = log_b.max(axis=2, keepdims=True)
    b = np.exp(log_b - shift)
    a = pi0 * b[:, 0]
    total = np.log(a.sum(axis=1))
    alpha = a / a.sum(axis=1, keepdims=True)
    for t in range(1, log_b.shape[1]):
        a = (alpha @ P) * b[:, t]
        s = a.sum(axis=1)
        total += np.log(s)
        alpha = a / s[:, None]
    return total + shift[..., 0].sum(axis=1)


def batch_mean(per_sequence, batch_size):
    return np.mean([per_sequence[i:i + batch_size].mean() for i in range(0, len(per_sequence), batch_size)])


def osl_evidence(x_sessions, means, covs, P, sequence_length, batch_size):
    # Returns (fixed, buggy) evidence as computed by HMM.evidence() after / before the fix
    n_seq = x_sessions.shape[1] // sequence_length
    seqs = x_sessions[:, :n_seq * sequence_length].reshape(-1, sequence_length, x_sessions.shape[2])
    log_b = log_likelihoods(seqs.reshape(-1, seqs.shape[2]), means, covs).reshape(len(seqs), sequence_length, -1)
    pi = stationary_distribution(P)
    fixed = batch_mean(log_evidence(log_b, P, pi), batch_size)
    buggy = batch_mean(log_evidence(log_b, P, np.exp(pi)), batch_size)
    return fixed, buggy


parser = argparse.ArgumentParser()
parser.add_argument("batch")
parser.add_argument("--K", type=int, nargs="*", default=None)
parser.add_argument("--no-values", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

batch_dir = RESULTS_DIR / args.batch
with open(batch_dir / "config_root.yaml") as f:
    root_load = yaml.safe_load(f)["load_data"]

# Prepared data of the whole batch (all sessions and channels), as in training
with tempfile.TemporaryDirectory() as store_dir:
    data = load_data(inputs=root_load["inputs"], kwargs={**root_load.get("kwargs", {}), "store_dir": store_dir},
                     prepare=root_load.get("prepare", {}))
    x_all = np.stack([np.array(ts, dtype=np.float64) for ts in data.time_series(prepared=True, concatenate=False)])
    data.delete_dir()

K_dirs = sorted(glob.glob(str(batch_dir / "hmm_state_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
if args.K:
    K_dirs = [p for p in K_dirs if int(p.rsplit("_", 1)[1]) in args.K]

summary = {}
for K_dir in K_dirs:
    for metrics_file in sorted(glob.glob(f"{K_dir}/*/**/metrics/metrics.json", recursive=True)):
        metrics = json.load(open(metrics_file))
        if "evidence" not in metrics:
            continue
        run_dir = Path(metrics_file).parent.parent
        kind = os.path.relpath(run_dir, K_dir).split(os.sep)
        kind = kind[0].split("_")[0] + ("/" + "/".join(kind[1:]) if len(kind) > 1 else "")
        prepared = yaml.safe_load(open(run_dir / "prepared_config.yaml"))
        select = (prepared["load_data"].get("prepare") or {}).get("select") or {}
        root_select = (root_load.get("prepare") or {}).get("select") or {}
        assert select.get("timepoints") == root_select.get("timepoints"), run_dir

        keep = prepared.get("keep_list")
        if isinstance(keep, str):
            keep = json.load(open(keep))
        sessions = sorted(keep) if keep is not None else list(range(len(x_all)))
        x = x_all[sessions]
        if select.get("channels") is not None:
            x = x[:, :, select["channels"]]

        model_config = prepared["train_hmm"]["config_kwargs"]
        fixed, buggy = osl_evidence(
            x,
            np.load(run_dir / "inf_params" / "means.npy").astype(np.float64),
            np.load(run_dir / "inf_params" / "covs.npy").astype(np.float64),
            np.load(run_dir / "model" / "trans_prob.npy").astype(np.float64),
            model_config["sequence_length"], model_config["batch_size"],
        )
        stored_buggy = metrics.get("evidence_buggy", metrics["evidence"])
        ok = abs(buggy - stored_buggy) < TOLERANCE
        s = summary.setdefault(kind, {"n": 0, "reproduced": 0, "max_diff": 0.0, "failed": []})
        s["n"] += 1
        s["reproduced"] += ok
        s["max_diff"] = max(s["max_diff"], abs(buggy - stored_buggy))
        if not ok:
            s["failed"].append(str(run_dir))
        elif not args.dry_run:
            metrics["evidence_buggy"] = stored_buggy
            metrics["evidence"] = float(fixed)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f)

label = f"{args.batch} K={args.K if args.K else 'all'}" + (" (dry run)" if args.dry_run else "")
for kind, s in sorted(summary.items()):
    line = f"{label} {kind}: {s['reproduced']}/{s['n']} reproduced stored evidence"
    if not args.no_values:
        line += f", max |with-bug - stored| {s['max_diff']:.4f}"
    print(line + ("" if args.dry_run else f"; corrected {s['reproduced']}"))
    for run in s["failed"][:5]:
        print(f"   NOT reproduced, left untouched: {run}")
