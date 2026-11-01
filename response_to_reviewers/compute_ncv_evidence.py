"""Held-out predictive log-likelihood (evidence) for the HMM naive cross-validation (NCV) runs.

For each batch, NCV fold and number of states K, this computes log p(X_test | theta_hat) with the
forward algorithm, using the parameters fixed after training on that fold (inf_params/covs.npy,
model/trans_prob.npy, zero means). Each held-out session is one chain starting from the stationary
distribution of trans_prob. Values are in nats per 400 samples, the same scale as ncv_free_energy.json.

Data are loaded and prepared exactly as in training, with the load_data settings in the batch's
config_root.yaml. Saves <batch>/analysis/hmm_naive_evidence.json.

Run from the osl-dynamics repo root, with batch paths relative to results_final/:
    python response_to_reviewers/compute_ncv_evidence.py simulation/hmm_iid_real simulation/hmm_hrf
    python response_to_reviewers/compute_ncv_evidence.py --no-values real/ICA_50
With --no-values (for real data) only the selected K per fold and checks are printed, never metric values.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import solve_triangular

from osl_dynamics.config_api.wrappers import load_data

RESULTS_DIR = Path("results_final")
N_FOLDS, SEQUENCE_LENGTH = 5, 400


def log_likelihoods(x, covs):
    # log N(x_t; 0, Sigma_k) for every sample and state, shape (n_samples, K)
    d = x.shape[1]
    out = np.empty((x.shape[0], len(covs)))
    for k, cov in enumerate(covs):
        L = np.linalg.cholesky(cov)
        y = solve_triangular(L, x.T, lower=True)
        out[:, k] = -0.5 * (d * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + np.sum(y**2, axis=0))
    return out


def stationary_distribution(P):
    eigval, eigvec = np.linalg.eig(P.T)
    pi = np.real(eigvec[:, np.argmin(np.abs(eigval - 1))])
    return pi / pi.sum()


def log_evidence(log_b, P, pi0):
    # Scaled forward algorithm for chains of equal length; log_b has shape (n_chains, T, K)
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


def load_sessions(config_root):
    # All sessions, prepared as in training (memmaps go to a temporary directory)
    load_kwargs = config_root["load_data"]
    with tempfile.TemporaryDirectory() as store_dir:
        data = load_data(
            inputs=load_kwargs["inputs"],
            kwargs={**load_kwargs.get("kwargs", {}), "store_dir": store_dir},
            prepare=load_kwargs.get("prepare", {}),
        )
        sessions = np.stack([np.array(ts, dtype=np.float64) for ts in data.time_series(prepared=True, concatenate=False)])
        data.delete_dir()
    return sessions


no_values = "--no-values" in sys.argv
for batch in [a for a in sys.argv[1:] if a != "--no-values"]:
    batch_dir = RESULTS_DIR / batch
    with open(batch_dir / "config_root.yaml") as f:
        config_root = yaml.safe_load(f)
    Ks = np.array([K for K in config_root["n_states"] if K > 1])
    x_all = load_sessions(config_root)
    n_time, n_channels = x_all.shape[1:]

    evidence = np.full((len(Ks), N_FOLDS), np.nan)
    free_energy = np.full((len(Ks), N_FOLDS), np.nan)
    for j in range(1, N_FOLDS + 1):
        with open(batch_dir / "ncv_partition" / f"fold_indices_{j}.json") as f:
            test = sorted(json.load(f)["row_test"])
        x = x_all[test].reshape(-1, n_channels)
        for i, K in enumerate(Ks):
            run_dir = batch_dir / f"hmm_state_{K}" / f"ncv_{j}"
            if (run_dir / "model" / "trans_prob.npy").exists():
                covs = np.load(run_dir / "inf_params" / "covs.npy").astype(np.float64)
                P = np.load(run_dir / "model" / "trans_prob.npy").astype(np.float64)
                lb = log_likelihoods(x, covs).reshape(len(test), n_time, K)
                evidence[i, j - 1] = log_evidence(lb, P, stationary_distribution(P)).sum() / (len(test) * n_time / SEQUENCE_LENGTH)
            if (run_dir / "ncv_free_energy.json").exists():
                free_energy[i, j - 1] = json.load(open(run_dir / "ncv_free_energy.json"))[0]

    out_dir = batch_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "hmm_naive_evidence.json", "w") as f:
        json.dump({
            "definition": "held-out log p(X_test | theta_hat), forward algorithm, one chain per test session, "
                          "stationary initial distribution; nats per 400 samples",
            "n_states": Ks.tolist(),
            "folds": list(range(1, N_FOLDS + 1)),
            "log_evidence": [[None if np.isnan(v) else float(v) for v in row] for row in evidence],
        }, f, indent=1)

    best_e = Ks[np.nanargmax(evidence, axis=0)]
    best_f = Ks[np.nanargmin(free_energy, axis=0)]
    worse_steps = [int(np.sum(np.diff(evidence[:, j]) < 0)) for j in range(N_FOLDS)]
    print(f"== {batch}  (sessions {x_all.shape[0]} x {n_time} x {n_channels}; "
          f"missing evidence {int(np.isnan(evidence).sum())}, missing free energy {int(np.isnan(free_energy).sum())})")
    print(f"   K selected by held-out evidence (max):    {best_e.tolist()}")
    print(f"   K selected by held-out free energy (min): {best_f.tolist()}")
    print(f"   steps where adding a state lowered the evidence, per fold: {worse_steps}")
    if not no_values:
        rel6 = evidence - evidence[Ks == 6]
        print("   evidence relative to K = 6, mean over folds: "
              + "  ".join(f"K{K}:{np.nanmean(r):+.1f}" for K, r in zip(Ks, rel6)))
    print(f"   saved {out_dir / 'hmm_naive_evidence.json'}")
