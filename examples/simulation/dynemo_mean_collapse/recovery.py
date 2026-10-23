"""Post-hoc recovery of mode means and FC for fitted runs (K = true number of modes only).

Modes are matched to the truth by mode time-course correlation. For each matched pair we
report corr(mu_model, mu_true) and the correlation between the off-diagonal elements of the
model and true mode correlation matrices (FC). True covariances include the private noise.

Usage: python recovery.py <runs_dir> [<runs_dir> ...]
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))


def fc(C):
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    return R[np.triu_indices(len(R), 1)]


def main():
    print(f"{'run':<55} {'mean corr':>9} {'FC corr':>8} {'FC contrast corr':>16}")
    for runs_dir in sys.argv[1:]:
        for d in sorted(glob.glob(f"{runs_dir}/*/")):
            r = json.load(open(f"{d}/metrics.json"))
            truth = np.load(f"{HERE}/output/{r['sim']}/truth.npz")
            if r["n_modes"] != truth["alpha"].shape[-1] or not os.path.exists(f"{d}/covs.npy"):
                continue
            K = r["n_modes"]
            a = np.load(f"{d}/alpha.npy").reshape(-1, K)[::20]
            a_true = truth["alpha"].reshape(-1, K)[::20]
            corr = np.nan_to_num(np.corrcoef(a.T, a_true.T)[:K, K:])
            rows, cols = linear_sum_assignment(-corr)
            means, covs = np.load(f"{d}/means.npy"), np.load(f"{d}/covs.npy")
            if os.path.exists(f"{d}/private_var.npy"):
                covs = covs + np.diag(np.load(f"{d}/private_var.npy"))
            covs_true = truth["covs"] + np.diag(truth["private_var"])
            mean_corr = [np.corrcoef(means[i], truth["means"][j])[0, 1] for i, j in zip(rows, cols)]
            fc_corr = [np.corrcoef(fc(covs[i]), fc(covs_true[j]))[0, 1] for i, j in zip(rows, cols)]
            # FC differences between modes (the dynamic part): each mode minus the mode average
            fc_m = np.array([fc(covs[i]) for i in rows])
            fc_t = np.array([fc(covs_true[j]) for j in cols])
            contrast = np.corrcoef((fc_m - fc_m.mean(0)).ravel(), (fc_t - fc_t.mean(0)).ravel())[0, 1]
            name = f"{os.path.basename(os.path.normpath(runs_dir))}/{os.path.basename(os.path.normpath(d))}"
            print(f"{name:<55} {np.mean(mean_corr):>9.2f} {np.mean(fc_corr):>8.2f} {contrast:>16.2f}")


if __name__ == "__main__":
    main()
