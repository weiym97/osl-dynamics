"""Mean BCV Y_test log-likelihood per K and per-fold best K for several runs of the same folds.

Usage (from the repository root): python examples/simulation/dynemo_mode_recovery/bcv_curves.py <sim>
"""

import json
import os
import sys

import numpy as np

sim = sys.argv[1]
H = "examples/simulation/dynemo_mode_recovery/output/runs"
SOURCES = {
    "original/baseline": f"./results_final/simulation/{sim}_baseline"
    if os.path.exists(f"./results_final/simulation/{sim}_baseline") else f"./results_final/simulation/{sim}",
    "N-FINDR": f"./results_final/simulation/{sim}_nfindr",
}
# Every tagged run of this simulation made with bcv_oracle.py
for group in ["bcv_long", "oracle"]:
    for tag in sorted(os.listdir(f"{H}/{group}/{sim}")) if os.path.isdir(f"{H}/{group}/{sim}") else []:
        SOURCES[f"{tag} ({group})"] = f"{H}/{group}/{sim}/{tag}"
KS, FOLDS = list(range(4, 9)), range(1, 6)


def ll(folder, K, f):
    p = f"{folder}/dynemo_state_{K}/bcv_{f}/Y_test/metrics.json"
    return json.load(open(p))["log_likelihood"] if os.path.exists(p) else np.nan


print(f"== {sim}: mean Y_test LL per sample over folds 1-5 (* = best K)")
for name, folder in SOURCES.items():
    a = np.array([[ll(folder, K, f) for f in FOLDS] for K in KS])
    if np.all(np.isnan(a)):
        continue
    m = np.nanmean(a, 1) if not np.all(np.isnan(a), 1).any() else np.array([np.nanmean(r) if not np.all(np.isnan(r)) else np.nan for r in a])
    best = int(np.nanargmax(m))
    cells = " ".join(f"K{K}:{v:9.4f}{'*' if i == best else ' '}" if not np.isnan(v) else f"K{K}:    -     " for i, (K, v) in enumerate(zip(KS, m)))
    line = f"{name:>26}: {cells}"
    if np.isnan(a).sum() == 0:
        line += f" | per-fold best K {[KS[i] for i in np.argmax(a, 0)]}"
        i6 = KS.index(6)
        line += " | K6-K5 {:+.4f} ({}/5), K6-K7 {:+.4f} ({}/5)".format(
            np.mean(a[i6] - a[i6 - 1]), int(np.sum(a[i6] > a[i6 - 1])), np.mean(a[i6] - a[i6 + 1]), int(np.sum(a[i6] > a[i6 + 1])))
    print(line)
