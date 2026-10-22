"""Collect metrics.json files from a runs directory into one table.

Usage: python summarize.py <runs_dir>
"""

import glob
import json
import sys

import numpy as np

COLUMNS = [
    ("lam_min", "lam_min", "{:.3f}"),
    ("n<0.5", "n_lam_below_0.5", "{}"),
    ("KL_truth", "kl_truth_mean", "{:.2f}"),
    ("R2 mean~private", "r2_model_mean_on_private_noise", "{:.2f}"),
    ("R2 mean~true", "r2_model_mean_on_true_mean", "{:.2f}"),
    ("alpha corr", "alpha_corr_matched", "{:.2f}"),
    ("mean max a", "alpha_mean_max", "{:.2f}"),
    ("psi/true", None, "{}"),
    ("min", "train_minutes", "{:.0f}"),
]


def main():
    rows = []
    for f in sorted(glob.glob(f"{sys.argv[1]}/*/metrics.json")):
        r = json.load(open(f))
        cells = [r["sim"], r["variant"], str(r["seed"])]
        for _, key, fmt in COLUMNS:
            if key is None:
                cells.append(f"{r['private_var_mean']:.2f}/{r['private_var_true_mean']:.2f}"
                             if "private_var_mean" in r else "-")
            elif key == "alpha_corr_matched":
                cells.append(fmt.format(np.mean(r[key])))
            else:
                cells.append(fmt.format(r[key]))
        rows.append(cells)
    header = ["sim", "variant", "seed"] + [c[0] for c in COLUMNS]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)] if rows else []
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


if __name__ == "__main__":
    main()
