"""Response to reviewers: naive CV vs bi-cross-validation for the HMM on real HCP data.

Top row: held-out predictive log-likelihood log p(X_test | theta_hat) from naive 5-fold CV
(the held-out free energy, which equals it because Baum-Welch gives the exact posterior).
Bottom row: BCV held-out log-likelihood of the unseen channels, with the K selected by the
non-inferiority rule (best K by median over splits; smallest K not significantly worse,
paired one-sided t-test, p > 0.001).

Run from the osl-dynamics repo root. Only the selected K values are printed.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = Path("results_final/real")
OUT_DIR = Path("response_to_reviewers")
DATASETS = {"ICA_15": "ICA 15", "ICA_25": "ICA 25", "ICA_50": "ICA 50", "ICA_100": "ICA 100"}
SEQUENCE_LENGTH = 400  # held-out free energy is stored per sequence of 400 samples
N_FOLDS, N_SPLITS, P_THRESHOLD = 5, 100, 0.001

ncv_color, ncv_fold_color, bcv_color = "#2a78d6", "#86b6ef", "#eb6834"
ink, ink_secondary, rule = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": rule,
    "axes.labelcolor": ink_secondary,
    "xtick.color": ink_secondary,
    "ytick.color": ink_secondary,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def state_counts(dataset):
    return sorted(int(p.name.rsplit("_", 1)[1]) for p in (RESULTS_DIR / dataset).glob("hmm_state_*"))


def load_ncv(dataset):
    # Held-out log-likelihood per time point, shape (n_K, n_folds); F is stored per 400-sample sequence
    Ks = [K for K in state_counts(dataset) if K > 1]
    F = np.array([[json.load(open(RESULTS_DIR / dataset / f"hmm_state_{K}" / f"ncv_{j}" / "ncv_free_energy.json"))[0]
                   for j in range(1, N_FOLDS + 1)] for K in Ks])
    return np.array(Ks), -F / SEQUENCE_LENGTH


def load_bcv(dataset):
    # BCV held-out log-likelihood per time point, shape (n_K, n_splits)
    Ks = state_counts(dataset)
    ll = np.array([[json.load(open(RESULTS_DIR / dataset / f"hmm_state_{K}" / f"bcv_{i}" / "Y_test" / "metrics.json"))["log_likelihood"]
                    for i in range(1, N_SPLITS + 1)] for K in Ks])
    return np.array(Ks), ll


def non_inferiority_selection(ll):
    # Smallest K that is not significantly worse than the best K (by median), paired one-sided t-test
    best = int(np.argmax(np.median(ll, axis=1)))
    for i in range(len(ll)):
        if i == best or stats.ttest_rel(ll[i], ll[best], alternative="less").pvalue > P_THRESHOLD:
            return i
    return best


def mark_selected(ax, x, y, label):
    # Ring the selected point and put the label in whichever right-hand corner the curve leaves empty
    ax.scatter([x], [y], s=60, facecolor="none", edgecolor=ink, linewidth=1.2, zorder=5, clip_on=False)
    line = ax.get_lines()[-1]
    y_end = line.get_ydata()[-1]
    y_min, y_max = ax.get_ylim()
    top = (y_end - y_min) / (y_max - y_min) < 0.6
    ax.text(0.97, 0.78 if top else 0.05, label, transform=ax.transAxes, ha="right",
            va="top" if top else "bottom", fontsize=7.5, color=ink)


def style_axes(ax):
    ax.axhline(0, color=rule, lw=0.6, zorder=1)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, 26)
    ax.set_xticks([1, 5, 10, 15, 20, 25])


fig = plt.figure(figsize=(7.2, 5.0), layout="constrained")
top_fig, bottom_fig = fig.subfigures(2, 1, hspace=0.04)
top_axes = top_fig.subplots(1, len(DATASETS))
bottom_axes = bottom_fig.subplots(1, len(DATASETS))
top_fig.suptitle("a   Naive cross-validation: held-out predictive log-likelihood, log p(X_test | θ)",
                 x=0.01, ha="left", fontsize=9, color=ink)
bottom_fig.suptitle("b   Bi-cross-validation: held-out log-likelihood of the unseen channels",
                    x=0.01, ha="left", fontsize=9, color=ink)

for col, (dataset, title) in enumerate(DATASETS.items()):
    # Top row: naive CV, relative to K = 2 in each fold
    ax = top_axes[col]
    style_axes(ax)
    Ks, ncv = load_ncv(dataset)
    rel = ncv - ncv[Ks == 2]
    for j in range(rel.shape[1]):
        ax.plot(Ks, rel[:, j], color=ncv_fold_color, lw=0.8, zorder=2,
                label="Single fold" if j == 0 else None)
    mean = rel.mean(axis=1)
    ax.plot(Ks, mean, color=ncv_color, lw=1.8, marker="o", ms=2.5, zorder=3, label="Mean of 5 folds")
    i_best = int(np.argmax(mean))
    mark_selected(ax, Ks[i_best], mean[i_best], f"best: K = {Ks[i_best]}\n(largest tested)")
    ax.set_title(title, loc="left", color=ink)
    print(f"{dataset}: naive CV best K per fold {Ks[np.argmax(ncv, axis=0)].tolist()}")

    # Bottom row: BCV, mean paired difference from K = 2 with 95% CI over splits
    ax = bottom_axes[col]
    style_axes(ax)
    Ks_b, ll = load_bcv(dataset)
    diff = ll - ll[Ks_b == 2]
    mean = diff.mean(axis=1)
    ci = 1.96 * diff.std(axis=1, ddof=1) / np.sqrt(diff.shape[1])
    ax.fill_between(Ks_b, mean - ci, mean + ci, color=bcv_color, alpha=0.25, lw=0, zorder=2,
                    label="95% CI")
    ax.plot(Ks_b, mean, color=bcv_color, lw=1.8, marker="o", ms=2.5, zorder=3, label="Mean")
    i_sel = non_inferiority_selection(ll)
    mark_selected(ax, Ks_b[i_sel], mean[i_sel], f"selected: K = {Ks_b[i_sel]}")
    ax.set_title(title, loc="left", color=ink)
    ax.set_xlabel("Number of states K")
    print(f"{dataset}: BCV selected K = {Ks_b[i_sel]} (raw argmax of mean K = {Ks_b[np.argmax(ll.mean(axis=1))]})")

top_axes[0].set_ylabel("Relative to K = 2\n(nats per time point)")
bottom_axes[0].set_ylabel("Relative to K = 2\n(nats per time point)")
top_axes[0].legend(frameon=False, fontsize=7, loc="upper left")
bottom_axes[0].legend(frameon=False, fontsize=7, loc="lower left")

OUT_DIR.mkdir(exist_ok=True)
for ext in ["pdf", "png"]:
    fig.savefig(OUT_DIR / f"fig_real_ncv_vs_bcv.{ext}", dpi=300)
print(f"Saved {OUT_DIR}/fig_real_ncv_vs_bcv.pdf and .png")
