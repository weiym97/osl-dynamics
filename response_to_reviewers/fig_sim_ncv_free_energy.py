"""Response to reviewers: naive cross-validated free energy of the HMM on simulations with K = 6.

fig_sim_ncv_free_energy:
    hmm_iid_real: 6-state HMM with Gaussian observations (the fitted model is correctly specified).
    hmm_hrf: the same kind of 6-state process convolved with a haemodynamic response function.
fig_sim_ncv_free_energy_repeats:
    hmm_iid_real, hmm_iid_real_2, hmm_iid_real_3: the same simulation generated from three
    real-data HMM fits (results_final/real/ICA_50/hmm_state_6/repeat_1, 2, 3).

The held-out free energy equals -log p(X_test | theta_hat), because Baum-Welch gives the exact
posterior. Top row: relative to K = 2 within each fold. Bottom row: K >= 6 relative to K = 6
within each fold, so that differences after K = 6 are visible.

Run from the osl-dynamics repo root.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results_final/simulation")
OUT_DIR = Path("response_to_reviewers")
BLUE, ORANGE, AQUA = ("#2a78d6", "#cde2fb"), ("#eb6834", "#fbd9cb"), ("#1baf7a", "#c8eedf")
COVARIANCE_SIMULATIONS = {
    "hmm_iid_real": ("Real-data covariances", *BLUE),
    "hmm_iid": ("Random covariances", *AQUA),
    "hmm_hrf": ("Random covariances + HRF", *ORANGE),
}
# Output name -> (metric, simulations)
FIGURES = {
    "fig_sim_ncv_free_energy": ("free_energy", {
        "hmm_iid_real": ("HMM simulation", *BLUE),
        "hmm_hrf": ("HMM simulation convolved with an HRF", *ORANGE),
    }),
    "fig_sim_ncv_free_energy_covariances": ("free_energy", COVARIANCE_SIMULATIONS),
    "fig_sim_ncv_evidence_covariances": ("evidence", COVARIANCE_SIMULATIONS),
    "fig_sim_ncv_free_energy_repeats": ("free_energy", {
        "hmm_iid_real": ("HMM simulation, repeat 1", *BLUE),
        "hmm_iid_real_2": ("HMM simulation, repeat 2", *BLUE),
        "hmm_iid_real_3": ("HMM simulation, repeat 3", *BLUE),
    }),
}
K_RANGE = np.arange(2, 17)
TRUE_K, N_FOLDS = 6, 5

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


def load_free_energy(sim):
    # Held-out free energy per 400-sample sequence, shape (n_K, n_folds); NaN where a run is missing
    F = np.full((len(K_RANGE), N_FOLDS), np.nan)
    for i, K in enumerate(K_RANGE):
        for j in range(N_FOLDS):
            f = RESULTS_DIR / sim / f"hmm_state_{K}" / f"ncv_{j + 1}" / "ncv_free_energy.json"
            if f.exists():
                F[i, j] = json.load(open(f))[0]
    return F


def load_evidence(sim):
    # Held-out log p(X_test | theta_hat) per 400 samples from compute_ncv_evidence.py, shape (n_K, n_folds)
    with open(RESULTS_DIR / sim / "analysis" / "hmm_naive_evidence.json") as f:
        saved = json.load(f)
    assert saved["n_states"] == K_RANGE.tolist()
    return np.array(saved["log_evidence"], dtype=float)


METRICS = {
    "free_energy": dict(load=load_free_energy, select=np.nanargmin, better="lower",
                        title="Held-out free energy", ylabel="Free energy"),
    "evidence": dict(load=load_evidence, select=np.nanargmax, better="higher",
                     title="Held-out log-likelihood log p(X_test | θ)", ylabel="Log-likelihood"),
}


def box_with_points(ax, Ks, values, color, fill):
    data = [v[~np.isnan(v)] for v in values]
    ax.boxplot(data, positions=Ks, widths=0.6, whis=(0, 100), showfliers=False, patch_artist=True, manage_ticks=False,
               boxprops=dict(facecolor=fill, edgecolor=color, lw=0.9),
               medianprops=dict(color=color, lw=1.6),
               whiskerprops=dict(color=color, lw=0.9), capprops=dict(color=color, lw=0.9))
    rng = np.random.default_rng(0)
    for K, v in zip(Ks, data):
        ax.scatter(K + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=color,
                   edgecolor="#fcfcfb", linewidth=0.5, zorder=4)


def style_axes(ax, Ks):
    ax.axvline(TRUE_K, color=rule, ls="--", lw=0.9, zorder=1)
    ax.axhline(0, color=rule, lw=0.6, zorder=1)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(Ks[0] - 0.7, Ks[-1] + 0.7)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(K) if K % 2 == 0 else "" for K in Ks])


def make_figure(metric, simulations, out_name):
    m = METRICS[metric]
    n_cols = len(simulations)
    fig = plt.figure(figsize=(7.2, 5.4), layout="constrained")
    top_fig, bottom_fig = fig.subfigures(2, 1, hspace=0.04)
    top_axes = top_fig.subplots(1, n_cols)
    bottom_axes = bottom_fig.subplots(1, n_cols)
    top_fig.suptitle(f"a   {m['title']} relative to K = 2 ({m['better']} is better); dashed line: true K = 6",
                     x=0.01, ha="left", fontsize=9, color=ink)
    bottom_fig.suptitle("b   Zoom on K ≥ 6, relative to K = 6 within each fold; dashed line: true K = 6",
                        x=0.01, ha="left", fontsize=9, color=ink)

    for col, (sim, (title, color, fill)) in enumerate(simulations.items()):
        F = m["load"](sim)
        best = K_RANGE[m["select"](F, axis=0)]
        n_missing = int(np.isnan(F).sum())

        ax = top_axes[col]
        style_axes(ax, K_RANGE)
        box_with_points(ax, K_RANGE, F - F[0], color, fill)
        ax.set_title(title, loc="left", color=ink)
        if np.all(best == best[0]):
            label = f"selected: K = {best[0]}\n({N_FOLDS}/{N_FOLDS} folds)"
            if best[0] == K_RANGE[-1]:
                label += "\nlargest K tested"
        else:
            label = "selected K per fold:\n" + ", ".join(str(K) for K in sorted(best))
        ax.text(0.97, 0.55, label, transform=ax.transAxes, ha="right", va="center", fontsize=7.5, color=ink)

        ax = bottom_axes[col]
        zoom = K_RANGE >= TRUE_K
        style_axes(ax, K_RANGE[zoom])
        box_with_points(ax, K_RANGE[zoom], (F - F[K_RANGE == TRUE_K])[zoom], color, fill)
        ax.set_title(title, loc="left", color=ink)
        ax.set_xlabel("Number of states K")
        print(f"{sim}: selected K per fold {best.tolist()}, missing runs {n_missing}")

    top_axes[0].set_ylabel(f"{m['ylabel']}, relative to K = 2\n(nats per 400 samples)")
    bottom_axes[0].set_ylabel(f"{m['ylabel']}, relative to K = 6\n(nats per 400 samples)")

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{out_name}.{ext}", dpi=300)
    print(f"Saved {OUT_DIR}/{out_name}.pdf and .png")


for out_name, (metric, simulations) in FIGURES.items():
    make_figure(metric, simulations, out_name)
