"""Train DyNeMo.

"""

import numpy as np

from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

plotting.set_style({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

gt_alp = np.load("data/alp.npy")[:25600]
dyn_alp = np.load("models/dynemo/alp.npy")
hmm_alp = np.load("models/hmm/alp.npy")

gt_alp, dyn_alp, hmm_alp = modes.match_modes(gt_alp, dyn_alp, hmm_alp)

plotting.plot_alpha(
    gt_alp,
    dyn_alp,
    hmm_alp,
    y_labels=["Ground Truth", "DyNeMo", "HMM"],
    n_samples=1000,
    sampling_frequency=250,
    cmap="tab10",
    filename="plots_2/alpha.png",
)
