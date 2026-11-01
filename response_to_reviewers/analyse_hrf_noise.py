"""BatchAnalysis BCV and NCV plots for the hmm_hrf noise sweep (results_final/simulation/hmm_hrf_noise_<r>/).

Reads only Y_test/metrics.json (BCV) and ncv_free_energy.json (NCV); figures and the BCV summary go to
<batch>/analysis/. Settings follow hmm_iid_real_2: compare(demean_index=0, p_value=0.001, inset_start_index=6)
and plot_naive_cv demeaned at K=2 and at K=6.

Usage (from the repository root; arguments are batch names, or noise ratios for hmm_hrf_noise):
    python response_to_reviewers/analyse_hrf_noise.py 0.25 1.0 4.0
    python response_to_reviewers/analyse_hrf_noise.py hmm_ar1_0.0 hmm_ar1_0.5 hmm_hrf_ar1noise_1.0
"""

import sys

import matplotlib

matplotlib.use("Agg")

from osl_dynamics.config_api.batch import BatchAnalysis

for r in sys.argv[1:]:
    name = r if r.startswith("hmm_") else f"hmm_hrf_noise_{r}"
    batch_analysis = BatchAnalysis(f"./results_final/simulation/{name}/")
    batch_analysis.compare(demean_index=0, p_value=0.001, inset_start_index=6)
    # n_states = 1 is dropped in plot_naive_cv, so index 0 is K=2 and index 4 is K=6
    batch_analysis.plot_naive_cv(demean_index=0)
    batch_analysis.plot_naive_cv(demean_index=4)
    print(f"{name}: done")
