"""Rewrite only the BCV summary tables (<model>_Y_test_log_likelihood[_demean].txt) of a batch, with the p column.

Collects the BCV Y_test log-likelihoods exactly as BatchAnalysis.compare() does and runs the same plot_box
selection (p_value=0.001, demean_index=0, as for the paper figures), but plot_box draws into a temporary
folder that is deleted afterwards: only the .txt files are copied into <batch>/analysis/. Figures and .json
files in <batch>/analysis/ are not touched.

Usage (from the repository root):
    python response_to_reviewers/write_bcv_summaries.py ./results_final/real/ICA_15 ./results_final/real/ICA_25
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from osl_dynamics.config_api.batch import BatchAnalysis
from osl_dynamics.utils.plotting import plot_box

P_VALUE = 0.001
DEMEAN_INDEX = 0

for batch_dir in sys.argv[1:]:
    batch_analysis = BatchAnalysis(batch_dir)
    models = batch_analysis.config_root["model"].keys()
    n_states_list = batch_analysis.config_root["n_states"]

    # Same collection as BatchAnalysis.compare()
    metrics = {model: {str(int(num)): [] for num in n_states_list} for model in models}
    for i in range(len(batch_analysis.config_list)):
        config = batch_analysis.indexparser.parse(i)
        model = next(iter(config["model"]))
        n_states = config["n_states"] if model != "dynemo" else config["n_modes"]
        if "bcv" in config["mode"]:
            try:
                with open(os.path.join(config["save_dir"], "Y_test/", "metrics.json"), "r") as file:
                    metric = json.load(file)["log_likelihood"]
            except Exception:
                print(f"save_dir {config['save_dir']} fails!")
                metric = np.nan
            metrics[model][str(int(n_states))].append(metric)

    with tempfile.TemporaryDirectory() as tmp:
        for model in models:
            values = [metrics[model][str(key)] for key in n_states_list]
            for demean, suffix in [(True, "_demean"), (False, "")]:
                name = f"{model}_Y_test_log_likelihood{suffix}"
                # plot_box (and demean_list) print debugging output; keep the console clean
                with contextlib.redirect_stdout(io.StringIO()):
                    plot_box(data=values, labels=n_states_list, demean=demean, demean_index=DEMEAN_INDEX,
                             mark_best=True, p_value=P_VALUE, filename=os.path.join(tmp, f"{name}.pdf"))
                plt.close("all")
                shutil.copy(os.path.join(tmp, f"{name}.txt"), os.path.join(batch_analysis.analysis_path, f"{name}.txt"))
                print(f"{batch_dir}: wrote analysis/{name}.txt")
