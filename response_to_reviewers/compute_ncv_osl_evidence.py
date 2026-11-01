"""Held-out evidence for the HMM naive cross-validation (NCV) runs, with osl-dynamics' own (fixed) evidence().

For each batch, NCV fold and number of states K, this loads the trained model, selects the held-out
sessions exactly as the pipeline's NCV step does (data.set_keep(row_test)), and saves
    <batch>/hmm_state_K/ncv_j/ncv_evidence.json   ([evidence], nats per sequence, like ncv_free_energy.json)
The held-out free energy is recomputed as a check against the stored ncv_free_energy.json (not overwritten).

Requires the evidence() fix in commit a7768e5f (log of the initial state distribution) and the 2026-10-31 fix
that averages over sequences rather than batches (the smaller last batch was over-weighted).

Run from the osl-dynamics repo root, with batch paths relative to results_final/:
    python response_to_reviewers/compute_ncv_osl_evidence.py simulation/hmm_iid_real
    python response_to_reviewers/compute_ncv_osl_evidence.py --no-values real/ICA_50
With --no-values (for real data) only the selected K per fold and pass/fail checks are printed.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from osl_dynamics import models
from osl_dynamics.config_api.wrappers import load_data

RESULTS_DIR = Path("results_final")
N_FOLDS = 5

no_values = "--no-values" in sys.argv
for batch in [a for a in sys.argv[1:] if a != "--no-values"]:
    batch_dir = RESULTS_DIR / batch
    with open(batch_dir / "config_root.yaml") as f:
        config_root = yaml.safe_load(f)
    Ks = np.array([K for K in config_root["n_states"] if K > 1])
    load_kwargs = config_root["load_data"]

    evidence = np.full((len(Ks), N_FOLDS), np.nan)
    free_energy = np.full((len(Ks), N_FOLDS), np.nan)
    stored_free_energy = np.full((len(Ks), N_FOLDS), np.nan)
    with tempfile.TemporaryDirectory() as store_dir:
        data = load_data(
            inputs=load_kwargs["inputs"],
            kwargs={**load_kwargs.get("kwargs", {}), "store_dir": store_dir},
            prepare=load_kwargs.get("prepare", {}),
        )
        for j in range(1, N_FOLDS + 1):
            with open(batch_dir / "ncv_partition" / f"fold_indices_{j}.json") as f:
                test = json.load(f)["row_test"]
            with data.set_keep(test):
                for i, K in enumerate(Ks):
                    run_dir = batch_dir / f"hmm_state_{K}" / f"ncv_{j}"
                    if not (run_dir / "model").exists():
                        continue
                    model = models.load(str(run_dir / "model"))
                    evidence[i, j - 1] = float(model.evidence(data))
                    free_energy[i, j - 1] = float(model.free_energy(data))
                    with open(run_dir / "ncv_evidence.json", "w") as f:
                        json.dump([evidence[i, j - 1]], f)
                    if (run_dir / "ncv_free_energy.json").exists():
                        stored_free_energy[i, j - 1] = json.load(open(run_dir / "ncv_free_energy.json"))[0]
        data.delete_dir()

    fe_diff = np.nanmax(np.abs(free_energy - stored_free_energy))
    print(f"== {batch}  (missing models {int(np.isnan(evidence).sum())}, missing stored free energy {int(np.isnan(stored_free_energy).sum())})")
    print(f"   K selected by held-out evidence (max):    {Ks[np.nanargmax(evidence, axis=0)].tolist()}")
    print(f"   K selected by held-out free energy (min): {Ks[np.nanargmin(free_energy, axis=0)].tolist()}")
    print(f"   recomputed free energy matches stored ncv_free_energy.json (within 0.01): {bool(fe_diff < 0.01)}")
    if not no_values:
        rel6 = evidence - evidence[Ks == 6]
        print(f"   max |recomputed - stored free energy|: {fe_diff:.4f}")
        print("   evidence relative to K = 6, mean over folds: "
              + "  ".join(f"K{K}:{np.nanmean(r):+.1f}" for K, r in zip(Ks, rel6)))
