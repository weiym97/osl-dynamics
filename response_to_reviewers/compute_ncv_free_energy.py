"""Held-out free energy for the HMM naive cross-validation (NCV) runs, with the fixed free_energy().

The 2026-10-31 fix averages over sequences rather than batches (the smaller last batch was over-weighted).
For each batch, NCV fold and number of states K, this loads the trained model, selects the held-out sessions
exactly as the pipeline's NCV step does (data.set_keep(row_test)), and overwrites
    <batch>/hmm_state_K/ncv_j/ncv_free_energy.json   ([free energy], nats per sequence)
Checks, per run:
    - the old batch weighting, recomputed batch by batch, reproduces the value in
      <batch>/analysis/ncv_free_energy_batch_weighted.json (right model and held-out sessions)
    - free_energy() equals the batch free energies weighted by their number of sequences

Run from the osl-dynamics repo root, with batch paths relative to results_final/:
    python response_to_reviewers/compute_ncv_free_energy.py simulation/hmm_iid_real
    python response_to_reviewers/compute_ncv_free_energy.py --no-values real/ICA_50
With --no-values (for real data) only the selected K per fold and pass/fail checks are printed.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
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
    with open(batch_dir / "analysis" / "ncv_free_energy_batch_weighted.json") as f:
        stored = np.array(json.load(f)["free_energy"], dtype=float)

    free_energy = np.full((len(Ks), N_FOLDS), np.nan)
    batch_weighted = np.full((len(Ks), N_FOLDS), np.nan)
    sequence_weighted = np.full((len(Ks), N_FOLDS), np.nan)
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
                    # Free energy of each batch on its own, and its number of sequences
                    fe_batches, n_batches = [], []
                    for x in model.make_dataset(data, concatenate=True):
                        fe_batches.append(float(model.free_energy(tf.data.Dataset.from_tensors(x))))
                        n_batches.append(int(x["data"].shape[0]))
                    batch_weighted[i, j - 1] = np.mean(fe_batches)
                    sequence_weighted[i, j - 1] = np.average(fe_batches, weights=n_batches)
                    free_energy[i, j - 1] = float(model.free_energy(data))
                    with open(run_dir / "ncv_free_energy.json", "w") as f:
                        json.dump([free_energy[i, j - 1]], f)
        data.delete_dir()

    old_diff = np.nanmax(np.abs(batch_weighted - stored))
    new_diff = np.nanmax(np.abs(free_energy - sequence_weighted))
    evidence = np.array([[json.load(open(batch_dir / f"hmm_state_{K}" / f"ncv_{j}" / "ncv_evidence.json"))[0]
                          for j in range(1, N_FOLDS + 1)] for K in Ks])
    print(f"== {batch}  (missing models {int(np.isnan(free_energy).sum())}, missing stored free energy {int(np.isnan(stored).sum())})")
    print(f"   K selected by held-out free energy (min): {Ks[np.nanargmin(free_energy, axis=0)].tolist()}")
    print(f"   K selected by held-out evidence (max):    {Ks[np.nanargmax(evidence, axis=0)].tolist()}")
    print(f"   old batch weighting reproduces stored ncv_free_energy.json (within 0.01): {bool(old_diff < 0.01)}")
    print(f"   free_energy() equals the sequence-weighted batch average (within 0.001): {bool(new_diff < 0.001)}")
    if not no_values:
        print(f"   max |old batch weighting - stored|: {old_diff:.4f};  max |free_energy() - sequence-weighted|: {new_diff:.5f}")
        d = evidence + free_energy
        print(f"   evidence - (-free energy): range [{np.nanmin(d):+.3f}, {np.nanmax(d):+.3f}] nats per sequence")
