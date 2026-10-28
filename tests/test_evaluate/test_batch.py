"""Running model selection from a config."""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

import reference_wip as ref
from osl_dynamics.evaluate import batch


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    covs = [np.eye(4), 0.5 * np.eye(4) + 0.5, np.diag([3.0, 1.0, 0.5, 2.0])]
    (tmp_path / "data").mkdir()
    for i in range(6):
        blocks = [
            rng.multivariate_normal(np.zeros(4), covs[k], 100) for k in [0, 1, 2] * 3
        ]
        np.save(tmp_path / "data" / f"sub-{i}.npy", np.concatenate(blocks))
    return str(tmp_path / "data")


def swc_config(data_dir, save_dir, modes=None):
    return {
        "data": data_dir,
        "save_dir": save_dir,
        "model": {"swc": {"window_length": 50, "window_offset": 25, "random_state": 0}},
        "n_states": [1, 3],
        "modes": modes
        or {
            "bcv": {
                "split_row": {"method_kwargs": {"n_splits": 2}},
                "split_column": {"method_kwargs": {"n_splits": 2}},
                "n_temp_save": 1,
            },
            "ncv": {"split_row": {"method_kwargs": {"n_splits": 2}}},
            "repeat": {"n_realizations": 1},
            "split": {"split_row": {"method_kwargs": {"n_splits": 1}}},
        },
    }


def test_normalize_minimal_config():
    config = batch.normalize_config(
        {"data": "npy_dir", "save_dir": "results/", "model": "hmm", "n_states": 4}
    )
    assert config["save_dir"] == "results"
    assert config["load_data"] == {"inputs": "npy_dir", "prepare": {"standardize": {}}}
    assert config["models"] == {
        "hmm": {
            "init_kwargs": None,
            "fit_kwargs": None,
            "spatial_method": None,
            "config_kwargs": {},
        }
    }
    assert config["n_states"] == [4]
    assert config["modes"] == {"bcv": batch.DEFAULT_MODE_SETTINGS["bcv"]}


def test_normalize_old_config():
    """The root config of the research code's IndexParser."""
    config = batch.normalize_config(
        {
            "load_data": {"inputs": "npy_dir", "prepare": {"standardize": {}}},
            "save_dir": "results",
            "model": {
                "hmm": {
                    "config_kwargs": {"n_channels": 50, "learn_means": False},
                    "init_kwargs": {"n_init": 10},
                    "infer_spatial": "sample",
                },
                "dynemo": {"n_channels": 50, "sequence_length": 100},
            },
            "n_states": [1, 2, 3],
            "mode": {
                "bcv": {
                    "split_row": {
                        "n_samples": 1003,
                        "method": "ShuffleSplit",
                        "method_kwargs": {"n_splits": 100, "train_size": 0.8},
                    },
                    "strategy": "pairing",
                    "n_temp_save": 3,
                },
                "repeat": {"n_realizations": 3},
            },
        }
    )
    assert config["models"]["hmm"]["spatial_method"] == "sample"
    assert config["models"]["hmm"]["init_kwargs"] == {"n_init": 10}
    assert config["models"]["dynemo"]["config_kwargs"] == {
        "n_channels": 50,
        "sequence_length": 100,
    }
    bcv = config["modes"]["bcv"]
    assert bcv["split_row"]["n_samples"] == 1003
    assert bcv["split_row"]["method_kwargs"] == {"n_splits": 100, "train_size": 0.8}
    assert bcv["split_column"] == batch.DEFAULT_MODE_SETTINGS["bcv"]["split_column"]
    assert bcv["n_temp_save"] == 3
    assert list(config["modes"]) == ["bcv", "repeat"]

    with pytest.raises(ValueError, match="cv_variant"):
        batch.normalize_config(
            {
                "data": "npy_dir",
                "save_dir": "results",
                "model": "hmm",
                "n_states": 2,
                "modes": {"bcv": {"cv_variant": 2}},
            }
        )


def test_job_order_matches_reference():
    config = batch.normalize_config(
        {
            "data": "npy_dir",
            "save_dir": "results",
            "model": ["hmm", "dynemo"],
            "n_states": [1, 2],
            "modes": {
                "bcv": {
                    "split_row": {"method_kwargs": {"n_splits": 3}},
                    "split_column": {"method_kwargs": {"n_splits": 3}},
                },
                "ncv": {"split_row": {"method_kwargs": {"n_splits": 2}}},
                "repeat": {"n_realizations": 2},
                "split": {"split_row": {"method_kwargs": {"n_splits": 2}}},
            },
        }
    )
    mode_index = [f"bcv_{i}" for i in [1, 2, 3]]
    mode_index += [f"ncv_{i}" for i in [1, 2]]
    mode_index += [f"repeat_{i}" for i in [1, 2]]
    mode_index += [f"split_{i}" for i in [1, 2]]
    expected = ref.job_order(["hmm", "dynemo"], [1, 2], mode_index)
    jobs = [(model, n, f"{mode}_{i}") for model, n, mode, i in batch.get_jobs(config)]
    assert jobs == expected


def test_run_sequential_and_resume(data_dir, tmp_path, monkeypatch):
    save_dir = str(tmp_path / "results")
    summary = batch.run(swc_config(data_dir, save_dir))

    assert len(summary["bcv"]) == 4
    assert len(summary["ncv"]) == 2  # skipped for one state
    assert len(summary["repeat"]) == 2
    assert len(summary["split"]) == 2
    assert summary["bcv"]["log_likelihood"].notna().all()
    for mode in ["bcv", "ncv", "repeat", "split"]:
        assert os.path.exists(f"{save_dir}/summary/{mode}.csv")
    assert os.path.exists(f"{save_dir}/summary/bcv_swc_log_likelihood.png")

    # Time courses are only saved for the first n_temp_save folds
    assert os.path.exists(f"{save_dir}/swc_state_3/bcv_1/Y_test/inf_params/alp.pkl")
    assert not os.path.exists(f"{save_dir}/swc_state_3/bcv_2/Y_test/inf_params")

    # Running again does not load the data or re-run any job
    def fail(*args, **kwargs):
        raise AssertionError("The data should not be loaded")

    monkeypatch.setattr(batch, "_load_data", fail)
    summary_2 = batch.run(swc_config(data_dir, save_dir))
    pd.testing.assert_frame_equal(summary["bcv"], summary_2["bcv"])


def test_init_index_and_summarise(data_dir, tmp_path):
    save_dir = str(tmp_path / "results")
    config_file = str(tmp_path / "config.yml")
    with open(config_file, "w") as file:
        yaml.safe_dump(swc_config(data_dir, save_dir), file)

    n_jobs = batch.init(config_file)
    assert n_jobs == 2 * (2 + 2 + 1 + 1)
    assert os.path.exists(f"{save_dir}/config_root.yaml")
    job_list = pd.read_csv(f"{save_dir}/config_list.csv", index_col=0)
    assert list(job_list.columns) == ["model_index", "n_states", "mode_index"]
    assert len(os.listdir(f"{save_dir}/bcv_partition")) == 2

    for index in range(1, n_jobs + 1):
        batch.main([config_file, "--index", str(index)])
    batch.main([config_file, "--summarise"])

    bcv = pd.read_csv(f"{save_dir}/summary/bcv.csv")
    assert len(bcv) == 4
    sequential = batch.run(swc_config(data_dir, str(tmp_path / "sequential")))
    pd.testing.assert_frame_equal(bcv, sequential["bcv"])


def test_index_requires_init_for_random_folds(data_dir, tmp_path):
    config = swc_config(
        data_dir,
        str(tmp_path / "results"),
        modes={"bcv": {"random_state": None}},
    )
    with pytest.raises(ValueError, match="--init"):
        batch.run(config, index=1)
    assert batch.init(config) == 2 * 10
    batch.run(config, index=1)


def test_data_object_and_numpy_values(data_dir, tmp_path):
    from osl_dynamics.data import Data

    data = Data(data_dir)
    config = {
        "data": data,
        "save_dir": str(tmp_path / "results"),
        "model": {
            "swc": {
                "window_length": np.int64(50),
                "window_offset": 25,
                "random_state": 0,
            }
        },
        "n_states": np.array([1, 3]),
        "modes": {"bcv": {"split_row": {"method_kwargs": {"n_splits": 2}}}},
    }
    assert batch.init(config) == 2 * 2
    batch.run(config, index=4)
    with open(
        tmp_path / "results" / "swc_state_3" / "bcv_2" / "batch_config.yaml"
    ) as file:
        assert yaml.safe_load(file)["config_kwargs"]["window_length"] == 50
    with open(tmp_path / "results" / "config_root.yaml") as file:
        assert yaml.safe_load(file)["load_data"] is None
    data.delete_dir()


def test_failed_jobs_are_recorded(data_dir, tmp_path, monkeypatch):
    save_dir = str(tmp_path / "results")
    config = swc_config(
        data_dir,
        save_dir,
        modes={
            "bcv": {
                "split_row": {"method_kwargs": {"n_splits": 2}},
                "split_column": {"method_kwargs": {"n_splits": 2}},
            }
        },
    )

    # Make every bi-cross-validation of 3 states fail
    original_validate = batch.BCV.validate

    def failing_validate(self, data, fold, output_dir=None, save_temporal=None):
        if self.ops.n_states == 3:
            raise RuntimeError("Simulated failure")
        return original_validate(self, data, fold, output_dir, save_temporal)

    monkeypatch.setattr(batch.BCV, "validate", failing_validate)
    summary = batch.run(config)

    # The other jobs ran and the failures were recorded
    assert len(summary["bcv"]) == 2
    with open(f"{save_dir}/summary/failed_jobs.txt") as file:
        failed = file.read().split()
    assert failed == [f"{save_dir}/swc_state_3/bcv_{i}" for i in [1, 2]]
    with open(f"{save_dir}/swc_state_3/bcv_1/error.txt") as file:
        assert "Simulated failure" in file.read()

    # A single job raises the error
    with pytest.raises(RuntimeError, match="Simulated failure"):
        batch.run(config, index=3)

    # Failed jobs are retried on the next run
    monkeypatch.setattr(batch.BCV, "validate", original_validate)
    summary = batch.run(config)
    assert len(summary["bcv"]) == 4
    assert not os.path.exists(f"{save_dir}/summary/failed_jobs.txt")
    assert not os.path.exists(f"{save_dir}/swc_state_3/bcv_1/error.txt")
