"""Splitting sessions and channels."""

import json
import os

import pytest

import reference_wip as ref
from osl_dynamics.evaluate.splits import CVSplit, load_fold

CONFIGS = {
    # Case 1: row ShuffleSplit, column KFold, combination
    1: {
        "split_row": {
            "n_samples": 10,
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 5, "train_size": 0.8},
        },
        "split_column": {
            "n_samples": 4,
            "method": "KFold",
            "method_kwargs": {"n_splits": 2},
        },
        "strategy": "combination",
    },
    # Case 2: row ShuffleSplit, column KFold, pairing
    2: {
        "split_row": {
            "n_samples": 5,
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 3, "train_size": 0.8},
        },
        "split_column": {
            "n_samples": 4,
            "method": "KFold",
            "method_kwargs": {"n_splits": 2},
        },
        "strategy": "pairing",
    },
    # Case 3: row KFold, column ShuffleSplit, combination
    3: {
        "split_row": {
            "n_samples": 5,
            "method": "KFold",
            "method_kwargs": {"n_splits": 5},
        },
        "split_column": {
            "n_samples": 4,
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 2, "train_size": 0.5},
        },
        "strategy": "combination",
    },
    # Case 4: row KFold, column ShuffleSplit, pairing
    4: {
        "split_row": {
            "n_samples": 5,
            "method": "KFold",
            "method_kwargs": {"n_splits": 5},
        },
        "split_column": {
            "n_samples": 4,
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 2, "train_size": 0.5},
        },
        "strategy": "pairing",
    },
    # Case 5: row ShuffleSplit, no column splitting
    5: {
        "split_row": {
            "n_samples": 5,
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 3, "train_size": 0.8},
        },
    },
    # Case 6: row KFold, no column splitting
    6: {
        "split_row": {
            "n_samples": 5,
            "method": "KFold",
            "method_kwargs": {"n_splits": 5},
        }
    },
}

N_SPLITS = {1: 10, 2: 2, 3: 10, 4: 2, 5: 3, 6: 5}


def read_folds(save_dir):
    files = sorted(
        os.listdir(save_dir), key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    folds = []
    for f in files:
        with open(f"{save_dir}/{f}") as file:
            folds.append(json.load(file))
    return folds


def test_CVSplit(tmp_path):
    """Ported from the research code (which printed the splits)."""
    for case, config in CONFIGS.items():
        splitter = CVSplit(**config)
        assert splitter.get_n_splits() == N_SPLITS[case]

        splits = list(splitter.split())
        assert len(splits) == N_SPLITS[case]
        for row_train, row_test, column_X, column_Y in splits:
            assert set(row_train).isdisjoint(row_test)
            assert set(column_X).isdisjoint(column_Y)

    splitter_1 = CVSplit(**CONFIGS[1])
    splitter_5 = CVSplit(**CONFIGS[5])
    splitter_1.save(str(tmp_path / "case_1"))
    splitter_5.save(str(tmp_path / "case_5"))

    folds_1 = read_folds(tmp_path / "case_1")
    folds_5 = read_folds(tmp_path / "case_5")
    assert len(folds_1) == 10
    assert len(folds_5) == 3
    assert set(folds_1[0]) == {"row_train", "row_test", "column_X", "column_Y"}
    assert set(folds_5[0]) == {"row_train", "row_test"}


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("random_state", [None, 0, 42])
def test_CVSplit_matches_reference(tmp_path, case, random_state):
    config = dict(CONFIGS[case])
    if random_state is not None:
        if any(
            config[key]["method"] == "KFold"
            for key in ["split_row", "split_column"]
            if key in config
        ):
            pytest.skip("KFold without shuffling does not accept random_state")
        config["random_state"] = random_state
    elif any(
        config[key]["method"] == "ShuffleSplit"
        for key in ["split_row", "split_column"]
        if key in config
    ):
        pytest.skip("ShuffleSplit without random_state is not reproducible")

    CVSplit(**config).save(str(tmp_path / "new"))
    ref.CVSplit(**config).save(str(tmp_path / "old"))
    assert read_folds(tmp_path / "new") == read_folds(tmp_path / "old")


def test_CVSplit_hcp_config_matches_reference(tmp_path):
    config = {
        "split_row": {
            "n_samples": 1003,
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 100, "train_size": 0.8},
        },
        "split_column": {
            "n_samples": 50,
            "method": "ShuffleSplit",
            "method_kwargs": {"n_splits": 100, "train_size": 0.5},
        },
        "strategy": "pairing",
        "random_state": 0,
    }
    CVSplit(**config).save(str(tmp_path / "new"))
    ref.CVSplit(**config).save(str(tmp_path / "old"))
    assert read_folds(tmp_path / "new") == read_folds(tmp_path / "old")


def test_load_fold(tmp_path):
    fold = {"row_train": [3, 1], "row_test": [0], "column_X": [2], "column_Y": [1, 0]}
    with open(tmp_path / "fold.json", "w") as file:
        json.dump(fold, file)

    loaded = load_fold(str(tmp_path / "fold.json"), n_sessions=4, n_channels=3)
    assert loaded == {
        "row_train": [1, 3],
        "row_test": [0],
        "column_X": [2],
        "column_Y": [0, 1],
    }

    with pytest.raises(ValueError, match="overlap"):
        load_fold({**fold, "row_test": [1]})
    with pytest.raises(ValueError, match="outside"):
        load_fold(fold, n_sessions=3)
    with pytest.raises(ValueError, match="missing"):
        load_fold({"row_train": [0], "row_test": [1]})
    with pytest.raises(ValueError, match="empty"):
        load_fold({**fold, "column_X": []})
