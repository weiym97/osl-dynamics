"""Repeated training and split-half reproducibility."""

import json
from math import sqrt

import numpy as np
import numpy.testing as npt
from scipy.optimize import linear_sum_assignment

import reference_wip as ref
from osl_dynamics.data import Data
from osl_dynamics.evaluate import repeat, split_half
from osl_dynamics.evaluate.reproducibility import pairwise_riemannian_distances

SWC_CONFIG = {
    "n_states": 3,
    "window_length": 50,
    "window_offset": 25,
    "random_state": 0,
}


def swc_data(seed=0):
    rng = np.random.default_rng(seed)
    covs = [np.eye(4), 0.5 * np.eye(4) + 0.5, np.diag([3.0, 1.0, 0.5, 2.0])]
    sessions = []
    for _ in range(6):
        blocks = [
            rng.multivariate_normal(np.zeros(4), covs[k], 100) for k in [0, 1, 2] * 3
        ]
        sessions.append(np.concatenate(blocks))
    return Data(sessions)


def test_regularisation():
    """Ported from tests/test_inference/test_metrics.py.

    Matrices are regularised by adding eps to the diagonal.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 3, 3))
    matrices = A @ A.transpose(0, 2, 1) + np.eye(3)
    for eps in [1e-6, 1e-3]:
        expected = ref.twopair_riemannian_distance(
            matrices, matrices[::-1], eps_value=[eps]
        )
        result = pairwise_riemannian_distances(
            matrices, matrices[::-1], eps_values=[eps]
        )
        npt.assert_allclose(result, expected)


def test_twopair_riemannian_distance():
    """Ported from tests/test_inference/test_metrics.py."""
    corr_1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    corr_2 = np.array([[2.0, 0.0], [0.0, 1.0]])
    corr_3 = np.array([[1.0, 0.0], [0.0, 2.0]])
    matrices_1 = np.stack([corr_1, corr_2])
    matrices_2 = np.stack([corr_1, corr_3])
    answer = np.array([[0.0, np.log(2)], [np.log(2), sqrt(2) * np.log(2)]])
    npt.assert_almost_equal(
        pairwise_riemannian_distances(matrices_1, matrices_2), answer, decimal=6
    )


def test_hungarian_pair():
    """Ported from tests/test_inference/test_modes.py.

    split_half pairs states with linear_sum_assignment on the distances.
    """
    matrix = np.array([[8, 4, 7], [5, 2, 3], [9, 4, 8]])
    row, col = linear_sum_assignment(matrix)
    indices, matrix_reordered = ref.hungarian_pair(matrix, distance=True)
    npt.assert_equal(row, [0, 1, 2])
    npt.assert_equal(col, [0, 2, 1])
    npt.assert_equal(row, indices["row"])
    npt.assert_equal(col, indices["col"])
    npt.assert_equal(matrix[row, col], np.diagonal(matrix_reordered))


def test_split_half_matches_reference(tmp_path):
    data = swc_data()
    fold = {"row_train": [0, 2, 4], "row_test": [1, 3, 5]}
    distance = split_half("swc", data, fold, str(tmp_path), config_kwargs=SWC_CONFIG)

    covs_1 = np.load(tmp_path / "partition_1" / "inf_params" / "covs.npy")
    covs_2 = np.load(tmp_path / "partition_2" / "inf_params" / "covs.npy")
    npt.assert_allclose(distance, ref.reproducibility_analysis(covs_1, covs_2))

    with open(tmp_path / "reproducibility.json") as file:
        result = json.load(file)
    assert result["riemannian_distance"] == distance
    assert sorted(result["pairing"]["col"]) == [0, 1, 2]


def test_split_half_static(tmp_path):
    data = swc_data()
    fold = {"row_train": [0, 1, 2], "row_test": [3, 4, 5]}
    distance = split_half(
        "hmm", data, fold, str(tmp_path), config_kwargs={"n_states": 1}
    )
    covs_1 = np.load(tmp_path / "partition_1" / "inf_params" / "covs.npy")
    covs_2 = np.load(tmp_path / "partition_2" / "inf_params" / "covs.npy")
    assert covs_1.shape == (1, 4, 4)
    npt.assert_allclose(distance, ref.reproducibility_analysis(covs_1, covs_2))


def test_repeat(tmp_path):
    data = swc_data()
    metrics = repeat("swc", data, str(tmp_path), config_kwargs=SWC_CONFIG)
    assert set(metrics) == {"log_likelihood"}
    assert (tmp_path / "model" / "config.yml").exists()
    assert (tmp_path / "inf_params" / "covs.npy").exists()
    assert (tmp_path / "inf_params" / "alp.pkl").exists()
    with open(tmp_path / "metrics" / "metrics.json") as file:
        assert json.load(file) == metrics

    assert (
        repeat("swc", data, str(tmp_path / "static"), config_kwargs={"n_states": 1})
        == {}
    )
    assert (tmp_path / "static" / "inf_params" / "covs.npy").exists()
