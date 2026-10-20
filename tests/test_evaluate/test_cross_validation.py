"""Bi-cross-validation steps, ported from the research code's tests.

Each test uses the same data, cases and tolerances as the corresponding
test in tests/test_evaluate/test_cross_validation.py (and
tests/test_config_api/test_wrappers.py) of the WIP branch. Changes:

- The model operations are called directly instead of through the
  CVHMM/CVSWC/CVDyNeMo classes and YAML configs.
- Random seeds are set so the tests are reproducible.
- Spatial parameters of a single channel are squeezed before comparison,
  because the old dual estimation squeezed them.
- The log-likelihood is averaged over time points (or windows for SWC),
  the definition since commits 3829b6c0 and 09128fd2. The expected values
  of the calculate_error tests used the previous definition (summed over
  time points and averaged over sessions), so they are divided by the
  number of time points per session (or the window length for SWC).
"""

import numpy as np
import numpy.testing as npt
import pytest

from osl_dynamics.data import Data
from osl_dynamics.evaluate.model_ops import get_model_ops, subset_data


def generate_obs(cov, mean=None, n_timepoints=100):
    if mean is None:
        mean = np.zeros(len(cov))
    return np.random.multivariate_normal(mean, cov, n_timepoints)


def multivariate_gaussian_log_likelihood(x, mu, cov):
    d = len(mu)
    log_det_cov = np.log(np.linalg.det(cov))
    quad_term = np.sum((x - mu) @ np.linalg.inv(cov) * (x - mu), axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + log_det_cov + quad_term)


def swc_window_log_likelihood(x, mean, cov):
    """Average log-likelihood of the time points in a window."""
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    n = x.shape[0]
    d = x.shape[1]
    term1, term2 = 0, 0
    for i in range(n):
        diff = x[i] - mean
        term1 += -0.5 * np.log((2 * np.pi) ** d * det_cov)
        term2 += -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
    return (term1 + term2) / n


def irrelevant_session():
    return generate_obs(np.eye(3) * 100, n_timepoints=300000)


def hmm_ops(spatial_method=None, **config_kwargs):
    defaults = {
        "n_states": 3,
        "learn_means": False,
        "learn_covariances": True,
        "learning_rate": 0.01,
        "n_epochs": 3,
        "sequence_length": 600,
        # The old tests did not pass batch_size, so they used the default of
        # the research code's train_hmm (32). The current default is 256.
        "batch_size": 32,
    }
    return get_model_ops(
        "hmm",
        {**defaults, **config_kwargs},
        init_kwargs={"n_init": 1, "n_epochs": 1},
        spatial_method=spatial_method,
    )


def dynemo_ops(**config_kwargs):
    defaults = {
        "batch_size": 64,
        "do_kl_annealing": True,
        "inference_n_units": 64,
        "inference_normalization": "layer",
        "initial_alpha_temperature": 1.0,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 5,
        "learn_alpha_temperature": True,
        "learn_covariances": True,
        "learn_means": True,
        "learning_rate": 0.01,
        "model_n_units": 64,
        "model_normalization": "layer",
        "n_epochs": 30,
        "n_kl_annealing_epochs": 10,
        "n_modes": 2,
    }
    init_kwargs = config_kwargs.pop("init_kwargs", {"n_init": 10, "n_epochs": 2})
    return get_model_ops("dynemo", {**defaults, **config_kwargs}, init_kwargs)


def hmm_split_data(i, means_X, vars_X, covs_Y, n_timepoints=100):
    """Session where Y has states i and i + 1 and X is a single channel."""
    obs = []
    for j in range(1500):
        observations_Y = [
            generate_obs(covs_Y[i], n_timepoints=n_timepoints),
            generate_obs(covs_Y[i + 1], n_timepoints=n_timepoints),
        ]
        observations_X = [
            generate_obs([[vars_X[i]]], [means_X[i]], n_timepoints=n_timepoints),
            generate_obs(
                [[vars_X[i + 1]]], [means_X[i + 1]], n_timepoints=n_timepoints
            ),
        ]
        obs.append(
            np.concatenate(
                [
                    np.hstack((Y[:, :1], X, Y[:, 1:]))
                    for X, Y in zip(observations_X, observations_Y)
                ],
                axis=0,
            )
        )
    return np.concatenate(obs, axis=0)


def dynemo_session(means_A, covs_A, means_B, vars_B, timepoints=100):
    """Session with two modes mixed by sin^2 and cos^2.

    A is two channels (0 and 2) and B is one channel (1).
    """
    obs = []
    alphas = []
    for j in range(3000):
        t = np.linspace(0, timepoints - 1, timepoints) / timepoints
        alpha_t1 = np.sin(2 * np.pi * t) ** 2
        alpha_t2 = np.cos(2 * np.pi * t) ** 2
        alphas.append(np.stack((alpha_t1, alpha_t2), axis=1))
        A_mean_t = np.outer(alpha_t1, means_A[0]) + np.outer(alpha_t2, means_A[1])
        A_cov_t = np.einsum("t,ij->tij", alpha_t1, covs_A[0]) + np.einsum(
            "t,ij->tij", alpha_t2, covs_A[1]
        )
        A_obs = np.array(
            [
                np.random.multivariate_normal(A_mean_t[t_idx], A_cov_t[t_idx])
                for t_idx in range(timepoints)
            ]
        )
        B_mean_t = alpha_t1 * means_B[0] + alpha_t2 * means_B[1]
        B_var_t = alpha_t1 * vars_B[0] + alpha_t2 * vars_B[1]
        B_obs = np.reshape(np.random.normal(B_mean_t, np.sqrt(B_var_t)), (-1, 1))
        obs.append(np.hstack((A_obs[:, :1], B_obs, A_obs[:, 1:])))
    return np.concatenate(obs, axis=0), np.concatenate(alphas, axis=0)


# --------------------------------------------------------------------------
# HMM
# --------------------------------------------------------------------------


def test_full_train():
    np.random.seed(0)
    n_states = 3
    row_train = [1, 2]
    column_Y = [0, 2]

    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]
    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]

    sessions = [irrelevant_session()]
    sessions += [hmm_split_data(i, means_X, vars_X, covs_Y) for i in range(2)]
    data = Data(sessions)

    ops = hmm_ops(n_states=n_states)
    spatial, _ = ops.fit(subset_data(data, row_train, column_Y))
    result_means, result_covs = spatial["means"], spatial["covs"]

    npt.assert_array_equal(result_means, np.zeros((n_states, len(column_Y))))

    # Assert diagonal elements are all one
    npt.assert_allclose(
        np.diagonal(result_covs, axis1=-2, axis2=-1), 1.0, rtol=0.05, atol=0.05
    )

    # Assert off-diagonal elements are equal to cors
    off_diagonal = np.array([float(result_covs[i, 0, 1]) for i in range(n_states)])
    npt.assert_allclose(np.sort(off_diagonal), cors_Y, atol=0.05, rtol=0.05)


def test_infer_spatial():
    np.random.seed(0)
    n_states = 3
    row_train = [1, 2]
    column_X = [1]

    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]
    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]
    n_timepoints = 100

    sessions = [irrelevant_session()]
    hidden_states = []
    for i in range(0, 2):
        hv_temp = np.zeros((n_timepoints * 2, n_states))
        hv_temp[:, i] = np.array([0.6] * n_timepoints + [0.4] * n_timepoints)
        hv_temp[:, i + 1] = np.array([0.4] * n_timepoints + [0.6] * n_timepoints)
        hidden_states.append(np.tile(hv_temp, (1500, 1)))
        sessions.append(hmm_split_data(i, means_X, vars_X, covs_Y, n_timepoints))
    data = Data(sessions)

    ops = hmm_ops(n_states=n_states, learn_means=True)
    result = ops.infer_spatial(subset_data(data, row_train, column_X), hidden_states)

    npt.assert_allclose(means_X, np.squeeze(result["means"]), rtol=1e-2, atol=1e-2)
    npt.assert_allclose(vars_X, np.squeeze(result["covs"]), rtol=1e-2, atol=1e-2)


def test_infer_spatial_hmm():
    """Ported from tests/test_config_api/test_wrappers.py."""
    np.random.seed(0)
    n_states = 3
    row_train = [1, 2]
    column_X = [1]

    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]
    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]
    means_X_estimate = [1.4, 2, 2.6]
    vars_X_estimate = [0.94, 1.5, 1.84]
    n_timepoints = 100

    sessions = [irrelevant_session()]
    hidden_states = []
    for i in range(0, 2):
        hv_temp = np.zeros((n_timepoints * 2, n_states))
        hv_temp[:, i] = np.array([0.6] * n_timepoints + [0.4] * n_timepoints)
        hv_temp[:, i + 1] = np.array([0.4] * n_timepoints + [0.6] * n_timepoints)
        hidden_states.append(np.tile(hv_temp, (1500, 1)))
        sessions.append(hmm_split_data(i, means_X, vars_X, covs_Y, n_timepoints))
    data = Data(sessions, load_memmaps=False)
    X_train = subset_data(data, row_train, column_X)

    ops = hmm_ops(n_states=n_states, learn_means=True, spatial_method="mle")
    result = ops.infer_spatial(X_train, hidden_states)
    npt.assert_allclose(
        means_X_estimate, np.squeeze(result["means"]), rtol=1e-2, atol=1e-2
    )
    npt.assert_allclose(
        vars_X_estimate, np.squeeze(result["covs"]), rtol=1e-2, atol=1e-2
    )

    ops = hmm_ops(n_states=n_states, learn_means=True, spatial_method="sample")
    result = ops.infer_spatial(X_train, hidden_states)
    npt.assert_allclose(means_X, np.squeeze(result["means"]), rtol=1e-2, atol=1e-2)
    npt.assert_allclose(vars_X, np.squeeze(result["covs"]), rtol=1e-2, atol=1e-2)


def test_infer_temporal():
    np.random.seed(0)
    n_states = 3
    row_test = [1, 2]
    column_X = [0, 2]

    means_X = [np.array([-10.0, -10.0]), np.array([0.0, 0.0]), np.array([10.0, 10.0])]
    cors_X = [-0.5, 0.0, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]
    means_Y = [1.0, 2.0, 3.0]
    vars_Y = [0.5, 1.0, 2.0]
    spatial_X_train = {"means": np.array(means_X), "covs": np.stack(covs_X)}

    sessions = [irrelevant_session()]
    hidden_states = []
    n_timepoints = 100
    for i in range(0, 2):
        hv_temp = np.zeros((n_timepoints * 2, n_states))
        hv_temp[:, i] = np.array([1.0] * n_timepoints + [0.0] * n_timepoints)
        hv_temp[:, i + 1] = np.array([0.0] * n_timepoints + [1.0] * n_timepoints)
        hidden_states.append(np.tile(hv_temp, (1500, 1)))

        obs = []
        for j in range(1500):
            observations_X = [
                generate_obs(covs_X[i], means_X[i], n_timepoints),
                generate_obs(covs_X[i + 1], means_X[i + 1], n_timepoints),
            ]
            observations_Y = [
                generate_obs([[vars_Y[i]]], [means_Y[i]], n_timepoints),
                generate_obs([[vars_Y[i + 1]]], [means_Y[i + 1]], n_timepoints),
            ]
            obs.append(
                np.concatenate(
                    [
                        np.hstack((X[:, :1], Y, X[:, 1:]))
                        for X, Y in zip(observations_X, observations_Y)
                    ],
                    axis=0,
                )
            )
        sessions.append(np.concatenate(obs, axis=0))
    data = Data(sessions)

    ops = hmm_ops(n_states=n_states, n_epochs=10)
    alpha = ops.infer_temporal(subset_data(data, row_test, column_X), spatial_X_train)

    # The old test only checked the first session
    for i in range(2):
        npt.assert_allclose(alpha[i], hidden_states[i], atol=1e-6)


def test_calculate_error():
    row_test = [1, 2]
    column_Y = [0, 2]

    data_1 = np.zeros((2, 3))
    data_2 = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    data_3 = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    data = Data([data_1, data_2, data_3])

    means = np.zeros((3, 2))
    covs = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.8], [0.8, 1.5]],
            [[0.5, -0.25], [-0.25, 0.5]],
        ]
    )
    spatial = {"means": means, "covs": covs}
    alpha = [
        np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
        np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
    ]

    ops = hmm_ops()
    result = ops.log_likelihood(subset_data(data, row_test, column_Y), alpha, spatial)

    ll_1 = multivariate_gaussian_log_likelihood(
        data_2[:1, [0, 2]], np.array([0, 0]), covs[0]
    )
    ll_2 = 0.5 * multivariate_gaussian_log_likelihood(
        data_2[1:2, [0, 2]], np.array([0, 0]), covs[1]
    ) + 0.5 * multivariate_gaussian_log_likelihood(
        data_2[1:2, [0, 2]], np.array([0, 0]), covs[2]
    )
    ll_3 = 0.5 * multivariate_gaussian_log_likelihood(
        data_3[:1, [0, 2]], np.array([0, 0]), covs[0]
    ) + 0.5 * multivariate_gaussian_log_likelihood(
        data_3[:1, [0, 2]], np.array([0, 0]), covs[1]
    )
    ll_4 = multivariate_gaussian_log_likelihood(
        data_3[1:2, [0, 2]], np.array([0, 0]), covs[2]
    )

    # Old expected value: (ll_1 + ll_2 + ll_3 + ll_4) / 2, i.e. summed over
    # the 2 time points and averaged over sessions. Now: averaged over time
    # points.
    n_timepoints = 2
    ll = (ll_1 + ll_2 + ll_3 + ll_4) / 2 / n_timepoints
    npt.assert_almost_equal(ll, result, decimal=3)


# --------------------------------------------------------------------------
# SWC
# --------------------------------------------------------------------------

SWC_DATA_1 = np.array(
    [
        [1.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0],
        [2.0, 0.0, -2.0],
        [-2.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 10.0],
        [-10.0, 0.0, -10.0],
    ]
)
SWC_DATA_2 = np.array(
    [
        [2.0, 0.0, -2.0],
        [-2.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 10.0],
        [-10.0, 0.0, -10.0],
        [0.0, 0.0, 3.0],
        [-3.0, 0.0, 0.0],
        [3.0, 0.0, -3.0],
    ]
)


def swc_ops(**config_kwargs):
    defaults = {
        "n_states": 3,
        "learn_means": False,
        "learn_covariances": True,
        "window_length": 100,
        "window_offset": 100,
        "random_state": 0,
    }
    return get_model_ops("swc", {**defaults, **config_kwargs})


def test_swc_full_train():
    np.random.seed(0)
    n_states = 3
    row_train = [1, 2]
    column_Y = [0, 2]

    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]
    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]

    sessions = [irrelevant_session()]
    sessions += [hmm_split_data(i, means_X, vars_X, covs_Y) for i in range(2)]
    data = Data(sessions)

    spatial, _ = swc_ops().fit(subset_data(data, row_train, column_Y))
    result_means, result_covs = spatial["means"], spatial["covs"]

    npt.assert_array_equal(result_means, np.zeros((n_states, len(column_Y))))
    npt.assert_allclose(
        np.diagonal(result_covs, axis1=-2, axis2=-1), 1.0, rtol=0.05, atol=0.05
    )
    off_diagonal = np.array([float(result_covs[i, 0, 1]) for i in range(n_states)])
    npt.assert_allclose(np.sort(off_diagonal), cors_Y, atol=0.05, rtol=0.05)


def test_swc_infer_spatial():
    np.random.seed(0)
    row_train = [1, 2]
    column_X = [1]

    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]
    means_X = [0.0, 0.0, 0.0]
    vars_X = [0.5, 1.0, 2.0]

    sessions = [irrelevant_session()]
    hidden_states = []
    for i in range(0, 2):
        hidden_states.append(np.tile([i, i + 1], 1500))
        sessions.append(hmm_split_data(i, means_X, vars_X, covs_Y))
    data = Data(sessions)

    result = swc_ops().infer_spatial(
        subset_data(data, row_train, column_X), hidden_states
    )
    npt.assert_allclose(means_X, np.squeeze(result["means"]), rtol=1e-2, atol=1e-2)
    npt.assert_allclose(vars_X, np.squeeze(result["covs"]), rtol=1e-2, atol=1e-2)


def test_swc_infer_temporal():
    np.random.seed(0)
    row_test = [1, 2]
    column_X = [0, 2]

    means_X = [np.array([-10.0, -10.0]), np.array([0.0, 0.0]), np.array([10.0, 10.0])]
    cors_X = [-0.5, 0.0, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]
    means_Y = [1.0, 2.0, 3.0]
    vars_Y = [0.5, 1.0, 2.0]
    spatial_X_train = {"means": np.array(means_X), "covs": np.stack(covs_X)}

    sessions = [irrelevant_session()]
    hidden_states = []
    n_timepoints = 100
    for i in range(0, 2):
        hidden_states.append(np.tile([i, i + 1], 1500))
        obs = []
        for j in range(1500):
            observations_X = [
                generate_obs(covs_X[i], means_X[i], n_timepoints),
                generate_obs(covs_X[i + 1], means_X[i + 1], n_timepoints),
            ]
            observations_Y = [
                generate_obs([[vars_Y[i]]], [means_Y[i]], n_timepoints),
                generate_obs([[vars_Y[i + 1]]], [means_Y[i + 1]], n_timepoints),
            ]
            obs.append(
                np.concatenate(
                    [
                        np.hstack((X[:, :1], Y, X[:, 1:]))
                        for X, Y in zip(observations_X, observations_Y)
                    ],
                    axis=0,
                )
            )
        sessions.append(np.concatenate(obs, axis=0))
    data = Data(sessions)

    alpha = swc_ops().infer_temporal(
        subset_data(data, row_test, column_X), spatial_X_train
    )

    # The old test only checked the first session
    for i in range(2):
        npt.assert_allclose(alpha[i], hidden_states[i], atol=1e-6)


def test_swc_calculate_error():
    row_test = [1, 2]
    column_Y = [0, 2]
    data = Data([np.zeros((8, 3)), SWC_DATA_1, SWC_DATA_2])

    means = np.zeros((3, 2))
    covs = np.array(
        [
            [[2.5, -2.4999995], [-2.4999995, 2.5]],
            [[52.0, 48.0], [48.0, 52.0]],
            [[54.5, 47.75], [47.75, 54.5]],
        ]
    )
    alpha = [np.array([0, 1]), np.array([1, 2])]

    ops = swc_ops(window_length=5, window_offset=3)
    result = ops.log_likelihood(
        subset_data(data, row_test, column_Y), alpha, {"means": means, "covs": covs}
    )

    true_data_1 = SWC_DATA_1[:, [0, 2]]
    true_data_2 = SWC_DATA_2[:, [0, 2]]
    log_likelihood_1 = swc_window_log_likelihood(true_data_1[:5], means[0], covs[0])
    log_likelihood_2 = swc_window_log_likelihood(true_data_1[3:], means[1], covs[1])
    log_likelihood_3 = swc_window_log_likelihood(true_data_2[3:], means[2], covs[2])

    # Old expected value: (...) / 4 * 5. Now: averaged over windows.
    log_likelihood_answer = (
        log_likelihood_1 + 2 * log_likelihood_2 + log_likelihood_3
    ) / 4
    npt.assert_almost_equal(log_likelihood_answer, result)


# --------------------------------------------------------------------------
# DyNeMo
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_full_train_DyNeMo():
    np.random.seed(0)
    row_train = [1, 2]
    column_Y = [0, 2]

    cors_Y = [-0.5, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]
    means_Y = [[1.0, 0.0], [0.0, 1.0]]
    means_X = [1.0, 2.0]
    vars_X = [0.5, 2.0]

    sessions = [irrelevant_session()]
    alpha_truth = []
    for i in range(0, 2):
        obs, alpha = dynemo_session(means_Y, covs_Y, means_X, vars_X)
        sessions.append(obs)
        alpha_truth.append(alpha)
    data = Data(sessions)

    ops = dynemo_ops(
        initial_means=np.array(means_Y),
        initial_covariances=np.stack(covs_Y),
        learning_rate=0.001,
        sequence_length=100,
        init_kwargs={"n_init": 1, "n_epochs": 1},
    )
    spatial, alpha = ops.fit(subset_data(data, row_train, column_Y))

    npt.assert_allclose(spatial["means"], np.array(means_Y), rtol=1e-6, atol=1e-6)
    npt.assert_allclose(spatial["covs"], np.stack(covs_Y), rtol=1e-6, atol=1e-6)

    for truth, inferred in zip(alpha_truth, alpha):
        mean_difference = np.mean(np.abs(truth - inferred))
        npt.assert_array_less(
            mean_difference,
            5e-2,
            err_msg=f"Mean difference {mean_difference} exceeds 5e-2",
        )


@pytest.mark.slow
def test_infer_spatial_DyNeMo():
    np.random.seed(0)
    row_train = [1, 2]
    column_X = [0, 2]

    cors_X = [-0.5, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]
    means_X = [[1.0, -1.0], [-1.0, 1.0]]
    means_Y = [1.0, 2.0]
    vars_Y = [0.5, 2.0]

    sessions = [irrelevant_session()]
    alpha_truth = []
    for i in range(0, 2):
        obs, alpha = dynemo_session(means_X, covs_X, means_Y, vars_Y)
        sessions.append(obs)
        alpha_truth.append(alpha)
    data = Data(sessions)

    ops = dynemo_ops(sequence_length=150)
    result = ops.infer_spatial(subset_data(data, row_train, column_X), alpha_truth)

    npt.assert_allclose(result["means"], np.array(means_X), rtol=1e-2, atol=1e-2)
    npt.assert_allclose(result["covs"], np.stack(covs_X), rtol=3e-2, atol=3e-2)


@pytest.mark.slow
def test_infer_temporal_DyNeMo():
    np.random.seed(0)
    row_test = [1, 2]
    column_X = [0, 2]

    cors_X = [-0.5, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]
    means_X = [[1.0, -1.0], [-1.0, 1.0]]
    means_Y = [1.0, 2.0]
    vars_Y = [0.5, 2.0]

    sessions = [irrelevant_session()]
    alpha_truth = []
    for i in range(0, 2):
        obs, alpha = dynemo_session(means_X, covs_X, means_Y, vars_Y)
        sessions.append(obs)
        alpha_truth.append(alpha)
    data = Data(sessions)
    spatial_X_train = {"means": np.array(means_X), "covs": np.stack(covs_X)}

    ops = dynemo_ops(sequence_length=100)
    inferred_alpha = ops.infer_temporal(
        subset_data(data, row_test, column_X), spatial_X_train
    )

    for truth, inferred in zip(alpha_truth, inferred_alpha):
        mean_difference = np.mean(np.abs(truth - inferred))
        npt.assert_array_less(
            mean_difference,
            5e-2,
            err_msg=f"Mean difference {mean_difference} exceeds 5e-2",
        )


def test_calculate_error_DyNeMo():
    row_test = [1, 2]
    column_Y = [0, 2]

    data_1 = np.zeros((2, 3))
    data_2 = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    data_3 = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    data = Data([data_1, data_2, data_3])

    means = np.zeros((2, 2))
    covs = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.8], [0.8, 2.0]]])

    # The linear combination of means and covs gives 3 "states"
    covs_states = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.4], [0.4, 1.5]],
            [[1.2, 0.16], [0.16, 1.2]],
        ]
    )
    # The data occupy state 1, 2, 3, 1
    alpha = [np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([[0.8, 0.2], [1.0, 0.0]])]

    ops = dynemo_ops(sequence_length=100)
    result = ops.log_likelihood(
        subset_data(data, row_test, column_Y), alpha, {"means": means, "covs": covs}
    )

    ll_1 = multivariate_gaussian_log_likelihood(
        data_2[:1, [0, 2]], np.array([0, 0]), covs_states[0]
    )
    ll_2 = multivariate_gaussian_log_likelihood(
        data_2[1:2, [0, 2]], np.array([0, 0]), covs_states[1]
    )
    ll_3 = multivariate_gaussian_log_likelihood(
        data_3[:1, [0, 2]], np.array([0, 0]), covs_states[2]
    )
    ll_4 = multivariate_gaussian_log_likelihood(
        data_3[1:2, [0, 2]], np.array([0, 0]), covs_states[0]
    )

    # Old expected value: (ll_1 + ll_2 + ll_3 + ll_4) / 2. Now: averaged over
    # the 2 time points.
    n_timepoints = 2
    ll = (ll_1 + ll_2 + ll_3 + ll_4) / 2 / n_timepoints
    npt.assert_almost_equal(ll, result, decimal=3)
