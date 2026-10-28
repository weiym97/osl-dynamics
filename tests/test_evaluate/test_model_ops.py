"""Equivalence with the research code and guards against data leakage."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import multivariate_normal

import reference_wip as ref
from osl_dynamics.data import Data
from osl_dynamics.evaluate import BCV
from osl_dynamics.evaluate import cross_validation
from osl_dynamics.evaluate.model_ops import (
    fit_static,
    get_model_ops,
    static_log_likelihood,
    subset_data,
)


def f32(arrays):
    return [np.asarray(a, dtype=np.float32) for a in arrays]


def random_covs(rng, n, m):
    A = rng.normal(size=(n, m, m))
    return A @ A.transpose(0, 2, 1) + m * np.eye(m)


# --------------------------------------------------------------------------
# Equivalence with the research code
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sample", "mle"])
@pytest.mark.parametrize("learn_means", [False, True])
def test_hmm_dual_estimation_matches_reference(method, learn_means):
    rng = np.random.default_rng(0)
    # Data stores float32 arrays, so the reference gets the same values
    ts = f32([rng.normal(size=(500, 4)), rng.normal(size=(300, 4))])
    alpha = [rng.dirichlet(np.ones(3), size=500), rng.dirichlet(np.ones(3), size=300)]
    data = Data(ts)

    ops = get_model_ops(
        "hmm",
        {"n_states": 3, "learn_means": learn_means, "learn_covariances": True},
        spatial_method=method,
    )
    result = ops.infer_spatial(data, alpha)

    means, covs = ref.hmm_dual_estimation(
        ts,
        alpha,
        n_channels=4,
        learn_means=learn_means,
        initial_means=np.zeros((3, 4)),
        covariances_epsilon=1e-6,
        method=method,
    )
    npt.assert_allclose(result["means"], means, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(result["covs"], covs, rtol=1e-10, atol=1e-10)


def test_hmm_dual_estimation_empty_state_matches_reference():
    rng = np.random.default_rng(1)
    ts = f32([rng.normal(size=(400, 3))])
    # State 2 is never the most probable state
    alpha = rng.dirichlet(np.ones(3), size=400)
    alpha[:, 2] = 0
    alpha /= alpha.sum(axis=1, keepdims=True)
    data = Data(ts)

    ops = get_model_ops(
        "hmm", {"n_states": 3, "learn_means": False, "learn_covariances": True}
    )
    result = ops.infer_spatial(data, [alpha])
    _, covs = ref.hmm_dual_estimation(
        ts, [alpha], 3, False, np.zeros((3, 3)), 1e-6, method="sample"
    )
    npt.assert_allclose(result["covs"], covs, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(result["covs"][2], (1 + 1e-6) * np.eye(3))


def test_hmm_log_likelihood_matches_reference():
    rng = np.random.default_rng(2)
    n_sessions, n_samples, n_channels, n_states = 3, 200, 4, 3
    x = rng.normal(size=(n_sessions, n_samples, n_channels))
    gamma = rng.dirichlet(np.ones(n_states), size=(n_sessions, n_samples))
    means = np.zeros((n_states, n_channels))
    covs = random_covs(rng, n_states, n_channels)

    ops = get_model_ops(
        "hmm",
        {"n_states": n_states, "learn_means": False, "learn_covariances": True},
    )
    result = ops.log_likelihood(
        Data(list(x)), list(gamma), {"means": means, "covs": covs}
    )

    # The research code used the model's log-likelihood of each state
    ll = np.stack(
        [
            np.stack(
                [
                    multivariate_normal.logpdf(xi, means[k], covs[k])
                    for k in range(n_states)
                ],
                axis=-1,
            )
            for xi in x
        ]
    )
    expected = ref.hmm_average_log_likelihood(ll, gamma)
    npt.assert_allclose(result, expected, rtol=1e-5)


def test_dynemo_log_likelihood_matches_reference():
    rng = np.random.default_rng(3)
    n_sessions, n_samples, n_channels, n_modes = 2, 300, 4, 3
    x = rng.normal(size=(n_sessions, n_samples, n_channels)).astype(np.float32)
    alpha = rng.dirichlet(np.ones(n_modes), size=(n_sessions, n_samples))
    means = rng.normal(size=(n_modes, n_channels))
    covs = random_covs(rng, n_modes, n_channels)

    ops = get_model_ops(
        "dynemo",
        {"n_modes": n_modes, "learn_means": True, "learn_covariances": True},
    )
    result = ops.log_likelihood(
        Data(list(x)), list(alpha), {"means": means, "covs": covs}
    )
    expected = ref.dynemo_average_log_likelihood(x, alpha, means, covs)
    npt.assert_allclose(result, expected, rtol=1e-10)


@pytest.mark.parametrize("learn_means", [False, True])
def test_static_matches_reference(learn_means):
    rng = np.random.default_rng(4)
    train = f32(
        [rng.normal(loc=0.3, size=(500, 3)), rng.normal(loc=0.3, size=(500, 3))]
    )
    test = f32([rng.normal(size=(200, 3)), rng.normal(size=(200, 3))])

    spatial = fit_static(Data(train), learn_means=learn_means)
    mean, cov = ref.estimate_gaussian_distribution(
        np.concatenate(train), nonzero_means=learn_means
    )
    # With learn_means=True the research code averaged and centered the
    # float32 data in float32, so it only agrees to float32 precision
    rtol = 1e-6 if learn_means else 1e-10
    npt.assert_allclose(spatial["means"], mean, rtol=rtol, atol=1e-12)
    npt.assert_allclose(spatial["covs"], cov, rtol=rtol)

    result = static_log_likelihood(Data(test), spatial)
    expected = ref.estimate_gaussian_log_likelihood(np.stack(test), mean, cov)
    npt.assert_allclose(result, expected, rtol=rtol)


def test_estimate_gaussian_distribution():
    """Ported from tests/test_array_ops/test_array_ops.py."""
    data = Data([np.array([[1.0, -1.0], [3.0, 5.0], [5.0, 8.0]])])

    spatial = fit_static(data, learn_means=True)
    npt.assert_almost_equal(spatial["means"], np.array([[3.0, 4.0]]))
    npt.assert_almost_equal(spatial["covs"], np.array([[[8 / 3, 6.0], [6.0, 14.0]]]))

    spatial = fit_static(data, learn_means=False)
    npt.assert_almost_equal(spatial["means"], np.array([[0.0, 0.0]]))
    npt.assert_almost_equal(spatial["covs"], np.array([[[8 / 3, 6.0], [6.0, 14.0]]]))


def test_estimate_gaussian_log_likelihood():
    """Ported from tests/test_array_ops/test_array_ops.py."""
    means = np.zeros((1, 2))
    covs = np.array([[[1.5, 0.8], [0.8, 1.5]]])
    data = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]).T

    ll = static_log_likelihood(Data([data]), {"means": means, "covs": covs})
    answer = np.mean(
        multivariate_normal.logpdf(data, np.squeeze(means), np.squeeze(covs))
    )
    npt.assert_almost_equal(answer, ll)


def test_swc_matches_reference():
    rng = np.random.default_rng(5)
    window_length, window_offset = 30, 10
    ts = f32(
        [rng.normal(size=(300, 3)) * rng.uniform(0.5, 2, size=3) for _ in range(3)]
    )
    covs = random_covs(rng, 3, 3) / 3
    means = np.zeros((3, 3))
    data = Data(ts)

    ops = get_model_ops(
        "swc",
        {"n_states": 3, "window_length": window_length, "window_offset": window_offset},
    )
    labels = ops.infer_temporal(data, {"means": means, "covs": covs})
    expected = ref.swc_infer_temporal(ts, covs, window_length, window_offset)
    for a, b in zip(labels, expected):
        npt.assert_array_equal(a, b)

    result = ops.log_likelihood(data, labels, {"means": means, "covs": covs})
    expected = ref.swc_average_log_likelihood(
        ts, labels, means, covs, window_length, window_offset
    )
    npt.assert_allclose(result, expected, rtol=1e-10)


def test_mdynemo_log_likelihood_matches_model():
    import tensorflow as tf
    import tensorflow_probability as tfp

    from osl_dynamics.models import mdynemo

    rng = np.random.default_rng(6)
    n_samples, n_channels, n_modes = 50, 3, 2
    x = rng.normal(size=(n_samples, n_channels)).astype(np.float32)
    alpha = rng.dirichlet(np.ones(n_modes), size=n_samples).astype(np.float32)
    beta = rng.dirichlet(np.ones(n_modes), size=n_samples).astype(np.float32)
    means = rng.normal(size=(n_modes, n_channels))
    stds = np.array(
        [np.diag(rng.uniform(0.5, 2, size=n_channels)) for _ in range(n_modes)]
    )
    corrs = np.array(
        [np.corrcoef(rng.normal(size=(n_channels, 10))) for _ in range(n_modes)]
    )
    spatial = {"means": means, "stds": stds, "corrs": corrs}

    ops = get_model_ops(
        "mdynemo",
        {
            "n_modes": n_modes,
            "learn_means": True,
            "learn_stds": True,
            "learn_corrs": True,
            "sequence_length": n_samples,
        },
    )
    result = ops.log_likelihood(Data([x]), ([alpha], [beta]), spatial)

    # Mean and covariance from the model's own observation model layers
    model = mdynemo.Model(ops._config(n_channels, spatial))
    mu, E, R = model.get_means_stds_corrs()
    pca_mu, pca_E, pca_R = model.model.get_layer("pca")([mu, E, R])
    m = model.model.get_layer("mix_means")([alpha[None], pca_mu])
    G = model.model.get_layer("mix_stds")([alpha[None], pca_E])
    F = model.model.get_layer("mix_corrs")([beta[None], pca_R])
    C = model.model.get_layer("cov")([G, F, G])
    mvn = tfp.distributions.MultivariateNormalTriL(
        loc=m, scale_tril=tf.linalg.cholesky(C)
    )
    expected = np.mean(mvn.log_prob(x[None]).numpy())
    npt.assert_allclose(result, expected, rtol=1e-4)


def test_dynemo_mle_recovers_parameters():
    rng = np.random.default_rng(7)
    n_samples, n_channels = 20000, 2
    covs = np.array([[[1.0, -0.5], [-0.5, 1.0]], [[2.0, 1.0], [1.0, 2.0]]])
    t = np.arange(n_samples) / 100
    alpha = np.stack([np.sin(np.pi * t) ** 2, np.cos(np.pi * t) ** 2], axis=1)
    x = np.array(
        [
            rng.multivariate_normal(
                np.zeros(n_channels), np.einsum("k,kij->ij", a, covs)
            )
            for a in alpha
        ]
    )

    config = {
        "n_modes": 2,
        "learn_means": False,
        "learn_covariances": True,
        "sequence_length": 100,
        "batch_size": 32,
        "learning_rate": 0.01,
        "n_epochs": 20,
    }
    data = Data([x])
    mle = get_model_ops("dynemo", config, spatial_method="mle").infer_spatial(
        data, [alpha]
    )
    weighted = get_model_ops("dynemo", config, spatial_method="weighted").infer_spatial(
        data, [alpha]
    )
    npt.assert_allclose(mle["covs"], covs, atol=0.15)
    assert np.abs(mle["covs"] - covs).max() < np.abs(weighted["covs"] - covs).max()


# --------------------------------------------------------------------------
# Data leakage and correctness guards
# --------------------------------------------------------------------------


def simulated_hmm_data(seed=0):
    from osl_dynamics.simulation import HMM_MVN

    np.random.seed(seed)
    sim = HMM_MVN(
        n_samples=8000,
        n_states=3,
        n_channels=6,
        trans_prob="sequence",
        stay_prob=0.95,
        means="zero",
        covariances="random",
    )
    data = Data([x for x in sim.time_series.reshape(8, 1000, 6)])
    data.standardize()
    return sim, data


def test_shuffled_time_courses_lower_the_score():
    sim, data = simulated_hmm_data()
    stc = sim.state_time_course.reshape(8, 1000, 3)
    spatial = {"means": np.zeros((3, 6)), "covs": sim.obs_mod.covariances}

    ops = get_model_ops(
        "hmm", {"n_states": 3, "learn_means": False, "learn_covariances": True}
    )
    raw = Data([x for x in sim.time_series.reshape(8, 1000, 6)])
    ll = ops.log_likelihood(raw, list(stc), spatial)

    rng = np.random.default_rng(0)
    shuffled = [s[rng.permutation(s.shape[0])] for s in stc]
    assert ops.log_likelihood(raw, shuffled, spatial) < ll


def test_bcv_steps_use_the_right_sessions_and_channels(monkeypatch):
    _, data = simulated_hmm_data()
    fold = {
        "row_train": [0, 1, 2, 3, 4, 5],
        "row_test": [6, 7],
        "column_X": [0, 1],
        "column_Y": [2, 3, 4, 5],
    }

    # Record the subsets used by each step
    subsets = []
    original_subset_data = cross_validation.subset_data

    def recording_subset_data(data, rows, columns, store_dir=None):
        subsets.append((list(rows), list(columns)))
        return original_subset_data(data, rows, columns, store_dir)

    monkeypatch.setattr(cross_validation, "subset_data", recording_subset_data)

    # Record the data the time courses are inferred from
    from osl_dynamics.models import swc

    inferred = []
    original_get_alpha = swc.Model.get_alpha

    def recording_get_alpha(self, data, concatenate=False):
        inferred.append((len(data.keep), data.n_channels))
        return original_get_alpha(self, data, concatenate)

    monkeypatch.setattr(swc.Model, "get_alpha", recording_get_alpha)

    bcv = BCV(
        "swc",
        {"n_states": 3, "window_length": 50, "window_offset": 25, "random_state": 0},
    )
    bcv.validate(data, fold)

    assert subsets == [
        (fold["row_train"], fold["column_Y"]),
        (fold["row_train"], fold["column_X"]),
        (fold["row_test"], fold["column_X"]),
        (fold["row_test"], fold["column_Y"]),
    ]
    # Time courses are only inferred for the test sessions from channels X
    assert inferred == [(2, 2)]


def test_subset_data_only_contains_the_subset(tmp_path):
    arrays = [np.full((10, 4), i, dtype=np.float32) + np.arange(4) for i in range(5)]
    for load_memmaps in [False, True]:
        data = Data(arrays, load_memmaps=load_memmaps, store_dir=str(tmp_path))
        subset = subset_data(data, [1, 3], [0, 2], store_dir=str(tmp_path / "subset"))
        assert subset.n_sessions == 2
        assert subset.n_channels == 2
        npt.assert_array_equal(subset.arrays[0], arrays[1][:, [0, 2]])
        npt.assert_array_equal(subset.arrays[1], arrays[3][:, [0, 2]])
        subset.delete_dir()
        data.delete_dir()


def test_static_is_the_same_for_every_model():
    _, data = simulated_hmm_data()
    fold = {
        "row_train": [0, 1, 2, 3, 4, 5],
        "row_test": [6, 7],
        "column_X": [0, 2, 4],
        "column_Y": [1, 3, 5],
    }
    results = [
        BCV("hmm", {"n_states": 1}).validate(data, fold),
        BCV("dynemo", {"n_modes": 1}).validate(data, fold),
        BCV("swc", {"n_states": 1, "window_length": 50, "window_offset": 25}).validate(
            data, fold
        ),
    ]
    npt.assert_allclose(results, results[0])


@pytest.mark.parametrize("learn_means", [False, True])
def test_rare_states_get_an_identity_covariance(learn_means):
    """A state with too few time points for a full-rank covariance.

    The research code only used an identity for states with no time points,
    so this intentionally differs from it. States with just enough time
    points still match the research code.
    """
    rng = np.random.default_rng(8)
    n_channels = 4
    min_samples = n_channels + int(learn_means)
    x = f32([rng.normal(size=(500, n_channels))])

    # State 1 is the most probable state at min_samples - 1 time points and
    # state 2 at min_samples time points
    labels = np.zeros(500, dtype=int)
    labels[: min_samples - 1] = 1
    labels[100 : 100 + min_samples] = 2
    alpha = np.eye(3)[labels] * 0.8 + 0.1

    ops = get_model_ops(
        "hmm",
        {"n_states": 3, "learn_means": learn_means, "learn_covariances": True},
    )
    result = ops.infer_spatial(Data(x), [alpha])
    means, covs = ref.hmm_dual_estimation(
        x, [alpha], n_channels, learn_means, np.zeros((3, n_channels)), 1e-6
    )

    npt.assert_allclose(result["covs"][1], (1 + 1e-6) * np.eye(n_channels))
    npt.assert_allclose(result["means"][1], 0)
    assert np.linalg.matrix_rank(covs[1] - 1e-6 * np.eye(n_channels)) < n_channels
    for state in [0, 2]:
        npt.assert_allclose(result["covs"][state], covs[state], rtol=1e-10)
        npt.assert_allclose(result["means"][state], means[state], rtol=1e-10)
