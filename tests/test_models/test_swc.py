"""Sliding Window Correlation model, ported from the research code's tests.

The tests come from tests/test_model/test_swc.py and the SWC tests in
tests/test_config_api/test_wrappers.py of the WIP branch, with the same data
and expected values. The API changed:

- :code:`fit` returns the state time course (the covariances are
  :code:`get_covariances()`).
- :code:`infer_spatial` is :code:`dual_estimation`.
- :code:`infer_temporal` is :code:`get_alpha` with the covariances set.
- :code:`log_likelihood` uses the means and covariances of the model.
"""

import numpy as np
import numpy.testing as npt
import pytest

import reference_wip as ref
from osl_dynamics.data import Data
from osl_dynamics.models import load, swc

DATA_1 = np.array(
    [
        [1.0, -1.0],
        [0.0, 0.0],
        [-1.0, 1.0],
        [2.0, -2.0],
        [-2.0, 2.0],
        [0.0, 0.0],
        [10.0, 10.0],
        [-10.0, -10.0],
    ]
)
DATA_2 = np.array(
    [
        [2.0, -2.0],
        [-2.0, 2.0],
        [0.0, 0.0],
        [10.0, 10.0],
        [-10.0, -10.0],
        [0.0, 3.0],
        [-3.0, 0.0],
        [3.0, -3.0],
    ]
)
CONFIG_KWARGS = {
    "n_states": 3,
    "n_channels": 2,
    "window_length": 5,
    "window_offset": 3,
    "learn_means": False,
    "learn_covariances": True,
}
COVS = np.array(
    [
        [[54.5, 47.75], [47.75, 54.5]],
        [[52.0, 48.0], [48.0, 52.0]],
        [[2.5, -2.4999995], [-2.4999995, 2.5]],
    ]
)


@pytest.fixture
def dataset(tmp_path):
    np.save(tmp_path / "10001.npy", DATA_1)
    np.save(tmp_path / "10002.npy", DATA_2)
    data = Data(str(tmp_path))
    yield data
    data.delete_dir()


def reorder(covs, labels):
    """Order the states by the first variance."""
    order = np.argsort(covs[:, 0, 0])
    label_mapping = {old: new for new, old in enumerate(order)}
    labels = [np.array([label_mapping[label] for label in s]) for s in labels]
    return covs[order], labels


def test_swc(dataset):
    """Also covers test_train_swc in tests/test_config_api/test_wrappers.py."""
    model = swc.Model(swc.Config(**CONFIG_KWARGS, random_state=0))
    labels = model.fit(dataset)
    covs = model.get_covariances()

    covs_answer = np.array(
        [
            [[2.5, -2.5], [-2.5, 2.5]],
            [[52.0, 48.0], [48.0, 52.0]],
            [[54.5, 47.75], [47.75, 54.5]],
        ]
    )
    labels_answer = [np.array([0, 1]), np.array([1, 2])]

    reordered_covs, reordered_labels = reorder(covs, labels)
    npt.assert_array_equal(model.get_means(), np.zeros((3, 2)))
    npt.assert_array_equal(covs_answer, reordered_covs)
    npt.assert_array_equal(labels_answer, reordered_labels)


def test_swc_spatial(dataset):
    """Also covers test_swc_infer_spatial in test_wrappers.py."""
    alpha = [np.array([2, 1]), np.array([1, 0])]
    model = swc.Model(swc.Config(**CONFIG_KWARGS))
    means, covs = model.dual_estimation(dataset, alpha)

    covs_answer = np.array(
        [
            [[54.5, 47.75], [47.75, 54.5]],
            [[52.0, 48.0], [48.0, 52.0]],
            [[2.5, -2.5], [-2.5, 2.5]],
        ]
    )
    npt.assert_array_equal(means, np.zeros((3, 2)))
    npt.assert_array_equal(covs_answer, covs)


def test_swc_temporal(dataset):
    """Also covers test_swc_infer_temporal in test_wrappers.py."""
    config = swc.Config(
        **{**CONFIG_KWARGS, "learn_covariances": False},
        initial_means=np.zeros((3, 2)),
        initial_covariances=COVS,
    )
    model = swc.Model(config)
    alpha_answer = [np.array([2, 1]), np.array([1, 0])]
    npt.assert_array_equal(alpha_answer, model.get_alpha(dataset))

    # fit only assigns the windows when the covariances are fixed
    npt.assert_array_equal(alpha_answer, model.fit(dataset))
    npt.assert_array_equal(model.get_covariances(), COVS)


def test_swc_log_likelihood(dataset):
    """Also covers test_swc_log_likelihood in test_wrappers.py.

    The research code's model test called log_likelihood with the means
    and covariances. The wrapper test expected the value before commit
    3829b6c0, (...) / 20 * 8, i.e. multiplied by the number of time points
    per session. Now the log-likelihood is averaged over windows.
    """
    means = np.zeros((3, 2))
    alpha = [np.array([2, 1]), np.array([1, 0])]
    model = swc.Model(swc.Config(**CONFIG_KWARGS))
    model.set_means_covariances(means, COVS)
    average_log_likelihood = model.log_likelihood(dataset, alpha)

    def log_likelihood_calculator(x, mean, cov):
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

    log_likelihood_1 = log_likelihood_calculator(DATA_1[:5], means[2], COVS[2])
    log_likelihood_2 = log_likelihood_calculator(DATA_1[3:], means[1], COVS[1])
    log_likelihood_3 = log_likelihood_calculator(DATA_2[3:], means[0], COVS[0])
    log_likelihood_answer = (
        log_likelihood_1 + 2 * log_likelihood_2 + log_likelihood_3
    ) / 4
    npt.assert_almost_equal(average_log_likelihood, log_likelihood_answer)

    expected = ref.swc_average_log_likelihood(
        [DATA_1, DATA_2], alpha, means, COVS, 5, 3
    )
    npt.assert_allclose(average_log_likelihood, expected, rtol=1e-10)


def test_swc_save_load(tmp_path, dataset):
    model = swc.Model(swc.Config(**CONFIG_KWARGS, random_state=0))
    model.fit(dataset)
    model.save(str(tmp_path / "model"))

    loaded = load(str(tmp_path / "model"))
    assert isinstance(loaded, swc.Model)
    assert loaded.config.window_length == 5
    npt.assert_array_equal(loaded.get_covariances(), model.get_covariances())
    npt.assert_array_equal(loaded.get_alpha(dataset), model.get_alpha(dataset))


def test_swc_config_validation():
    with pytest.raises(ValueError, match="learn_means"):
        swc.Config(**{**CONFIG_KWARGS, "learn_means": True})
    with pytest.raises(ValueError, match="initial_covariances"):
        swc.Config(**{**CONFIG_KWARGS, "learn_covariances": False})
    with pytest.raises(ValueError, match="window_length"):
        swc.Config(**{**CONFIG_KWARGS, "window_length": None})
