import os
import numpy as np
import numpy.testing as npt


def test_swc():
    from osl_dynamics.models import swc
    from osl_dynamics.data.base import Data

    config_kwargs = {
        'n_states': 3,
        'n_channels': 2,
        'window_length': 5,
        'window_offset': 3,
        'learn_means': False,
        'learn_covariances': True
    }
    data_dir = ('./test_swc_data/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)
    dataset = Data(data_dir)
    config = swc.Config(**config_kwargs)
    swc = swc.Model(config)
    covs, labels = swc.fit(dataset)

    covs_answer = np.array([[[2.5, -2.5], [-2.5, 2.5]],
                            [[52., 48.], [48., 52.]],
                            [[54.5, 47.75], [47.75, 54.5]]])
    labels_answer = [np.array([0, 1]), np.array([1, 2])]

    # Reorder and covs and update the labels accordingly
    order = np.argsort(covs[:, 0, 0])
    reordered_covs = covs[order]
    label_mapping = {old: new for new, old in enumerate(order)}
    reordered_labels = [np.array([label_mapping[label] for label in session_labels]) for session_labels in labels]

    npt.assert_array_equal(covs_answer, reordered_covs)
    npt.assert_array_equal(labels_answer, reordered_labels)


def test_swc_spatial():
    from osl_dynamics.models import swc
    from osl_dynamics.data.base import Data

    config_kwargs = {
        'n_states': 3,
        'n_channels': 2,
        'window_length': 5,
        'window_offset': 3,
        'learn_means': False,
        'learn_covariances': True
    }
    data_dir = ('./test_swc_data/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)
    dataset = Data(data_dir)

    alpha = [np.array([2, 1]), np.array([1, 0])]
    config = swc.Config(**config_kwargs)
    swc = swc.Model(config)
    covs = swc.infer_spatial(dataset, alpha)
    covs_answer = np.array([[[54.5, 47.75], [47.75, 54.5]],
                            [[52., 48.], [48., 52.]],
                            [[2.5, -2.5], [-2.5, 2.5]]])

    npt.assert_array_equal(covs_answer, covs)


def test_swc_temporal():
    from osl_dynamics.models import swc
    from osl_dynamics.data.base import Data

    config_kwargs = {
        'n_states': 3,
        'n_channels': 2,
        'window_length': 5,
        'window_offset': 3,
        'learn_means': False,
        'learn_covariances': True
    }
    data_dir = ('./test_swc_data/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)
    dataset = Data(data_dir)

    means = np.zeros((3, 2))
    covs = np.array([[[54.5, 47.75], [47.75, 54.5]],
                     [[52., 48.], [48., 52.]],
                     [[2.5, -2.4999995], [-2.4999995, 2.5]]])

    config = swc.Config(**config_kwargs)
    swc = swc.Model(config)
    alpha = swc.infer_temporal(dataset, means, covs)

    alpha_answer = [np.array([2, 1]), np.array([1, 0])]

    npt.assert_array_equal(alpha_answer, alpha)


def test_swc_log_likelihood():
    from osl_dynamics.models import swc

    from osl_dynamics.data.base import Data

    config_kwargs = {
        'n_states': 3,
        'n_channels': 2,
        'window_length': 5,
        'window_offset': 3,
        'learn_means': False,
        'learn_covariances': True
    }
    data_dir = ('./test_swc_data/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)
    dataset = Data(data_dir)

    means = np.zeros((3, 2))
    covs = np.array([[[54.5, 47.75], [47.75, 54.5]],
                     [[52., 48.], [48., 52.]],
                     [[2.5, -2.4999995], [-2.4999995, 2.5]]])

    alpha = [np.array([2, 1]), np.array([1, 0])]

    config = swc.Config(**config_kwargs)
    swc = swc.Model(config)
    average_log_likelihood = swc.log_likelihood(dataset, alpha, means, covs)

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

    # Compute the log-likelihood for each data point and average
    log_likelihood_1 = log_likelihood_calculator(data_1[:5], means[2], covs[2])
    log_likelihood_2 = log_likelihood_calculator(data_1[3:], means[1], covs[1])
    log_likelihood_3 = log_likelihood_calculator(data_2[3:], means[0], covs[0])
    log_likelihood_answer = (log_likelihood_1 + 2 * log_likelihood_2 + log_likelihood_3) / 4
    npt.assert_almost_equal(average_log_likelihood, log_likelihood_answer)
