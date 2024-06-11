import numpy as np
import numpy.testing as npt


def test_BICVkmeans():
    from osl_dynamics.evaluate.cross_validation import BICVkmeans

    cv = BICVkmeans(n_clusters=2, n_samples=8, n_channels=2, partition_rows=4, partition_columns=2)

    # Step 1: Check partition_indices
    cv.partition_indices()

    # Step 2: Check the fold specification
    cv.row_indices = [np.array([7, 4]),
                      np.array([5, 0]),
                      np.array([6, 3]),
                      np.array([2, 1])]
    cv.column_indices = [np.array([1]),
                         np.array([0])]
    npt.assert_equal(cv.fold_indices(0, 0),
                     (
                         np.array([0, 1, 2, 3, 5, 6]), np.array([4, 7]),
                         np.array([0]), np.array([1])
                     )
                     )
    npt.assert_equal(cv.fold_indices(2, 1),
                     (
                         np.array([0, 1, 2, 4, 5, 7]), np.array([3, 6]),
                         np.array([1]), np.array([0])
                     )
                     )

    # Fix the train/test rows and X/Y columns now.
    row_train = np.array([3, 2, 1, 0])
    row_test = np.array([7, 6, 5, 4])
    column_X = np.array([0])
    column_Y = np.array([1])

    # Step 3: Check the Y_train step
    data = np.array([[0.1, 1.0],
                     [100., 1.0],
                     [-1000., -1.0],
                     [0.0, -1.0],
                     [0.1, 0.5],
                     [100., 0.75],
                     [-1000., -0.25],
                     [0.0, -1.0]]
                    )
    spatial_Y_train, temporal_Y_train = cv.Y_train(data, row_train, column_Y)
    npt.assert_equal(spatial_Y_train ** 2, np.array([[1.], [1.]]))
    npt.assert_equal(spatial_Y_train[temporal_Y_train], data[row_train][:, column_Y])

    # Step 4: Check the X_train step
    temporal_Y_train = np.array([1, 1, 0, 0])
    spatial_X_train = cv.X_train(data, row_train, column_X, temporal_Y_train)
    npt.assert_equal(spatial_X_train, np.array([[50.05], [-500.]]))

    # Step 5: Check the X_test step
    spatial_X_train = np.array([[51.], [-49.]])
    temporal_X_test = cv.X_test(data, row_test, column_X, spatial_X_train)
    npt.assert_equal(temporal_X_test, np.array([1, 1, 0, 1]))

    # Step 6: Cehck the Y_test step
    temporal_X_test = np.array([1, 0, 1, 0])
    spatial_Y_train = np.array([[0.5], [-0.5]])
    metric = cv.Y_test(data, row_test, column_Y, temporal_X_test, spatial_Y_train)
    npt.assert_equal(metric, np.array([0.5 ** 2 + 1.25 ** 2 + 0.75 ** 2]) / 4)


def test_partition_indices():
    import os
    import shutil
    from osl_dynamics.evaluate.cross_validation import CVBase

    save_dir = './test_tmp_partition_indices/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(f'{save_dir}case_1/'):
        os.makedirs(f'{save_dir}case_1/')

    if not os.path.exists(f'{save_dir}case_2/'):
        os.makedirs(f'{save_dir}case_2/')

    n_samples = 1000
    n_channels = 50

    # Case 1: Default settings
    cv_1 = CVBase(n_samples=n_samples, n_channels=n_channels, save_dir=f'{save_dir}case_1/')
    row_indices = np.load(os.path.join(f'{save_dir}case_1/', 'row_indices.npz'))
    column_indices = np.load(os.path.join(f'{save_dir}case_1/', 'column_indices.npz'))
    row_indices = np.sort(np.concatenate([row_indices[key] for key in row_indices.keys()]))
    column_indices = np.sort(np.concatenate([column_indices[key] for key in column_indices.keys()]))
    npt.assert_array_equal(row_indices, np.arange(n_samples))
    npt.assert_array_equal(column_indices, np.arange(n_channels))

    # Case 2: Multi-folds
    cv_2 = CVBase(n_samples=n_samples, n_channels=n_channels, save_dir=f'{save_dir}case_2/',
                  partition_rows=7, partition_columns=9)
    row_indices = np.load(os.path.join(f'{save_dir}case_2/', 'row_indices.npz'))
    column_indices = np.load(os.path.join(f'{save_dir}case_2/', 'column_indices.npz'))
    row_indices = np.sort(np.concatenate([row_indices[key] for key in row_indices.keys()]))
    column_indices = np.sort(np.concatenate([column_indices[key] for key in column_indices.keys()]))
    npt.assert_array_equal(row_indices, np.arange(n_samples))
    npt.assert_array_equal(column_indices, np.arange(n_channels))

    # Case 3: Use the results from case 1:
    cv_3 = CVBase(n_samples=n_samples,
                  n_channels=n_channels,
                  row_indices=f'{save_dir}case_1/row_indices.npz',
                  column_indices=f'{save_dir}case_1/column_indices.npz')

    npt.assert_array_equal(cv_1.row_indices[0], cv_3.row_indices[0])
    npt.assert_array_equal(cv_1.row_indices[1], cv_3.row_indices[1])
    npt.assert_array_equal(cv_1.column_indices[0], cv_3.column_indices[0])
    npt.assert_array_equal(cv_1.column_indices[1], cv_3.column_indices[1])

    # Delete the directory afterwards
    shutil.rmtree(save_dir)


def test_fold_indices():
    from osl_dynamics.evaluate.cross_validation import CVBase

    n_samples = 7
    n_channels = 5
    cv = CVBase(n_samples=n_samples, n_channels=n_channels, partition_rows=3)
    cv.row_indices = [np.array([6, 2]), np.array([4, 0]), np.array([5, 3, 1])]
    cv.column_indices = [np.array([4, 2, 0]), np.array([3, 1])]

    # Fold (1,1)
    row_train, row_test, column_X, column_Y = cv.fold_indices(0, 0)
    npt.assert_array_equal(row_train, np.array([0, 1, 3, 4, 5]))
    npt.assert_array_equal(row_test, np.array([2, 6]))
    npt.assert_array_equal(column_X, np.array([1, 3]))
    npt.assert_array_equal(column_Y, np.array([0, 2, 4]))

    # Fold (1,2)
    row_train, row_test, column_X, column_Y = cv.fold_indices(0, 1)
    npt.assert_array_equal(row_train, np.array([0, 1, 3, 4, 5]))
    npt.assert_array_equal(row_test, np.array([2, 6]))
    npt.assert_array_equal(column_X, np.array([0, 2, 4]))
    npt.assert_array_equal(column_Y, np.array([1, 3]))

    # Fold (2,1)
    row_train, row_test, column_X, column_Y = cv.fold_indices(1, 0)
    npt.assert_array_equal(row_train, np.array([1, 2, 3, 5, 6]))
    npt.assert_array_equal(row_test, np.array([0, 4]))
    npt.assert_array_equal(column_X, np.array([1, 3]))
    npt.assert_array_equal(column_Y, np.array([0, 2, 4]))

    # Fold (2,2)
    row_train, row_test, column_X, column_Y = cv.fold_indices(1, 1)
    npt.assert_array_equal(row_train, np.array([1, 2, 3, 5, 6]))
    npt.assert_array_equal(row_test, np.array([0, 4]))
    npt.assert_array_equal(column_X, np.array([0, 2, 4]))
    npt.assert_array_equal(column_Y, np.array([1, 3]))

    # Fold (3,1)
    row_train, row_test, column_X, column_Y = cv.fold_indices(2, 0)
    npt.assert_array_equal(row_train, np.array([0, 2, 4, 6]))
    npt.assert_array_equal(row_test, np.array([1, 3, 5]))
    npt.assert_array_equal(column_X, np.array([1, 3]))
    npt.assert_array_equal(column_Y, np.array([0, 2, 4]))

    # Fold (3,2)
    row_train, row_test, column_X, column_Y = cv.fold_indices(2, 1)
    npt.assert_array_equal(row_train, np.array([0, 2, 4, 6]))
    npt.assert_array_equal(row_test, np.array([1, 3, 5]))
    npt.assert_array_equal(column_X, np.array([0, 2, 4]))
    npt.assert_array_equal(column_Y, np.array([1, 3]))


def test_full_train():
    import os
    import shutil
    import yaml
    from osl_dynamics.evaluate.cross_validation import CVHMM

    save_dir = './test_full_train/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    row_train = [1, 2]
    column_X = [1]
    column_Y = [0, 2]

    # Construct the data
    def generate_obs(cov, mean=None, n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean, cov, n_timepoints)

    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]

    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in range(0, 2):
        obs = []
        for j in range(1500):
            observations_Y = [generate_obs(covs_Y[i]), generate_obs(covs_Y[i + 1])]
            observations_X = [generate_obs([[vars_X[i]]], [means_X[i]]),
                              generate_obs([[vars_X[i + 1]]], [means_X[i + 1]])]
            observations = np.concatenate(
                [np.hstack((Y[:, :1], X, Y[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        timepoints:
                            - 0
                            - 300000
            n_states: {n_states}
            learn_means: False
            learn_covariances: True
            learning_rate: 0.01
            n_epochs: 3
            sequence_length: 600
            init_kwargs:
                n_init: 1
                n_epochs: 1
            save_dir: {save_dir}
            model: hmm

            """
    config = yaml.safe_load(config)

    train_keys = ['n_channels',
                  'n_states',
                  'learn_means',
                  'learn_covariances',
                  'learn_trans_prob',
                  'initial_means',
                  'initial_covariances',
                  'initial_trans_prob',
                  'sequence_length',
                  'batch_size',
                  'learning_rate',
                  'n_epochs',
                  ]
    cv = CVHMM(n_samples, n_channels, train_keys=train_keys)
    result, _ = cv.full_train(config, row_train, column_Y)

    result_means = np.load(result['means'])
    result_covs = np.load(result['covs'])
    npt.assert_array_equal(result_means, np.zeros((n_states, len(column_Y))))

    # Assert diagonal elements are all one
    npt.assert_allclose(np.diagonal(result_covs, axis1=-2, axis2=-1), 1.0, rtol=0.05, atol=0.05)

    # Assert off-diagonal elements are equal to cors
    off_diagonal = np.array([float(result_covs[i, 0, 1]) for i in range(n_states)])
    npt.assert_allclose(np.sort(off_diagonal), cors_Y, atol=0.05, rtol=0.05)


def test_infer_spatial():
    import os
    import pickle
    import shutil
    import yaml
    from osl_dynamics.evaluate.cross_validation import CVHMM

    save_dir = './test_infer_spatial/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    row_train = [1, 2]
    column_X = [1]
    column_Y = [0, 2]

    # Construct the data
    def generate_obs(cov, mean=None, n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean, cov, n_timepoints)

    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]

    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]

    n_timepoints = 100

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    hidden_states = []

    for i in range(0, 2):
        # Build up the hidden variable
        hv_temp = np.zeros((n_timepoints * 2, n_states))
        hv_temp[:, i] = np.array([0.6] * n_timepoints + [0.4] * n_timepoints)
        hv_temp[:, i + 1] = np.array([0.4] * n_timepoints + [0.6] * n_timepoints)
        hidden_states.append(np.tile(hv_temp, (1500, 1)))

        obs = []
        for j in range(1500):
            observations_Y = [generate_obs(covs_Y[i], n_timepoints=n_timepoints),
                              generate_obs(covs_Y[i + 1], n_timepoints=n_timepoints)]
            observations_X = [generate_obs([[vars_X[i]]], [means_X[i]], n_timepoints=n_timepoints),
                              generate_obs([[vars_X[i + 1]]], [means_X[i + 1]], n_timepoints=n_timepoints)]
            observations = np.concatenate(
                [np.hstack((Y[:, :1], X, Y[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    with open(f'{data_dir}alp.pkl', "wb") as file:
        pickle.dump(hidden_states, file)
    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        timepoints:
                            - 0
                            - 300000
            n_states: {n_states}
            learn_means: True
            learn_covariances: True
            learning_rate: 0.01
            n_epochs: 3
            sequence_length: 600
            save_dir: {save_dir}
            model: hmm
            """

    train_keys = ['n_channels',
                  'n_states',
                  'learn_means',
                  'learn_covariances',
                  'learn_trans_prob',
                  'initial_means',
                  'initial_covariances',
                  'initial_trans_prob',
                  'sequence_length',
                  'batch_size',
                  'learning_rate',
                  'n_epochs',
                  ]

    config = yaml.safe_load(config)
    cv = CVHMM(n_samples, n_channels, train_keys=train_keys)
    result = cv.infer_spatial(config, row_train, column_X, f'{data_dir}alp.pkl')

    result_means = np.load(result['means'])
    result_covs = np.load(result['covs'])
    npt.assert_allclose(means_X, result_means, rtol=1e-2, atol=1e-2)
    npt.assert_allclose(vars_X, result_covs, rtol=1e-2, atol=1e-2)


def test_infer_temporal():
    import os
    import shutil
    import yaml
    import pickle
    from osl_dynamics.evaluate.cross_validation import CVHMM

    save_dir = './test_infer_temporal/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    row_test = [1, 2]
    column_X = [0, 2]
    column_Y = [1]

    # Construct the data
    def generate_obs(cov, mean=None, n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean, cov, n_timepoints)

    # Define the covariance matrices of state 1,2 in both splits
    means_X = [np.array([-10.0, -10.0]), np.array([0.0, 0.0]), np.array([10.0, 10.0])]
    cors_X = [-0.5, 0.0, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]

    means_Y = [1.0, 2.0, 3.0]
    vars_Y = [0.5, 1.0, 2.0]

    np.save(f'{save_dir}/fixed_means.npy', np.array(means_X))
    np.save(f'{save_dir}/fixed_covs.npy', np.stack(covs_X))
    spatial_X_train = {'means': f'{save_dir}/fixed_means.npy', 'covs': f'{save_dir}/fixed_covs.npy'}

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    hidden_states = []
    n_timepoints = 100

    for i in range(0, 2):

        # Build up the hidden variable
        hv_temp = np.zeros((n_timepoints * 2, n_states))
        hv_temp[:, i] = np.array([1.0] * n_timepoints + [0.0] * n_timepoints)
        hv_temp[:, i + 1] = np.array([0.0] * n_timepoints + [1.0] * n_timepoints)
        hidden_states.append(np.tile(hv_temp, (1500, 1)))

        obs = []
        for j in range(1500):
            observations_X = [generate_obs(covs_X[i], means_X[i], n_timepoints),
                              generate_obs(covs_X[i + 1], means_X[i + 1], n_timepoints)]
            observations_Y = [generate_obs([[vars_Y[i]]], [means_Y[i]], n_timepoints),
                              generate_obs([[vars_Y[i + 1]]], [means_Y[i + 1]]), n_timepoints]
            observations = np.concatenate(
                [np.hstack((X[:, :1], Y, X[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        timepoints:
                            - 0
                            - 300000
            n_states: {n_states}
            learn_means: False
            learn_covariances: True
            learning_rate: 0.01
            n_epochs: 10
            sequence_length: 600
            init_kwargs:
                n_init: 1
                n_epochs: 1
            save_dir: {save_dir}
            model: hmm
            """
    config = yaml.safe_load(config)

    train_keys = ['n_channels',
                  'n_states',
                  'learn_means',
                  'learn_covariances',
                  'learn_trans_prob',
                  'initial_means',
                  'initial_covariances',
                  'initial_trans_prob',
                  'sequence_length',
                  'batch_size',
                  'learning_rate',
                  'n_epochs',
                  ]
    cv = CVHMM(n_samples, n_channels, train_keys=train_keys)
    result = cv.infer_temporal(config, row_test, column_X, spatial_X_train)

    # Read the alpha
    with open(result, 'rb') as file:
        alpha = pickle.load(file)

    for i in range(2):
        npt.assert_allclose(alpha[0], hidden_states[0], atol=1e-6)


def test_calculate_error():
    import os
    import json
    import shutil
    import yaml
    import pickle
    from osl_dynamics.evaluate.cross_validation import CVHMM

    save_dir = './test_calculate_error/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    row_test = [1, 2]
    column_Y = [0, 2]

    # Generate the data
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Build up subject data
    data_1 = np.zeros((2, 3))
    np.save(f'{data_dir}10001.npy', data_1)
    data_2 = np.array([[1., 0., 1., ], [-1., 0., 0.]])
    np.save(f'{data_dir}10002.npy', data_2)
    data_3 = np.array([[-1., 0., -1.], [1., 0., 0.]])
    np.save(f'{data_dir}10003.npy', data_3)

    def multivariate_gaussian_log_likelihood(x, mu, cov):
        """
        Calculate the log-likelihood for a multivariate Gaussian distribution.

        Parameters:
            x (ndarray): Observations (N, d), where N is the number of samples and d is the dimensionality.
            mu (ndarray): Mean vector of the distribution (d,).
            cov (ndarray): Covariance matrix of the distribution (d, d).

        Returns:
            float: Log-likelihood value.
        """
        # Dimensionality of the data
        d = len(mu)

        # Calculate the log determinant of the covariance matrix
        log_det_cov = np.log(np.linalg.det(cov))

        # Calculate the quadratic term in the exponent
        quad_term = np.sum((x - mu) @ np.linalg.inv(cov) * (x - mu), axis=1)

        # Calculate the log-likelihood
        log_likelihood = -0.5 * (d * np.log(2 * np.pi) + log_det_cov + quad_term)

        return log_likelihood

    config = f"""
                load_data:
                    inputs: {data_dir}
                    prepare:
                        select:
                            timepoints:
                                - 0
                                - 2
                n_states: 3
                learn_means: False
                learn_covariances: True
                learning_rate: 0.01
                n_epochs: 10
                sequence_length: 600
                init_kwargs:
                    n_init: 1
                    n_epochs: 1
                save_dir: {save_dir}
                model: hmm
                """
    config = yaml.safe_load(config)

    means = np.zeros((3, 2))
    covs = np.array([[[1.0, 0.0], [0.0, 1.0]],
                     [[1.5, 0.8], [0.8, 1.5]],
                     [[0.5, -0.25], [-0.25, 0.5]]])
    np.save(f'{save_dir}/means.npy', means)
    np.save(f'{save_dir}/covs.npy', covs)
    spatial = {
        'means': f'{save_dir}/means.npy',
        'covs': f'{save_dir}/covs.npy'
    }

    # Set up the alpha.pkl
    alpha = [np.array([[1., 0., 0.], [0.0, 0.5, 0.5]]),
             np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])]
    with open(f'{data_dir}alp.pkl', "wb") as file:
        pickle.dump(alpha, file)
    # Set up the cross validation
    train_keys = ['n_channels',
                  'n_states',
                  'learn_means',
                  'learn_covariances',
                  'learn_trans_prob',
                  'initial_means',
                  'initial_covariances',
                  'initial_trans_prob',
                  'sequence_length',
                  'batch_size',
                  'learning_rate',
                  'n_epochs',
                  ]
    cv = CVHMM(n_samples, n_channels, train_keys=train_keys)
    result = cv.calculate_error(config, row_test, column_Y, f'{data_dir}alp.pkl', spatial)

    ll_1 = multivariate_gaussian_log_likelihood(data_2[:1, [0, 2]], np.array([0, 0]), covs[0])
    ll_2 = 0.5 * multivariate_gaussian_log_likelihood(data_2[1:2, [0, 2]], np.array([0, 0]), covs[1]) + \
           0.5 * multivariate_gaussian_log_likelihood(data_2[1:2, [0, 2]], np.array([0, 0]), covs[2])
    ll_3 = 0.5 * multivariate_gaussian_log_likelihood(data_3[:1, [0, 2]], np.array([0, 0]), covs[0]) + \
           0.5 * multivariate_gaussian_log_likelihood(data_3[:1, [0, 2]], np.array([0, 0]), covs[1])
    ll_4 = multivariate_gaussian_log_likelihood(data_3[1:2, [0, 2]], np.array([0, 0]), covs[2])

    ll = (ll_1 + ll_2 + ll_3 + ll_4) / 2
    with open(result, 'r') as file:
        # Load the JSON data
        metrics = json.load(file)

    npt.assert_almost_equal(ll, metrics['log_likelihood'], decimal=3)


def test_split_column():
    import os
    import shutil
    from osl_dynamics.evaluate.cross_validation import CVHMM
    save_dir = './test_split_column/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)
    n_samples = 2
    n_channels = 4
    cv = CVHMM(n_samples, n_channels)
    config = {'save_dir': '123'}

    means = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    covs = np.array([
        [[1., 2., 3., 4.],
         [5., 6., 7., 8.],
         [9., 10., 11., 12.],
         [13., 14., 15., 16.]],
        [[17., 18., 19., 20.],
         [21., 22., 23., 24.],
         [25., 26., 27., 28.],
         [29., 30., 31., 32.]]
    ])
    column_X = [0, 2]
    column_Y = [1, 3]
    np.save(f'{save_dir}means.npy', means)
    np.save(f'{save_dir}covs.npy', covs)
    spatial_XY_train = {'means': f'{save_dir}means.npy', 'covs': f'{save_dir}covs.npy'}
    spatial_X_train, spatial_Y_train = cv.split_column(config, column_X, column_Y, spatial_XY_train,
                                                       save_dir=[f'{save_dir}X_train/', f'{save_dir}Y_train/']
                                                       )
    spatial_X_train_means = np.load(spatial_X_train['means'])
    spatial_X_train_covs = np.load(spatial_X_train['covs'])
    spatial_Y_train_means = np.load(spatial_Y_train['means'])
    spatial_Y_train_covs = np.load(spatial_Y_train['covs'])

    X_train_means = np.array([[1., 3.], [5., 7.]])
    X_train_covs = np.array([[[1., 3.], [9., 11.]], [[17., 19.], [25., 27.]]])
    Y_train_means = np.array([[2., 4.], [6., 8.]])
    Y_train_covs = np.array([[[6., 8.], [14., 16.]], [[22., 24.], [30., 32.]]])
    npt.assert_array_equal(spatial_X_train_means, X_train_means)
    npt.assert_array_equal(spatial_X_train_covs, X_train_covs)
    npt.assert_array_equal(spatial_Y_train_means, Y_train_means)
    npt.assert_array_equal(spatial_Y_train_covs, Y_train_covs)


def test_split_row():
    import os
    import pickle
    import shutil
    from osl_dynamics.evaluate.cross_validation import CVHMM

    save_dir = './test_split_row/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    n_samples = 4
    n_channels = 2
    cv = CVHMM(n_samples, n_channels)
    config = {'save_dir': '123'}
    alpha = [
        np.array([0., 0., 0.]),
        np.array([1., 1., 1.]),
        np.array([2., 2., 2.]),
        np.array([3., 3., 3.])
    ]
    with open(f'{save_dir}alp.pkl', 'wb') as file:
        pickle.dump(alpha, file)
    row_train = [0, 2]
    row_test = [1, 3]
    temporal_X_train, temporal_X_test = cv.split_row(config, row_train, row_test, f'{save_dir}alp.pkl',
                                                     save_dir=[f'{save_dir}/X_train/', f'{save_dir}/X_test/'])
    answer_1 = [np.array([0., 0., 0.]), np.array([2., 2., 2.]), ]
    answer_2 = [np.array([1., 1., 1.]), np.array([3., 3., 3.])]

    with open(temporal_X_train, 'rb') as file:
        temp_1 = pickle.load(file)

    with open(temporal_X_test, 'rb') as file:
        temp_2 = pickle.load(file)

    npt.assert_array_equal(temp_1, answer_1)
    npt.assert_array_equal(temp_2, answer_2)


def test_swc_full_train():
    import os
    import shutil
    import yaml
    from osl_dynamics.evaluate.cross_validation import CVSWC

    save_dir = './test_swc_full_train/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    row_train = [1, 2]
    column_X = [1]
    column_Y = [0, 2]

    # Construct the data
    def generate_obs(cov, mean=None, n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean, cov, n_timepoints)

    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]

    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in range(0, 2):
        obs = []
        for j in range(1500):
            observations_Y = [generate_obs(covs_Y[i]), generate_obs(covs_Y[i + 1])]
            observations_X = [generate_obs([[vars_X[i]]], [means_X[i]]),
                              generate_obs([[vars_X[i + 1]]], [means_X[i + 1]])]
            observations = np.concatenate(
                [np.hstack((Y[:, :1], X, Y[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        timepoints:
                            - 0
                            - 300000
            n_states: {n_states}
            learn_means: False
            learn_covariances: True
            window_length: 100
            window_offset: 100
            save_dir: {save_dir}
            model: swc

            """
    config = yaml.safe_load(config)

    cv = CVSWC(n_samples, n_channels)
    result, _ = cv.full_train(config, row_train, column_Y)

    result_means = np.load(result['means'])
    result_covs = np.load(result['covs'])
    npt.assert_array_equal(result_means, np.zeros((n_states, len(column_Y))))

    # Assert diagonal elements are all one
    npt.assert_allclose(np.diagonal(result_covs, axis1=-2, axis2=-1), 1.0, rtol=0.05, atol=0.05)

    # Assert off-diagonal elements are equal to cors
    off_diagonal = np.array([float(result_covs[i, 0, 1]) for i in range(n_states)])
    npt.assert_allclose(np.sort(off_diagonal), cors_Y, atol=0.05, rtol=0.05)


def test_swc_infer_spatial():
    import os
    import pickle
    import shutil
    import yaml
    from osl_dynamics.evaluate.cross_validation import CVSWC

    save_dir = './test_swc_infer_spatial/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    row_train = [1, 2]
    column_X = [1]
    column_Y = [0, 2]

    # Construct the data
    def generate_obs(cov, mean=None, n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean, cov, n_timepoints)

    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]

    means_X = [0.0, 0.0, 0.0]
    vars_X = [0.5, 1.0, 2.0]

    n_timepoints = 100

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    hidden_states = []

    for i in range(0, 2):
        # Build up the hidden variable
        hidden_states.append(np.tile([i, i + 1], 1500))

        obs = []
        for j in range(1500):
            observations_Y = [generate_obs(covs_Y[i], n_timepoints=n_timepoints),
                              generate_obs(covs_Y[i + 1], n_timepoints=n_timepoints)]
            observations_X = [generate_obs([[vars_X[i]]], [means_X[i]], n_timepoints=n_timepoints),
                              generate_obs([[vars_X[i + 1]]], [means_X[i + 1]], n_timepoints=n_timepoints)]
            observations = np.concatenate(
                [np.hstack((Y[:, :1], X, Y[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    with open(f'{data_dir}alp.pkl', "wb") as file:
        pickle.dump(hidden_states, file)
    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        timepoints:
                            - 0
                            - 300000
            n_states: {n_states}
            learn_means: False
            learn_covariances: True
            window_length: 100
            window_offset: 100
            save_dir: {save_dir}
            model: swc
           """

    config = yaml.safe_load(config)
    cv = CVSWC(n_samples, n_channels)
    result = cv.infer_spatial(config, row_train, column_X, f'{data_dir}alp.pkl')

    result_means = np.load(result['means'])
    result_covs = np.load(result['covs'])
    npt.assert_allclose(means_X, np.squeeze(result_means), rtol=1e-2, atol=1e-2)
    npt.assert_allclose(vars_X, np.squeeze(result_covs), rtol=1e-2, atol=1e-2)

def test_swc_infer_temporal():
    import os
    import shutil
    import yaml
    import pickle
    from osl_dynamics.evaluate.cross_validation import CVSWC

    save_dir = './test_swc_infer_temporal/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    row_test = [1, 2]
    column_X = [0, 2]
    column_Y = [1]

    # Construct the data
    def generate_obs(cov, mean=None, n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean, cov, n_timepoints)

    # Define the covariance matrices of state 1,2 in both splits
    means_X = [np.array([-10.0, -10.0]), np.array([0.0, 0.0]), np.array([10.0, 10.0])]
    cors_X = [-0.5, 0.0, 0.5]
    covs_X = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_X]

    means_Y = [1.0, 2.0, 3.0]
    vars_Y = [0.5, 1.0, 2.0]

    np.save(f'{save_dir}/means.npy', np.array(means_X))
    np.save(f'{save_dir}/covs.npy', np.stack(covs_X))
    spatial_X_train = {'means': f'{save_dir}/means.npy', 'covs': f'{save_dir}/covs.npy'}

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    hidden_states = []
    n_timepoints = 100

    for i in range(0, 2):
        hidden_states.append(np.tile([i, i + 1], 1500))
        obs = []
        for j in range(1500):
            observations_X = [generate_obs(covs_X[i], means_X[i], n_timepoints),
                              generate_obs(covs_X[i + 1], means_X[i + 1], n_timepoints)]
            observations_Y = [generate_obs([[vars_Y[i]]], [means_Y[i]], n_timepoints),
                              generate_obs([[vars_Y[i + 1]]], [means_Y[i + 1]]), n_timepoints]
            observations = np.concatenate(
                [np.hstack((X[:, :1], Y, X[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        timepoints:
                            - 0
                            - 300000
            n_states: {n_states}
            learn_means: False
            learn_covariances: True
            window_length: 100
            window_offset: 100
            save_dir: {save_dir}
            model: swc
            """
    config = yaml.safe_load(config)

    cv = CVSWC(n_samples, n_channels)
    result = cv.infer_temporal(config, row_test, column_X, spatial_X_train)

    # Read the alpha
    with open(result, 'rb') as file:
        alpha = pickle.load(file)

    for i in range(2):
        npt.assert_allclose(alpha[0], hidden_states[0], atol=1e-6)
