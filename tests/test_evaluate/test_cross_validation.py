import numpy as np
import numpy.testing as npt

def test_BICVkmeans():
    from osl_dynamics.evaluate.cross_validation import BICVkmeans

    cv = BICVkmeans(n_clusters=2,n_samples=8,n_channels=2,partition_rows=4,partition_columns=2)

    # Step 1: Check partition_indices
    cv.partition_indices()

    # Step 2: Check the fold specification
    cv.row_indices = [np.array([7,4]),
                      np.array([5,0]),
                      np.array([6,3]),
                      np.array([2,1])]
    cv.column_indices = [np.array([1]),
                         np.array([0])]
    npt.assert_equal(cv.fold_indices(0,0),
                    (
                        np.array([0,1,2,3,5,6]),np.array([4,7]),
                        np.array([0]),np.array([1])
                    )
                    )
    npt.assert_equal(cv.fold_indices(2, 1),
                     (
                         np.array([0,1,2,4,5,7]), np.array([3,6]),
                         np.array([1]), np.array([0])
                     )
                     )

    # Fix the train/test rows and X/Y columns now.
    row_train = np.array([3,2,1,0])
    row_test = np.array([7,6,5,4])
    column_X = np.array([0])
    column_Y = np.array([1])

    # Step 3: Check the Y_train step
    data = np.array([[0.1,1.0],
                       [100.,1.0],
                       [-1000.,-1.0],
                       [0.0,-1.0],
                       [0.1, 0.5],
                       [100., 0.75],
                       [-1000., -0.25],
                       [0.0, -1.0]]
    )
    spatial_Y_train, temporal_Y_train = cv.Y_train(data,row_train,column_Y)
    npt.assert_equal(spatial_Y_train**2,np.array([[1.],[1.]]))
    npt.assert_equal(spatial_Y_train[temporal_Y_train], data[row_train][:,column_Y])

    # Step 4: Check the X_train step
    temporal_Y_train = np.array([1,1,0,0])
    spatial_X_train = cv.X_train(data, row_train, column_X, temporal_Y_train)
    npt.assert_equal(spatial_X_train,np.array([[50.05],[-500.]]))

    # Step 5: Check the X_test step
    spatial_X_train = np.array([[51.],[-49.]])
    temporal_X_test = cv.X_test(data,row_test,column_X,spatial_X_train)
    npt.assert_equal(temporal_X_test,np.array([1,1,0,1]))

    # Step 6: Cehck the Y_test step
    temporal_X_test = np.array([1,0,1,0])
    spatial_Y_train = np.array([[0.5],[-0.5]])
    metric = cv.Y_test(data,row_test,column_Y,temporal_X_test,spatial_Y_train)
    npt.assert_equal(metric,np.array([0.5**2+1.25**2+0.75**2])/4)


def test_BICVHMM():
    import os
    import yaml
    from osl_dynamics.evaluate.cross_validation import BICVHMM

    save_dir = './test_tmp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    row_train = [1,2]
    column_X = [1]
    column_Y = [0,2]

    # Construct the data
    def generate_obs(cov,mean=None,n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean,cov,n_timepoints)
    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5,0.0,0.5]
    covs_Y = [np.array([[1.0,cor],[cor,1.0]]) for cor in cors_Y]
    #observations_Y = [generate_obs(cov) for cov in covs_Y]

    means_X = [1.0,2.0,3.0]
    vars_X = [0.5,1.0,2.0]
    #observations_X = [generate_obs([[var]], [mean]) for var, mean in zip(vars_X, means_X)]

    #observations = [np.hstack((Y[:,:1], X, Y[:, 1:])) for X,Y in zip(observations_X,observations_Y)]

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in range(0,2):
        obs = []
        for j in range(1500):
            observations_Y = [generate_obs(covs_Y[i]),generate_obs(covs_Y[i+1])]
            observations_X = [generate_obs([[vars_X[i]]], [means_X[i]]),generate_obs([[vars_X[i+1]]], [means_X[i+1]])]
            observations = np.concatenate([np.hstack((Y[:, :1], X, Y[:, 1:])) for X, Y in zip(observations_X, observations_Y)],axis=0)
            obs.append(observations)

        obs = np.concatenate(obs,axis=0)
        np.save(f"{data_dir}{10002+i}.npy", obs)

    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy",generate_obs(np.eye(3)*100,n_timepoints=300000))

    initial_covariances = np.stack([np.array([[1.0,rho],[rho,1.0]]) for rho in cors_Y])
    np.save(f'{save_dir}init_covs.npy',initial_covariances)

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        timepoints:
                            - 0
                            - 300000
            n_states: 3
            learn_means: False
            learn_covariances: False
            learn_trans_prob: True
            learning_rate: 0.01
            initial_covariances: {save_dir}init_covs.npy
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
    cv = BICVHMM(n_samples,n_channels)
    cv.Y_train(config,train_keys,row_train,column_Y)
    result_X_train = cv.X_train(config,row_train,column_X,f'{save_dir}/Y_train/inf_params/alp.pkl')

    means = np.load(result_X_train['means'])
    covs = np.load(result_X_train['covs'])
    npt.assert_almost_equal(np.squeeze(means),np.array(means_X),decimal=2)
    npt.assert_almost_equal(np.squeeze(covs), np.array(vars_X), decimal=3)