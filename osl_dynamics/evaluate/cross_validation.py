import os
import pickle
import yaml
import shutil
import json
from collections import OrderedDict
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from ..config_api.pipeline import run_pipeline_from_file
from ..config_api.wrappers import load_data
from ..inference.modes import argmax_time_courses
from ..array_ops import npz2list


class BICVkmeans():
    def __init__(self, n_clusters, n_samples, n_channels, partition_rows=2, partition_columns=2):
        '''
        Initialisation of BICVkmeans
        Parameters
        ----------
        n_clusters: int
            how many clusters to use in the k-means algorithm
        n_samples: int
            the number of samples
        n_channels: int
            the number of channels (the length of each sample)
        partition_rows: int
            the number of rows partition
        partition_columns: int
            the number of columns partition
        '''
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.partition_rows = partition_rows
        self.partition_columns = partition_columns
        self.partition_indices()

    def partition_indices(self, save_dir=None):
        '''
        Generate partition indices, if save_dir is not None,
        save the indices.
        Parameters
        ----------
        save_dir: str,optional
            the directory to save the indices

        '''
        # Generate random row and column indices
        row_indices = np.arange(self.n_samples)
        column_indices = np.arange(self.n_channels)
        np.random.shuffle(row_indices)
        np.random.shuffle(column_indices)

        # Divide rows into partitions
        self.row_indices = np.array_split(row_indices, self.partition_rows)

        # Divide columns into partitions
        self.column_indices = np.array_split(column_indices, self.partition_columns)

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savez(os.path.join(save_dir, 'row_indices.npz'), *row_indices)
            np.savez(os.path.join(save_dir, 'row_indices.npz'), *column_indices)

    def fold_indices(self, r, s):
        """
        Given the paritions, return the indices of fold (r,s).
        Fold (r,s) treats the rth row subset as "test", and the sth column
        subset as response.
        The parition should be in the format:
        (X_train, Y_train,
         X_test,  Y_test  )
        Parameters
        ----------
        r: int
            the row index
        s: int
            the column index

        Returns
        -------
        row_train: list
            row indices for X_train and Y_train
        row_test: list
            row indices for X_test and Y_test
        column_X: list
            column indices for X_train and X_test
        column_Y: list
            column indices for Y_train and Y_test
        """

        # Assuming row_indices and column_indices are available
        row_train = []
        row_test = []
        column_X = []
        column_Y = []

        # Assign row indices for X_train and Y_train based on r
        for i, row_index in enumerate(self.row_indices):
            if i == r:
                row_test.extend(row_index)
            else:
                row_train.extend(row_index)

        # Assign column indices for X_train and X_test based on s
        for j, column_index in enumerate(self.column_indices):
            if j == s:
                column_Y.extend(column_index)
            else:
                column_X.extend(column_index)

        row_train = sorted(list(map(int, row_train)))
        row_test = sorted(list(map(int, row_test)))
        column_X = sorted(list(map(int, column_X)))
        column_Y = sorted(list(map(int, column_Y)))

        return row_train, row_test, column_X, column_Y

    def Y_train(self, data, row_train, column_Y):
        data = data[row_train][:, column_Y]

        # Initialize the KMeans model with the number of clusters
        kmeans = KMeans(n_clusters=self.n_clusters)

        # Fit the model to the data
        kmeans.fit(data)

        # Get the cluster centroids
        spatial_Y_train = kmeans.cluster_centers_

        # Get the cluster labels for each data point
        temporal_Y_train = kmeans.labels_

        return spatial_Y_train, temporal_Y_train

    def X_train(self, data, row_train, column_X, temporal_Y_train):
        data = data[row_train][:, column_X]

        spatial_X_train = np.array([np.mean(data[temporal_Y_train == cluster_label], axis=0)
                                    for cluster_label in range(self.n_clusters)])

        return spatial_X_train

    def X_test(self, data, row_test, column_X, spatial_X_train):
        data = data[row_test][:, column_X]
        # Compute distances between data points and centroids
        distances = cdist(data, spatial_X_train, metric='euclidean')

        # Assign each data point to the nearest centroid
        temporal_X_test = np.argmin(distances, axis=1)

        return temporal_X_test

    def Y_test(self, data, row_test, column_Y, temporal_X_test, spatial_Y_train):
        data = data[row_test][:, column_Y]

        centroids = spatial_Y_train[temporal_X_test]

        # Compute squared differences between data and centroids
        mean_squared_diff = np.mean(np.sum((data - centroids) ** 2, axis=-1), axis=0)

        return mean_squared_diff

    def validate(self, data, save_dir=None):
        metrics = []
        for i in range(self.partition_rows):
            for j in range(self.partition_columns):
                row_train, row_test, column_X, column_Y = self.fold_indices(i, j)
                spatial_Y_train, temporal_Y_train = self.Y_train(data, row_train, column_Y)
                spatial_X_train = self.X_train(data, row_train, column_X, temporal_Y_train)
                temporal_X_test = self.X_test(data, row_test, column_X, spatial_X_train)
                if save_dir is not None:
                    np.save(os.path.join(save_dir, f'fold_{i + 1}_{j + 1}_spatial_Y_train.npy'), spatial_Y_train)
                    np.save(os.path.join(save_dir, f'fold_{i + 1}_{j + 1}_temporal_Y_train.npy'), temporal_Y_train)
                    np.save(os.path.join(save_dir, f'fold_{i + 1}_{j + 1}_spatial_X_train.npy'), spatial_X_train)
                    np.save(os.path.join(save_dir, f'fold_{i + 1}_{j + 1}_temporal_X_test.npy'), temporal_X_test)
                metric = float(self.Y_test(data, row_test, column_Y, temporal_X_test, spatial_Y_train))
                metrics.append(metric)

        return metrics


class BICVHMM():
    def __init__(self, n_samples=None, n_channels=None, row_indices=None, column_indices=None, save_dir=None,
                 partition_rows=2, partition_columns=2, ):
        '''
        Initialisation of BICVHMM.
        If both row_indices and column indices are not None, use them to initialise BICVHMM.
        Otherwise randomly generate row and column indices.
        Parameters
        ----------
        n_samples: int
            the number of samples
        n_channels: int
            the number of channels (the length of each sample)
        row_indices: str or list
            the list of row indices. Read from file if it's a string.
        column_indices: str or list
            the list of column indices. Read from file if it's a string.
        partition_rows: int
            the number of rows partition
        partition_columns: int
            the number of columns partition

        '''
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.save_dir = save_dir

        # Initialise the class using row_indices/column_indices
        if (row_indices is not None) and (column_indices is not None):
            if isinstance(row_indices, str):
                self.row_indices = npz2list(np.load(row_indices))
            if isinstance(column_indices, str):
                self.column_indices = npz2list(np.load(column_indices))
            self.partition_rows = len(self.row_indices)
            self.partition_columns = len(self.column_indices)
            # Update the number of samples and number of channels
            self.n_samples = sum(len(arr) for arr in self.row_indices)
            self.n_channels = sum(len(arr) for arr in self.column_indices)
        else:
            self.partition_rows = partition_rows
            self.partition_columns = partition_columns
            self.partition_indices()

    def partition_indices(self):
        '''
        Generate partition indices, if save_dir is not None,
        save the indices.
        Parameters
        ----------
        save_dir: str,optional
            the directory to save the indices

        '''
        # Generate random row and column indices
        row_indices = np.arange(self.n_samples)
        column_indices = np.arange(self.n_channels)
        np.random.shuffle(row_indices)
        np.random.shuffle(column_indices)

        # Divide rows into partitions
        self.row_indices = np.array_split(row_indices, self.partition_rows)

        # Divide columns into partitions
        self.column_indices = np.array_split(column_indices, self.partition_columns)

        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            np.savez(os.path.join(self.save_dir, 'row_indices.npz'), *self.row_indices)
            np.savez(os.path.join(self.save_dir, 'column_indices.npz'), *self.column_indices)

    def fold_indices(self, r, s):
        """
        Given the partitions, return the indices of fold (r,s).
        Fold (r,s) treats the rth row subset as "test", and the sth column
        subset as response.
        The parition should be in the format:
        (X_train, Y_train,
         X_test,  Y_test  )
        Parameters
        ----------
        r: int
            the row index
        s: int
            the column index

        Returns
        -------
        row_train: list
            row indices for X_train and Y_train
        row_test: list
            row indices for X_test and Y_test
        column_X: list
            column indices for X_train and X_test
        column_Y: list
            column indices for Y_train and Y_test
        """

        # Assuming row_indices and column_indices are available
        row_train = []
        row_test = []
        column_X = []
        column_Y = []

        # Assign row indices for X_train and Y_train based on r
        for i, row_index in enumerate(self.row_indices):
            if i == r:
                row_test.extend(row_index)
            else:
                row_train.extend(row_index)

        # Assign column indices for X_train and X_test based on s
        for j, column_index in enumerate(self.column_indices):
            if j == s:
                column_Y.extend(column_index)
            else:
                column_X.extend(column_index)

        row_train = sorted(list(map(int, row_train)))
        row_test = sorted(list(map(int, row_test)))
        column_X = sorted(list(map(int, column_X)))
        column_Y = sorted(list(map(int, column_Y)))

        return row_train, row_test, column_X, column_Y

    def Y_train(self, config, train_keys, row_train, column_Y, ):
        # Specify the save directory
        save_dir = os.path.join(config['save_dir'], 'Y_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        prepare_config['load_data']['prepare']['select']['channels'] = column_Y

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in train_keys if key in config},
            'init_kwargs':
                config['init_kwargs']
        }
        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column_Y)
        prepare_config['keep_list'] = row_train

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        params_dir = f'{save_dir}/inf_params/'
        return f'{save_dir}/model/', f'{params_dir}alp.pkl'

    def X_train(self, config, train_keys, row_train, column_X, temporal_Y_train):
        # Update 23rd March 2024
        # A completely new implementation of X_train

        save_dir = os.path.join(config['save_dir'], 'X_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create a new directory "config['save_dir']/X_train/inf_params
        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        shutil.copy(temporal_Y_train, f'{save_dir}inf_params/')

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        prepare_config['load_data']['prepare']['select']['channels'] = column_X

        prepare_config[f'build_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in train_keys if key in config},
        }
        prepare_config[f'build_{config["model"]}']['config_kwargs']['n_channels'] = len(column_X)
        prepare_config['dual_estimation'] = {'concatenate': True}

        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row_train

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)

        return {'means': f'{save_dir}/dual_estimates/means.npy',
                'covs': f'{save_dir}/dual_estimates/covs.npy'}
        #########################################################3
        '''
        save_dir = os.path.join(config['save_dir'], 'X_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Read in the temporal_Y_train
        if isinstance(temporal_Y_train,str):
            # Open the pickle file in binary mode
            with open(temporal_Y_train, 'rb') as f:
                # Load the object from the file
                alpha = pickle.load(f)

        # Specify the load data configuration
        load_data_kwargs = config['load_data']
        load_data_kwargs['prepare']['select']['channels'] = column_X

        # Build up the data object
        data = load_data(**load_data_kwargs)
        ts = data.time_series(prepared=True,concatenate=False)

        means, covs = self._set_state_stats_using_alpha(ts,row_train,alpha)

        np.save(f'{save_dir}/means.npy',means)
        np.save(f'{save_dir}/covs.npy',covs)

        return {'means':f'{save_dir}/means.npy','covs':f'{save_dir}covs.npy'}
        '''

    def _set_state_stats_using_alpha(self, ts, row_train, alpha):
        """Sets the means/covariances using specified state time course.

        Parameters
        ----------
        ts : list of np.ndarray
            Timeseries. Returned from osl_dynamics.data.Data.time_series()
        row_train: list
            Indices to use on the ts.
        alpha: list
            The corresponding state time courses.
        """

        # Mean and covariance for each state
        if isinstance(ts, np.ndarray):
            n_channels = ts.shape[-1]
        elif isinstance(ts, list):
            n_channels = ts[0].shape[-1]
        else:
            raise TypeError('The variable ts type is incorrect!')
        n_states = alpha[0].shape[-1]
        means = np.zeros(
            [n_states, n_channels], dtype=np.float32
        )
        covariances = np.zeros(
            [n_states, n_channels, n_channels],
            dtype=np.float32,
        )

        # If ts is a list, turn into numpy.ndarray first
        if isinstance(ts, list):
            ts = np.stack(ts, axis=0)
        ts = ts[row_train]
        ts_1, ts_2, ts_3 = ts.shape
        data_concat = np.reshape(ts, (ts_1 * ts_2, ts_3))
        # You need hard parcellation first!
        alpha = argmax_time_courses(alpha)
        alpha = np.concatenate(alpha, axis=0)
        for j in range(n_states):
            x = data_concat[alpha[:, j] == 1]
            means[j, :] = np.mean(x, axis=0)
            covariances[j, :, :] = np.cov(x, rowvar=False)

        return means, covariances

    def X_test(self, config, train_keys, row_test, column_X, spatial_X_train):
        # Specify the save directory
        save_dir = os.path.join(config['save_dir'], 'X_test/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        prepare_config['load_data']['prepare']['select']['channels'] = column_X

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in train_keys if key in config},
            'init_kwargs':
                config['init_kwargs']
        }
        # Fix the means and covariances
        prepare_config[f'train_{config["model"]}']['config_kwargs']['learn_means'] = False
        prepare_config[f'train_{config["model"]}']['config_kwargs']['learn_covariances'] = False
        prepare_config[f'train_{config["model"]}']['config_kwargs']['initial_means'] = spatial_X_train['means']
        prepare_config[f'train_{config["model"]}']['config_kwargs']['initial_covariances'] = spatial_X_train['covs']

        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column_X)
        prepare_config['keep_list'] = row_test

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        params_dir = f'{save_dir}/inf_params/'
        return f'{params_dir}/alp.pkl'

    def Y_test(self, config, row_test, column_Y, temporal_X_test, spatial_Y_train):
        # Specify the save directory
        save_dir = os.path.join(config['save_dir'], 'Y_test/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #################################################
        # Update 25th March 2024
        # This is a new implementation of the Y_test using
        # customised test function.
        shutil.copytree(spatial_Y_train, f'{save_dir}/model/')

        # Create a new directory "config['save_dir']/X_train/inf_params
        # And copy the temporal info from X_test
        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        shutil.copy(temporal_X_test, f'{save_dir}inf_params/')

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']
        prepare_config['load_data']['prepare']['select']['channels'] = column_Y
        prepare_config['log_likelihood'] = {}
        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row_test

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)

        with open(f'{save_dir}/metrics.json', 'r') as file:
            # Load the JSON data
            metrics = json.load(file)
        return metrics['log_likelihood']

        ################################################
        '''
        # Read in the temporal_Y_train
        if isinstance(temporal_X_test, str):
            # Open the pickle file in binary mode
            with open(temporal_X_test, 'rb') as f:
                # Load the object from the file
                alpha = pickle.load(f)
        alpha = np.stack(alpha,axis=0)

            # Specify the load data configuration
        load_data_kwargs = config['load_data']
        load_data_kwargs['prepare']['select']['channels'] = column_Y

        # Build up the data object
        data = load_data(**load_data_kwargs)
        ts = np.stack(data.time_series(prepared=True, concatenate=False),axis=0)[row_test]

        from osl_dynamics.models import load
        model = load(spatial_Y_train)
        metrics = float(model.get_posterior_expected_log_likelihood(ts,alpha))

        # Write metrics data into the JSON file
        with open(f'{save_dir}/metrics.json', 'w') as json_file:
            json.dump({'log_likelihood':metrics}, json_file)

        return metrics
        '''

    def validate(self, config, train_keys, i, j):
        row_train, row_test, column_X, column_Y = self.fold_indices(i - 1, j - 1)

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Save the dictionary as a pickle file
        with open(os.path.join(config['save_dir'], 'fold_indices.json'), 'w') as f:
            json.dump({
                'row_train': row_train,
                'row_test': row_test,
                'column_X': column_X,
                'column_Y': column_Y
            }, f)

        spatial_Y_train, temporal_Y_train = self.Y_train(config, train_keys, row_train, column_Y)
        spatial_X_train = self.X_train(config, train_keys, row_train, column_X, temporal_Y_train)
        temporal_X_test = self.X_test(config, train_keys, row_test, column_X, spatial_X_train)
        metric = self.Y_test(config, row_test, column_Y, temporal_X_test, spatial_Y_train)

        # Write metrics data into the JSON file
        with open(os.path.join(config['save_dir'], 'metrics.json'), 'w') as json_file:
            json.dump({'log_likelihood': metric}, json_file)


class BICVHMM_2():
    def __init__(self, n_samples=None, n_channels=None, row_indices=None, column_indices=None, save_dir=None,
                 partition_rows=2, partition_columns=2, ):
        '''
        Initialisation of BICVHMM_2. This is a slightly different implementation of
        cross validation comapred to BICVHMM
        If both row_indices and column indices are not None, use them to initialise BICVHMM_2.
        Otherwise randomly generate row and column indices.
        Parameters
        ----------
        n_samples: int
            the number of samples
        n_channels: int
            the number of channels (the length of each sample)
        row_indices: str or list
            the list of row indices. Read from file if it's a string.
        column_indices: str or list
            the list of column indices. Read from file if it's a string.
        partition_rows: int
            the number of rows partition
        partition_columns: int
            the number of columns partition

        '''
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.save_dir = save_dir

        # Initialise the class using row_indices/column_indices
        if (row_indices is not None) and (column_indices is not None):
            if isinstance(row_indices, str):
                self.row_indices = npz2list(np.load(row_indices))
            if isinstance(column_indices, str):
                self.column_indices = npz2list(np.load(column_indices))
            self.partition_rows = len(self.row_indices)
            self.partition_columns = len(self.column_indices)
            # Update the number of samples and number of channels
            self.n_samples = sum(len(arr) for arr in self.row_indices)
            self.n_channels = sum(len(arr) for arr in self.column_indices)
        else:
            self.partition_rows = partition_rows
            self.partition_columns = partition_columns
            self.partition_indices()

    def partition_indices(self):
        '''
        Generate partition indices, if save_dir is not None,
        save the indices.
        Parameters
        ----------
        save_dir: str,optional
            the directory to save the indices

        '''
        # Generate random row and column indices
        row_indices = np.arange(self.n_samples)
        column_indices = np.arange(self.n_channels)
        np.random.shuffle(row_indices)
        np.random.shuffle(column_indices)

        # Divide rows into partitions
        self.row_indices = np.array_split(row_indices, self.partition_rows)

        # Divide columns into partitions
        self.column_indices = np.array_split(column_indices, self.partition_columns)

        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            np.savez(os.path.join(self.save_dir, 'row_indices.npz'), *self.row_indices)
            np.savez(os.path.join(self.save_dir, 'column_indices.npz'), *self.column_indices)

    def fold_indices(self, r, s):
        """
        Given the partitions, return the indices of fold (r,s).
        Fold (r,s) treats the rth row subset as "test", and the sth column
        subset as response.
        The parition should be in the format:
        (X_train, Y_train,
         X_test,  Y_test  )
        Parameters
        ----------
        r: int
            the row index
        s: int
            the column index

        Returns
        -------
        row_train: list
            row indices for X_train and Y_train
        row_test: list
            row indices for X_test and Y_test
        column_X: list
            column indices for X_train and X_test
        column_Y: list
            column indices for Y_train and Y_test
        """

        # Assuming row_indices and column_indices are available
        row_train = []
        row_test = []
        column_X = []
        column_Y = []

        # Assign row indices for X_train and Y_train based on r
        for i, row_index in enumerate(self.row_indices):
            if i == r:
                row_test.extend(row_index)
            else:
                row_train.extend(row_index)

        # Assign column indices for X_train and X_test based on s
        for j, column_index in enumerate(self.column_indices):
            if j == s:
                column_Y.extend(column_index)
            else:
                column_X.extend(column_index)

        row_train = sorted(list(map(int, row_train)))
        row_test = sorted(list(map(int, row_test)))
        column_X = sorted(list(map(int, column_X)))
        column_Y = sorted(list(map(int, column_Y)))

        return row_train, row_test, column_X, column_Y

    def X_train(self, config, train_keys, row_train, column_X, ):
        # Specify the save directory
        save_dir = os.path.join(config['save_dir'], 'X_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        prepare_config['load_data']['prepare']['select']['channels'] = column_X

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in train_keys if key in config},
            'init_kwargs':
                config['init_kwargs']
        }
        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column_X)
        prepare_config['keep_list'] = row_train

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        params_dir = f'{save_dir}/inf_params/'
        return {'means': f'{params_dir}means.npy',
                'covs': f'{params_dir}covs.npy'}, f'{params_dir}alp.pkl'

    def Y_train(self, config, train_keys, row_train, column_Y, temporal_X_train):
        # Update 23rd March 2024
        # A completely new implementation of X_train

        save_dir = os.path.join(config['save_dir'], 'Y_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create a new directory "config['save_dir']/X_train/inf_params
        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        shutil.copy(temporal_X_train, f'{save_dir}inf_params/')

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        prepare_config['load_data']['prepare']['select']['channels'] = column_Y

        prepare_config[f'build_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in train_keys if key in config},
        }
        prepare_config[f'build_{config["model"]}']['config_kwargs']['n_channels'] = len(column_Y)
        prepare_config['dual_estimation'] = {'concatenate': True}

        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row_train

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)

        return {'means': f'{save_dir}/dual_estimates/means.npy',
                'covs': f'{save_dir}/dual_estimates/covs.npy'}

    def X_test(self, config, train_keys, row_test, column_X, spatial_X_train):
        # Specify the save directory
        save_dir = os.path.join(config['save_dir'], 'X_test/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        prepare_config['load_data']['prepare']['select']['channels'] = column_X

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in train_keys if key in config},
            'init_kwargs':
                config['init_kwargs']
        }
        # Fix the means and covariances
        prepare_config[f'train_{config["model"]}']['config_kwargs']['learn_means'] = False
        prepare_config[f'train_{config["model"]}']['config_kwargs']['learn_covariances'] = False
        prepare_config[f'train_{config["model"]}']['config_kwargs']['initial_means'] = spatial_X_train['means']
        prepare_config[f'train_{config["model"]}']['config_kwargs']['initial_covariances'] = spatial_X_train['covs']

        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column_X)
        prepare_config['keep_list'] = row_test

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        params_dir = f'{save_dir}/inf_params/'
        return f'{params_dir}/alp.pkl'

    def Y_test(self, config, train_keys, row_test, column_Y, temporal_X_test, spatial_Y_train):
        # Specify the save directory
        save_dir = os.path.join(config['save_dir'], 'Y_test/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #################################################
        # Update 25th March 2024
        # This is a new implementation of the Y_test using
        # customised test function.
        # shutil.copytree(spatial_Y_train,f'{save_dir}/model/')

        # Create a new directory "config['save_dir']/X_train/inf_params
        # And copy the temporal info from X_test
        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        shutil.copy(temporal_X_test, f'{save_dir}inf_params/')

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']
        prepare_config['load_data']['prepare']['select']['channels'] = column_Y

        prepare_config[f'build_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in train_keys if key in config},
        }
        prepare_config[f'build_{config["model"]}']['config_kwargs']['n_channels'] = len(column_Y)
        prepare_config[f'build_{config["model"]}']['config_kwargs']['initial_means'] = spatial_Y_train['means']
        prepare_config[f'build_{config["model"]}']['config_kwargs']['initial_covariances'] = spatial_Y_train['covs']

        prepare_config['log_likelihood'] = {}
        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row_test

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)

        with open(f'{save_dir}/metrics.json', 'r') as file:
            # Load the JSON data
            metrics = json.load(file)
        return metrics['log_likelihood']

    def validate(self, config, train_keys, i, j):
        row_train, row_test, column_X, column_Y = self.fold_indices(i - 1, j - 1)

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Save the dictionary as a pickle file
        with open(os.path.join(config['save_dir'], 'fold_indices.json'), 'w') as f:
            json.dump({
                'row_train': row_train,
                'row_test': row_test,
                'column_X': column_X,
                'column_Y': column_Y
            }, f)

        spatial_X_train, temporal_X_train = self.X_train(config, train_keys, row_train, column_X)
        spatial_Y_train = self.Y_train(config, train_keys, row_train, column_Y, temporal_X_train)
        temporal_X_test = self.X_test(config, train_keys, row_test, column_X, spatial_X_train)
        metric = self.Y_test(config, train_keys, row_test, column_Y, temporal_X_test, spatial_Y_train)

        # Write metrics data into the JSON file
        with open(os.path.join(config['save_dir'], 'metrics.json'), 'w') as json_file:
            json.dump({'log_likelihood': metric}, json_file)


class CVBase():
    """
    Base class for bi-cross validation in dFC models.
    The idea comes from [k-means bi cross validation](https://www.tandfonline.com/doi/full/10.1080/10618600.2019.1647846)
    """

    def __init__(self, n_samples=None, n_channels=None, row_indices=None, column_indices=None,
                 save_dir=None, partition_rows=2, partition_columns=2):
        '''
        Initialisation of CVBase.
        If both row_indices and column indices are not None, use them to initialise CVBase.
        Otherwise randomly generate row and column indices.

        Parameters
        ----------
        n_samples: int
            the number of samples
        n_channels: int
            the number of channels (the length of each sample)
        row_indices: str or list
            the list of row indices. Read from file if it's a string.
        column_indices: str or list
            the list of column indices. Read from file if it's a string.
        partition_rows: int
            the number of rows partition
        partition_columns: int
            the number of columns partition
        '''
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.save_dir = save_dir

        # Initialise the class using row_indices/column_indices
        if (row_indices is not None) and (column_indices is not None):
            if isinstance(row_indices, str):
                 row_indices= npz2list(np.load(row_indices))
            if isinstance(column_indices, str):
                column_indices = npz2list(np.load(column_indices))
            self.row_indices = row_indices
            self.column_indices = column_indices
            self.partition_rows = len(self.row_indices)
            self.partition_columns = len(self.column_indices)
            # Update the number of samples and number of channels
            self.n_samples = sum(len(arr) for arr in self.row_indices)
            self.n_channels = sum(len(arr) for arr in self.column_indices)
        else:
            self.partition_rows = partition_rows
            self.partition_columns = partition_columns
            self.partition_indices()

    def partition_indices(self):
        '''
        Generate partition indices, if save_dir is not None,
        save the indices.
        Parameters
        ----------
        save_dir: str,optional
            the directory to save the indices

        '''
        # Generate random row and column indices
        row_indices = np.arange(self.n_samples)
        column_indices = np.arange(self.n_channels)
        np.random.shuffle(row_indices)
        np.random.shuffle(column_indices)

        # Divide rows into partitions
        self.row_indices = np.array_split(row_indices, self.partition_rows)

        # Divide columns into partitions
        self.column_indices = np.array_split(column_indices, self.partition_columns)

        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            np.savez(os.path.join(self.save_dir, 'row_indices.npz'), *self.row_indices)
            np.savez(os.path.join(self.save_dir, 'column_indices.npz'), *self.column_indices)

    def fold_indices(self, r, s):
        """
        Given the partitions, return the indices of fold (r,s).
        Fold (r,s) treats the rth row subset as "test", and the sth column
        subset as response.
        The parition should be in the format:
        (X_train, Y_train,
         X_test,  Y_test  )
        Parameters
        ----------
        r: int
            the row index
        s: int
            the column index

        Returns
        -------
        row_train: list
            row indices for X_train and Y_train
        row_test: list
            row indices for X_test and Y_test
        column_X: list
            column indices for X_train and X_test
        column_Y: list
            column indices for Y_train and Y_test
        """

        # Assuming row_indices and column_indices are available
        row_train = []
        row_test = []
        column_X = []
        column_Y = []

        # Assign row indices for X_train and Y_train based on r
        for i, row_index in enumerate(self.row_indices):
            if i == r:
                row_test.extend(row_index)
            else:
                row_train.extend(row_index)

        # Assign column indices for X_train and X_test based on s
        for j, column_index in enumerate(self.column_indices):
            if j == s:
                column_Y.extend(column_index)
            else:
                column_X.extend(column_index)

        row_train = sorted(list(map(int, row_train)))
        row_test = sorted(list(map(int, row_test)))
        column_X = sorted(list(map(int, column_X)))
        column_Y = sorted(list(map(int, column_Y)))

        return row_train, row_test, column_X, column_Y


class CVKmeans(CVBase):
    '''
    Bi cross validation for K-means model (inherited from CVBase)
    '''

    def __init__(self, n_samples=None, n_channels=None, row_indices=None, column_indices=None,
                 save_dir=None, partition_rows=2, partition_columns=2):
        super().__init__(n_samples, n_channels, row_indices, column_indices,
                         save_dir, partition_rows, partition_columns)

    def full_train(self, config, data, row, column, save_dir=None):
        if isinstance(data, str):
            data = np.load(data)
        data = data[row][:, column]

        if save_dir is None:
            save_dir = f'{config["save_dir"]}/full_train/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize the KMeans model with the number of clusters
        kmeans = KMeans(n_clusters=config['n_clusters'])

        # Fit the model to the data
        kmeans.fit(data)

        # Get the cluster centroids
        spatial_Y_train = kmeans.cluster_centers_

        # Get the cluster labels for each data point
        temporal_Y_train = kmeans.labels_

        # Save the results
        np.save(f'{save_dir}/centre.npy', spatial_Y_train)
        np.save(f'{save_dir}/label.npy', temporal_Y_train)
        return f'{save_dir}/centre.npy', f'{save_dir}/label.npy'

    def infer_spatial(self, config, data, row, column, temporal, save_dir=None):
        if isinstance(data, str):
            data = np.load(data)

        if isinstance(temporal, str):
            temporal = np.load(temporal)

        data = data[row][:, column]

        if save_dir is None:
            save_dir = f'{config["save_dir"]}/infer_spatial/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        spatial = np.array([np.mean(data[temporal == cluster_label], axis=0)
                            for cluster_label in range(config['n_clusters'])])

        np.save(f'{save_dir}centre.npy', spatial)

        return f'{save_dir}centre.npy'

    def infer_temporal(self, config, data, row, column, spatial, save_dir=None):
        if isinstance(data, str):
            data = np.load(data)

        if isinstance(spatial, str):
            spatial = np.load(spatial)

        data = data[row][:, column]

        if save_dir is None:
            save_dir = f'{config["save_dir"]}/infer_spatial/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        distances = cdist(data, spatial, metric='euclidean')

        # Assign each data point to the nearest centroid
        temporal = np.argmin(distances, axis=1)

        np.save(f'{save_dir}label.npy', temporal)

        return f'{save_dir}label.npy'

    def calculate_error(self, config, data, row, column, temporal, spatial, save_dir=None):
        if isinstance(data, str):
            data = np.load(data)

        if isinstance(spatial, str):
            spatial = np.load(spatial)
        if isinstance(temporal, str):
            temporal = np.load(temporal)

        data = data[row][:, column]

        if save_dir is None:
            save_dir = f'{config["save_dir"]}/calculate_error/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        centroids = spatial[temporal]

        # Compute squared differences between data and centroids
        mean_squared_diff = np.mean(np.sum((data - centroids) ** 2, axis=-1), axis=0)

        with open(f'{save_dir}/metrics.json', "w") as file:
            json.dump({'mse': mean_squared_diff}, file)
        return f'{save_dir}/metrics.json'

    def split_column(self,config,data,column_1,column_2,spatial,save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = [os.path.join(config['save_dir'], 'column_1_spatial/'),
                        os.path.join(config['save_dir'], 'column_2_spatial/')]
        for dir in save_dir:
            if not os.path.exists(dir):
                os.makedirs(dir)

        centre = np.load(spatial)
        centre_1 = centre[:,column_1]
        centre_2 = centre[:,column_2]
        np.save(f'{save_dir[0]}/centre.npy',centre_1)
        np.save(f'{save_dir[1]}/centre.npy',centre_2)
        return f'{save_dir[0]}/centre.npy',f'{save_dir[1]}/centre.npy'


    def validate(self, config, data, i, j):
        row_train, row_test, column_X, column_Y = self.fold_indices(i - 1, j - 1)

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Save the dictionary as a pickle file
        with open(os.path.join(config['save_dir'], 'fold_indices.json'), 'w') as f:
            json.dump({
                'row_train': row_train,
                'row_test': row_test,
                'column_X': column_X,
                'column_Y': column_Y
            }, f)
        if str(config['cv_variant']) == '1':
            spatial_Y_train, temporal_Y_train = self.full_train(config, data,row_train, column_Y,
                                                                save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            spatial_X_train = self.infer_spatial(config, data, row_train, column_X, temporal_Y_train,
                                                 save_dir=os.path.join(config['save_dir'], 'X_train/'))
            temporal_X_test = self.infer_temporal(config, data, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, data, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))
        if str(config['cv_variant']) == '2':
            spatial_X_train, temporal_X_train = self.full_train(config, data, row_train, column_X,
                                                                save_dir=os.path.join(config['save_dir'], 'X_train/'))
            spatial_Y_train = self.infer_spatial(config, data, row_train, column_Y, temporal_X_train,
                                                 save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            temporal_X_test = self.infer_temporal(config, data, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, data, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))

        elif str(config['cv_variant']) == '3':
            spatial_XY_train, _ = self.full_train(config, data, row_train, sorted(column_X + column_Y),
                                                  save_dir=os.path.join(config['save_dir'], 'XY_train/'))
            spatial_X_train, spatial_Y_train = self.split_column(config, data, column_X, column_Y, spatial_XY_train,
                                                                 save_dir=[os.path.join(config['save_dir'], 'X_train/'),
                                                                           os.path.join(config['save_dir'], 'Y_train/')]
                                                                 )
            temporal_X_test = self.infer_temporal(config, data, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, data, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))


class CVHMM(CVBase):
    '''
    Bi-cross validation for HMM model (inherited from CVBase)
    '''

    def __init__(self, n_samples=None, n_channels=None, row_indices=None, column_indices=None,
                 save_dir=None, partition_rows=2, partition_columns=2, train_keys=None):
        super().__init__(n_samples, n_channels, row_indices, column_indices,
                         save_dir, partition_rows, partition_columns)
        if train_keys is None:
            self.train_keys = ['n_channels',
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
        else:
            self.train_keys = train_keys

    def full_train(self, config, row, column, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'full_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        # If select key is not specificed before
        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
            'init_kwargs':
                config['init_kwargs']
        }
        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        params_dir = f'{save_dir}/inf_params/'
        return {'means': f'{params_dir}means.npy',
                'covs': f'{params_dir}covs.npy'}, f'{params_dir}alp.pkl'

    def infer_spatial(self, config, row, column, temporal, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'infer_spatial/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Do nothing if n_states = 1
        if config['n_states'] == 1:
            return '0'

        # Create a new directory "config['save_dir']/X_train/inf_params
        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        shutil.move(temporal, f'{save_dir}inf_params/')

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'build_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
        }
        prepare_config[f'build_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
        prepare_config['dual_estimation'] = {'concatenate': True}

        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)

        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)

        # Compress the representation of alp.pkl file
        if os.path.exists(f'{save_dir}inf_params/alp.pkl'):
            from osl_dynamics.inference.modes import prob2onehot
            with open(f'{save_dir}inf_params/alp.pkl', 'rb') as file:
                alpha = pickle.load(file)
            alpha = prob2onehot(alpha)
            os.remove(f'{save_dir}inf_params/alp.pkl')
            with open(f'{save_dir}inf_params/alp.pkl', 'wb') as file:
                pickle.dump(alpha, file)

        return {'means': f'{save_dir}/dual_estimates/means.npy',
                'covs': f'{save_dir}/dual_estimates/covs.npy'}

    def infer_temporal(self, config, row, column, spatial, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'infer_temporal/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Do nothing if n_states = 1
        if config['n_states'] == 1:
            return '0'

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
            'init_kwargs':
                config['init_kwargs']
        }
        # Fix the means and covariances
        prepare_config[f'train_{config["model"]}']['config_kwargs']['learn_means'] = False
        prepare_config[f'train_{config["model"]}']['config_kwargs']['learn_covariances'] = False
        prepare_config[f'train_{config["model"]}']['config_kwargs']['initial_means'] = spatial['means']
        prepare_config[f'train_{config["model"]}']['config_kwargs']['initial_covariances'] = spatial['covs']

        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        params_dir = f'{save_dir}/inf_params/'
        return f'{params_dir}/alp.pkl'

    def calculate_error(self, config, row, column, temporal, spatial, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'calculate_error/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        # Compress the file
        if os.path.exists(temporal):
            from osl_dynamics.array_ops import convert_arrays_to_dtype
            with open(temporal, 'rb') as file:
                alpha = pickle.load(file)
            alpha = convert_arrays_to_dtype(alpha,np.float16)
            os.remove(temporal)
            with open(temporal, 'wb') as file:
                pickle.dump(alpha, file)

            shutil.move(temporal, f'{save_dir}inf_params/')




        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        if config['n_states'] > 1:
            prepare_config[f'build_{config["model"]}'] = {
                'config_kwargs':
                    {key: config[key] for key in self.train_keys if key in config},
            }
            prepare_config[f'build_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
            prepare_config[f'build_{config["model"]}']['config_kwargs']['initial_means'] = spatial['means']
            prepare_config[f'build_{config["model"]}']['config_kwargs']['initial_covariances'] = spatial['covs']

            prepare_config['log_likelihood'] = {'static_FC':False}
        else:
            prepare_config['log_likelihood'] = {'static_FC':True,'spatial':spatial}
        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        return f'{save_dir}/metrics.json'

    def split_column(self, config, column_1, column_2, spatial, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = [os.path.join(config['save_dir'], 'column_1_spatial/'),
                        os.path.join(config['save_dir'], 'column_2_spatial/')]
        for dir in save_dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
        means = np.load(spatial['means'])
        covs = np.load(spatial['covs'])

        means_1 = means[:, column_1]
        means_2 = means[:, column_2]
        covs_1 = (covs[:, :, column_1])[:, column_1, :]
        covs_2 = (covs[:, :, column_2])[:, column_2, :]

        np.save(f'{save_dir[0]}means.npy', means_1)
        np.save(f'{save_dir[0]}covs.npy', covs_1)
        np.save(f'{save_dir[1]}means.npy', means_2)
        np.save(f'{save_dir[1]}covs.npy', covs_2)

        return ({'means': f'{save_dir[0]}means.npy', 'covs': f'{save_dir[0]}covs.npy'},
                {'means': f'{save_dir[1]}means.npy', 'covs': f'{save_dir[1]}covs.npy'})

    def split_row(self, config, row_1, row_2, temporal, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = [os.path.join(config['save_dir'], 'row_1_temporal/'),
                        os.path.join(config['save_dir'], 'row_2_temporal/')]
        for dir in save_dir:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # Read the whole alpha file
        with open(temporal, 'rb') as file:
            # Load the Python object from the pickle file
            alpha = pickle.load(file)

        # Extract alpha_1 and alpha_2
        alpha_1 = [alpha[i] for i in row_1]
        alpha_2 = [alpha[i] for i in row_2]

        with open(f'{save_dir[0]}/alp.pkl', 'wb') as file:
            pickle.dump(alpha_1, file)
        with open(f'{save_dir[1]}/alp.pkl', 'wb') as file:
            pickle.dump(alpha_2, file)

        # Remove the original file to save disc space
        os.remove(temporal)

        return f'{save_dir[0]}/alp.pkl', f'{save_dir[1]}/alp.pkl'

    def validate(self, config, i, j):
        row_train, row_test, column_X, column_Y = self.fold_indices(i - 1, j - 1)

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Save the dictionary as a pickle file
        with open(os.path.join(config['save_dir'], 'fold_indices.json'), 'w') as f:
            json.dump({
                'row_train': row_train,
                'row_test': row_test,
                'column_X': column_X,
                'column_Y': column_Y
            }, f)
        if str(config['cv_variant']) == '1':
            spatial_Y_train, temporal_Y_train = self.full_train(config, row_train, column_Y,
                                                                save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            spatial_X_train = self.infer_spatial(config, row_train, column_X, temporal_Y_train,
                                                 save_dir=os.path.join(config['save_dir'], 'X_train/'))
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))

        elif str(config['cv_variant']) == '2':
            spatial_X_train, temporal_X_train = self.full_train(config, row_train, column_X,
                                                                save_dir=os.path.join(config['save_dir'], 'X_train/'))
            spatial_Y_train = self.infer_spatial(config, row_train, column_Y, temporal_X_train,
                                                 save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))

        elif str(config['cv_variant']) == '3':
            spatial_XY_train, _ = self.full_train(config, row_train, sorted(column_X + column_Y),
                                                  save_dir=os.path.join(config['save_dir'], 'XY_train/'))
            spatial_X_train, spatial_Y_train = self.split_column(config, column_X, column_Y, spatial_XY_train,
                                                                 save_dir=[os.path.join(config['save_dir'], 'X_train/'),
                                                                           os.path.join(config['save_dir'], 'Y_train/')]
                                                                 )
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))
        elif str(config['cv_variant']) == '4':
            _, temporal_X_traintest = self.full_train(config, sorted(row_train + row_test), column_X,
                                                      save_dir=os.path.join(config['save_dir'], 'X_traintest/'))
            temporal_X_train, temporal_X_test = self.split_row(config, row_train, row_test, temporal_X_traintest,
                                                               save_dir=[os.path.join(config['save_dir'], 'X_train/'),
                                                                         os.path.join(config['save_dir'], 'X_test/')])
            spatial_Y_train = self.infer_spatial(config, row_train, column_Y, temporal_X_train,
                                                 save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))
        else:
            raise ValueError('Currently the cv_variant unavailable!')

class CVSWC(CVBase):
    '''
    Bi-cross validation for Sliding Window Correlation (inherited from CVBase)
    '''

    def __init__(self, n_samples=None, n_channels=None, row_indices=None, column_indices=None,
                 save_dir=None, partition_rows=2, partition_columns=2, train_keys=None):
        super().__init__(n_samples, n_channels, row_indices, column_indices,
                         save_dir, partition_rows, partition_columns)
        if train_keys is None:
            self.train_keys = ['n_channels',
                               'n_states',
                               'learn_means',
                               'learn_covariances',
                               'window_length',
                               'window_offset',
                               'window_type',
                               ]
        else:
            self.train_keys = train_keys

    def full_train(self, config, row, column, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'full_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        # If select key is not specificed before
        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
            #'init_kwargs':
            #    config['init_kwargs']
        }
        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)

        params_dir = f'{save_dir}/inf_params/'
        return {'means': f'{params_dir}means.npy',
                'covs': f'{params_dir}covs.npy'}, f'{params_dir}alp.pkl'

    def infer_spatial(self, config, row, column, temporal, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'infer_spatial/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Do nothing if n_states = 1
        if config['n_states'] == 1:
            return '0'

        # Create a new directory "config['save_dir']/X_train/inf_params
        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        shutil.move(temporal, f'{save_dir}inf_params/')

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'train_{config["model"]}_spatial'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
        }
        prepare_config[f'train_{config["model"]}_spatial']['config_kwargs']['n_channels'] = len(column)

        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)

        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)

        return {'means': f'{save_dir}/dual_estimates/means.npy',
                'covs': f'{save_dir}/dual_estimates/covs.npy'}


    def infer_temporal(self, config, row, column, spatial, save_dir=None):

        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'infer_temporal/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Do nothing if n_states = 1
        if config['n_states'] == 1:
            return '0'

        params_dir = f'{save_dir}/inf_params/'
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)
        shutil.copy(spatial['means'], params_dir)
        shutil.copy(spatial['covs'], params_dir)

        # Do nothing if n_states = 1
        #if config['n_states'] == 1:
        #    return '0'

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'train_{config["model"]}_temporal'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
            #'init_kwargs':
            #    config['init_kwargs']
        }
        # Fix the means and covariances
        prepare_config[f'train_{config["model"]}_temporal']['config_kwargs']['learn_means'] = False
        prepare_config[f'train_{config["model"]}_temporal']['config_kwargs']['learn_covariances'] = False

        prepare_config[f'train_{config["model"]}_temporal']['config_kwargs']['n_channels'] = len(column)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)


        return f'{params_dir}/alp.pkl'


    def calculate_error(self, config, row, column, temporal, spatial, save_dir=None):

        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'calculate_error/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        if os.path.exists(temporal):
            shutil.move(temporal, f'{save_dir}inf_params/')
        if os.path.exists(spatial['means']):
            shutil.copy(spatial['means'],f'{save_dir}inf_params/')
        if os.path.exists(spatial['covs']):
            shutil.copy(spatial['covs'], f'{save_dir}inf_params/')

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        if config['n_states'] > 1:
            prepare_config[f'train_{config["model"]}_log_likelihood'] = {
                'config_kwargs':
                    {key: config[key] for key in self.train_keys if key in config},
            }
            prepare_config[f'train_{config["model"]}_log_likelihood']['config_kwargs']['n_channels'] = len(column)

        else:
            prepare_config['log_likelihood'] = {'static_FC':True,'spatial':spatial}
        #prepare_config[f'build_{config["model"]}']['config_kwargs']['initial_means'] = spatial['means']
        #prepare_config[f'build_{config["model"]}']['config_kwargs']['initial_covariances'] = spatial['covs']

        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        
        return f'{save_dir}/metrics.json'


    def split_column(self, config, column_1, column_2, spatial, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = [os.path.join(config['save_dir'], 'column_1_spatial/'),
                        os.path.join(config['save_dir'], 'column_2_spatial/')]
        for dir in save_dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
        means = np.load(spatial['means'])
        covs = np.load(spatial['covs'])

        means_1 = means[:, column_1]
        means_2 = means[:, column_2]
        covs_1 = (covs[:, :, column_1])[:, column_1, :]
        covs_2 = (covs[:, :, column_2])[:, column_2, :]

        np.save(f'{save_dir[0]}means.npy', means_1)
        np.save(f'{save_dir[0]}covs.npy', covs_1)
        np.save(f'{save_dir[1]}means.npy', means_2)
        np.save(f'{save_dir[1]}covs.npy', covs_2)

        return ({'means': f'{save_dir[0]}means.npy', 'covs': f'{save_dir[0]}covs.npy'},
                {'means': f'{save_dir[1]}means.npy', 'covs': f'{save_dir[1]}covs.npy'})

    def split_row(self, config, row_1, row_2, temporal, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = [os.path.join(config['save_dir'], 'row_1_temporal/'),
                        os.path.join(config['save_dir'], 'row_2_temporal/')]
        for dir in save_dir:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # Read the whole alpha file
        with open(temporal, 'rb') as file:
            # Load the Python object from the pickle file
            alpha = pickle.load(file)

        # Extract alpha_1 and alpha_2
        alpha_1 = [alpha[i] for i in row_1]
        alpha_2 = [alpha[i] for i in row_2]

        with open(f'{save_dir[0]}/alp.pkl', 'wb') as file:
            pickle.dump(alpha_1, file)
        with open(f'{save_dir[1]}/alp.pkl', 'wb') as file:
            pickle.dump(alpha_2, file)

        # Remove the original file to save disc space
        os.remove(temporal)

        return f'{save_dir[0]}/alp.pkl', f'{save_dir[1]}/alp.pkl'

    def validate(self, config, i, j):
        row_train, row_test, column_X, column_Y = self.fold_indices(i - 1, j - 1)

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Save the dictionary as a pickle file
        with open(os.path.join(config['save_dir'], 'fold_indices.json'), 'w') as f:
            json.dump({
                'row_train': row_train,
                'row_test': row_test,
                'column_X': column_X,
                'column_Y': column_Y
            }, f)
        if str(config['cv_variant']) == '1':
            spatial_Y_train, temporal_Y_train = self.full_train(config, row_train, column_Y,
                                                                save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            spatial_X_train = self.infer_spatial(config, row_train, column_X, temporal_Y_train,
                                                 save_dir=os.path.join(config['save_dir'], 'X_train/'))
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))

        elif str(config['cv_variant']) == '2':
            spatial_X_train, temporal_X_train = self.full_train(config, row_train, column_X,
                                                                save_dir=os.path.join(config['save_dir'], 'X_train/'))
            spatial_Y_train = self.infer_spatial(config, row_train, column_Y, temporal_X_train,
                                                 save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))

        elif str(config['cv_variant']) == '3':
            spatial_XY_train, _ = self.full_train(config, row_train, sorted(column_X + column_Y),
                                                  save_dir=os.path.join(config['save_dir'], 'XY_train/'))
            spatial_X_train, spatial_Y_train = self.split_column(config, column_X, column_Y, spatial_XY_train,
                                                                 save_dir=[os.path.join(config['save_dir'], 'X_train/'),
                                                                           os.path.join(config['save_dir'], 'Y_train/')]
                                                                 )
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))
        elif str(config['cv_variant']) == '4':
            _, temporal_X_traintest = self.full_train(config, sorted(row_train + row_test), column_X,
                                                      save_dir=os.path.join(config['save_dir'], 'X_traintest/'))
            temporal_X_train, temporal_X_test = self.split_row(config, row_train, row_test, temporal_X_traintest,
                                                               save_dir=[os.path.join(config['save_dir'], 'X_train/'),
                                                                         os.path.join(config['save_dir'], 'X_test/')])
            spatial_Y_train = self.infer_spatial(config, row_train, column_Y, temporal_X_train,
                                                 save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))
        else:
            raise ValueError('Currently the cv_variant unavailable!')

class CVDyNeMo(CVBase):
    '''
    Bi-cross validation for DyNeMo model (inherited from CVBase)
    '''

    def __init__(self, n_samples=None, n_channels=None, row_indices=None, column_indices=None,
                 save_dir=None, partition_rows=2, partition_columns=2, train_keys=None):
        super().__init__(n_samples, n_channels, row_indices, column_indices,
                         save_dir, partition_rows, partition_columns)
        if train_keys is None:
            self.train_keys = ['n_channels',
                          'n_states',
                          'n_modes',
                          'learn_means',
                          'learn_covariances',
                          'learn_trans_prob',
                          'window_length',
                          'window_offset',
                          'initial_means',
                          'initial_covariances',
                          'initial_trans_prob',
                          'sequence_length',
                          'batch_size',
                          'learning_rate',
                          'n_epochs',
                          # The followings are for Dynemo
                          'inference_n_units',
                          'inference_normalization',
                          'model_n_units',
                          'model_normalization',
                          'learn_alpha_temperature',
                          'initial_alpha_temperature',
                          'do_kl_annealing',
                          'kl_annealing_curve',
                          'kl_annealing_sharpness',
                          'n_kl_annealing_epochs'
                               ]
        else:
            self.train_keys = train_keys

    def full_train(self, config, row, column, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'full_train/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        # If select key is not specificed before
        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
            'init_kwargs':
                config['init_kwargs']
        }
        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        params_dir = f'{save_dir}/inf_params/'
        return {'means': f'{params_dir}means.npy',
                'covs': f'{params_dir}covs.npy'}, f'{params_dir}alp.pkl'

    def infer_spatial(self, config, row, column, temporal, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'infer_spatial/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Do nothing if n_states = 1
        if config['n_modes'] == 1:
            return '0'

        # Create a new directory "config['save_dir']/X_train/inf_params
        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        shutil.move(temporal, f'{save_dir}inf_params/')

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'build_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
        }
        prepare_config[f'build_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
        prepare_config['dual_estimation'] = {'concatenate': True}

        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)

        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        '''
        # Compress the representation of alp.pkl file
        if os.path.exists(f'{save_dir}inf_params/alp.pkl'):
            from osl_dynamics.inference.modes import prob2onehot
            with open(f'{save_dir}inf_params/alp.pkl', 'rb') as file:
                alpha = pickle.load(file)
            alpha = prob2onehot(alpha)
            os.remove(f'{save_dir}inf_params/alp.pkl')
            with open(f'{save_dir}inf_params/alp.pkl', 'wb') as file:
                pickle.dump(alpha, file)
        '''
        return {'means': f'{save_dir}/dual_estimates/means.npy',
                'covs': f'{save_dir}/dual_estimates/covs.npy'}

    def infer_temporal(self, config, row, column, spatial, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'infer_temporal/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Do nothing if n_states = 1
        if config['n_modes'] == 1:
            return '0'

        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        prepare_config[f'train_{config["model"]}'] = {
            'config_kwargs':
                {key: config[key] for key in self.train_keys if key in config},
            'init_kwargs':
                config['init_kwargs']
        }
        # Fix the means and covariances
        prepare_config[f'train_{config["model"]}']['config_kwargs']['learn_means'] = False
        prepare_config[f'train_{config["model"]}']['config_kwargs']['learn_covariances'] = False
        prepare_config[f'train_{config["model"]}']['config_kwargs']['initial_means'] = spatial['means']
        prepare_config[f'train_{config["model"]}']['config_kwargs']['initial_covariances'] = spatial['covs']

        prepare_config[f'train_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        params_dir = f'{save_dir}/inf_params/'
        return f'{params_dir}/alp.pkl'

    def calculate_error(self, config, row, column, temporal, spatial, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = os.path.join(config['save_dir'], 'calculate_error/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(f'{save_dir}inf_params/'):
            os.makedirs(f'{save_dir}inf_params/')

        if os.path.exists(temporal):
            shutil.move(temporal, f'{save_dir}inf_params/')




        prepare_config = {}
        prepare_config['load_data'] = config['load_data']

        if 'select' not in prepare_config['load_data']['prepare'].keys():
            prepare_config['load_data']['prepare']['select'] = {}
        prepare_config['load_data']['prepare']['select']['channels'] = column

        if config['n_modes'] > 1:
            prepare_config[f'build_{config["model"]}'] = {
                'config_kwargs':
                    {key: config[key] for key in self.train_keys if key in config},
            }
            prepare_config[f'build_{config["model"]}']['config_kwargs']['n_channels'] = len(column)
            prepare_config[f'build_{config["model"]}']['config_kwargs']['initial_means'] = spatial['means']
            prepare_config[f'build_{config["model"]}']['config_kwargs']['initial_covariances'] = spatial['covs']

            prepare_config['log_likelihood'] = {'static_FC':False}
        else:
            prepare_config['log_likelihood'] = {'static_FC':True,'spatial':spatial}
        # Note the 'keep_list' value is in order (from small to large number)
        prepare_config['keep_list'] = row

        with open(f'{save_dir}/prepared_config.yaml', 'w') as file:
            yaml.safe_dump(prepare_config, file, default_flow_style=False, sort_keys=False)
        run_pipeline_from_file(f'{save_dir}/prepared_config.yaml',
                               save_dir)
        return f'{save_dir}/metrics.json'

    def split_column(self, config, column_1, column_2, spatial, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = [os.path.join(config['save_dir'], 'column_1_spatial/'),
                        os.path.join(config['save_dir'], 'column_2_spatial/')]
        for dir in save_dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
        means = np.load(spatial['means'])
        covs = np.load(spatial['covs'])

        means_1 = means[:, column_1]
        means_2 = means[:, column_2]
        covs_1 = (covs[:, :, column_1])[:, column_1, :]
        covs_2 = (covs[:, :, column_2])[:, column_2, :]

        np.save(f'{save_dir[0]}means.npy', means_1)
        np.save(f'{save_dir[0]}covs.npy', covs_1)
        np.save(f'{save_dir[1]}means.npy', means_2)
        np.save(f'{save_dir[1]}covs.npy', covs_2)

        return ({'means': f'{save_dir[0]}means.npy', 'covs': f'{save_dir[0]}covs.npy'},
                {'means': f'{save_dir[1]}means.npy', 'covs': f'{save_dir[1]}covs.npy'})

    def split_row(self, config, row_1, row_2, temporal, save_dir=None):
        # Specify the save directory
        if save_dir is None:
            save_dir = [os.path.join(config['save_dir'], 'row_1_temporal/'),
                        os.path.join(config['save_dir'], 'row_2_temporal/')]
        for dir in save_dir:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # Read the whole alpha file
        with open(temporal, 'rb') as file:
            # Load the Python object from the pickle file
            alpha = pickle.load(file)

        # Extract alpha_1 and alpha_2
        alpha_1 = [alpha[i] for i in row_1]
        alpha_2 = [alpha[i] for i in row_2]

        with open(f'{save_dir[0]}/alp.pkl', 'wb') as file:
            pickle.dump(alpha_1, file)
        with open(f'{save_dir[1]}/alp.pkl', 'wb') as file:
            pickle.dump(alpha_2, file)

        # Remove the original file to save disc space
        os.remove(temporal)

        return f'{save_dir[0]}/alp.pkl', f'{save_dir[1]}/alp.pkl'

    def validate(self, config, i, j):
        row_train, row_test, column_X, column_Y = self.fold_indices(i - 1, j - 1)

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

        # Save the dictionary as a pickle file
        with open(os.path.join(config['save_dir'], 'fold_indices.json'), 'w') as f:
            json.dump({
                'row_train': row_train,
                'row_test': row_test,
                'column_X': column_X,
                'column_Y': column_Y
            }, f)
        if str(config['cv_variant']) == '1':
            spatial_Y_train, temporal_Y_train,= self.full_train(config, row_train, column_Y,
                                                                save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            spatial_X_train = self.infer_spatial(config, row_train, column_X, temporal_Y_train,
                                                 save_dir=os.path.join(config['save_dir'], 'X_train/'))
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))
        '''
        elif str(config['cv_variant']) == '2':
            spatial_X_train, temporal_X_train = self.full_train(config, row_train, column_X,
                                                                save_dir=os.path.join(config['save_dir'], 'X_train/'))
            spatial_Y_train = self.infer_spatial(config, row_train, column_Y, temporal_X_train,
                                                 save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))

        elif str(config['cv_variant']) == '3':
            spatial_XY_train, _ = self.full_train(config, row_train, sorted(column_X + column_Y),
                                                  save_dir=os.path.join(config['save_dir'], 'XY_train/'))
            spatial_X_train, spatial_Y_train = self.split_column(config, column_X, column_Y, spatial_XY_train,
                                                                 save_dir=[os.path.join(config['save_dir'], 'X_train/'),
                                                                           os.path.join(config['save_dir'], 'Y_train/')]
                                                                 )
            temporal_X_test = self.infer_temporal(config, row_test, column_X, spatial_X_train,
                                                  save_dir=os.path.join(config['save_dir'], 'X_test/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))
        elif str(config['cv_variant']) == '4':
            _, temporal_X_traintest = self.full_train(config, sorted(row_train + row_test), column_X,
                                                      save_dir=os.path.join(config['save_dir'], 'X_traintest/'))
            temporal_X_train, temporal_X_test = self.split_row(config, row_train, row_test, temporal_X_traintest,
                                                               save_dir=[os.path.join(config['save_dir'], 'X_train/'),
                                                                         os.path.join(config['save_dir'], 'X_test/')])
            spatial_Y_train = self.infer_spatial(config, row_train, column_Y, temporal_X_train,
                                                 save_dir=os.path.join(config['save_dir'], 'Y_train/'))
            metric = self.calculate_error(config, row_test, column_Y, temporal_X_test, spatial_Y_train,
                                          save_dir=os.path.join(config['save_dir'], 'Y_test/'))
        else:
            raise ValueError('Currently the cv_variant unavailable!')
        '''