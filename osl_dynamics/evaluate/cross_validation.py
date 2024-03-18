import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class BICVkmeans():
    def __init__(self,n_clusters,n_samples,n_channels,partition_rows=2,partition_columns=2):
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


    def partition_indices(self,save_dir=None):
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
            np.savez(os.path.join(save_dir,'row_indices.npz'), *row_indices)
            np.savez(os.path.join(save_dir, 'row_indices.npz'), *column_indices)

    def fold_indices(self,r,s):
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

        return row_train, row_test, column_X, column_Y

    def Y_train(self,data,row_train,column_Y):
        data = data[row_train][:,column_Y]

        # Initialize the KMeans model with the number of clusters
        kmeans = KMeans(n_clusters=self.n_clusters)

        # Fit the model to the data
        kmeans.fit(data)

        # Get the cluster centroids
        spatial_Y_train = kmeans.cluster_centers_

        # Get the cluster labels for each data point
        temporal_Y_train = kmeans.labels_

        return spatial_Y_train, temporal_Y_train

    def X_train(self,data,row_train,column_X,temporal_Y_train):
        data = data[row_train][:,column_X]

        spatial_X_train = np.array([np.mean(data[temporal_Y_train == cluster_label], axis=0)
                              for cluster_label in range(self.n_clusters)])

        return spatial_X_train

    def X_test(self,data,row_test,column_X,spatial_X_train):
        data = data[row_test][:,column_X]
        # Compute distances between data points and centroids
        distances = cdist(data, spatial_X_train, metric='euclidean')

        # Assign each data point to the nearest centroid
        temporal_X_test = np.argmin(distances, axis=1)

        return temporal_X_test

    


