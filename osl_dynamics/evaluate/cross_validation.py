import numpy as np

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


        def partition_indices(self):
            # Generate random row and column indices
            row_indices = np.arange(self.n_samples)
            column_indices = np.arange(self.n_channels)
            np.random.shuffle(row_indices)
            np.random.shuffle(column_indices)

            # Divide rows into partitions
            row_indices = np.array_split(row_indices, self.partition_rows)

            # Divide columns into partitions
            column_indices = np.array_split(column_indices, self.partition_columns)

            return row_indices, column_indices