"""Helper functions for manipulating `NumPy \
<https://numpy.org/doc/stable/user/index.html>`_ arrays.

"""

import numpy as np


def get_one_hot(values, n_states=None):
    """Expand a categorical variable to a series of boolean columns
    (one-hot encoding).

    +----------------------+
    | Categorical Variable |
    +======================+
    |           A          |
    +----------------------+
    |           C          |
    +----------------------+
    |           D          |
    +----------------------+
    |           B          |
    +----------------------+

    becomes

    +---+---+---+---+
    | A | B | C | D |
    +===+===+===+===+
    | 1 | 0 | 0 | 0 |
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    +---+---+---+---+
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+
    | 0 | 1 | 0 | 0 |
    +---+---+---+---+

    Parameters
    ----------
    values : np.ndarray | list[np.ndarray]
        1D array of categorical values with shape (n_samples,). The values
        should be integers (0, 1, 2, 3, ... , :code:`n_states` - 1). Or 2D
        array of shape (n_samples, n_states) to be binarized.
        Or list of np.ndarray
    n_states : int, optional
        Total number of states in :code:`values`. Must be at least the number
        of states present in :code:`values`. Default is the number of unique
        values in :code:`values`.

    Returns
    -------
    one_hot : np.ndarray | list[np.ndarray]
        A 2D array containing the one-hot encoded form of :code:`values`.
        Shape is (n_samples, n_states).
        Or list of 2D arrays
    """
    if isinstance(values, list):
        result = []
        for value in values:
            result.append(get_one_hot(value, n_states))
        return result
    if values.ndim == 2:
        values = values.argmax(axis=1)
    if n_states is None:
        n_states = values.max() + 1
    res = np.eye(n_states)[np.array(values).reshape(-1)]
    return res.reshape([*list(values.shape), n_states]).astype(int)


def cov2corr(cov):
    """Converts batches of covariance matrices into batches of correlation
    matrices.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrices. Shape must be (..., N, N).

    Returns
    -------
    corr : np.ndarray
        Correlation matrices. Shape is (..., N, N).
    """
    # Validation
    cov = np.array(cov)
    if cov.ndim < 2:
        raise ValueError("input covariances must have more than 1 dimension.")

    # Extract batches of standard deviations
    std = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    normalisation = np.expand_dims(std, -1) @ np.expand_dims(std, -2)
    return cov / normalisation


def cov2std(cov):
    """Get the standard deviation of batches of covariance matrices.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix. Shape must be (..., N, N).

    Returns
    -------
    std : np.ndarray
        Standard deviations. Shape is (..., N).
    """
    cov = np.array(cov)
    if cov.ndim < 2:
        raise ValueError("input covariances must have more than 1 dimension.")
    return np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))


def stdcorr2cov(stds, corrs, std_diagonal=False):
    """
    Convert batches of standard deviations and correlations into covariances
    Parameters
    ----------
    stds: np.ndarray
        Standard deviations. Shape is (..., N) or (..., N, N) if std_diagonal=True.
    cors: np.ndarray
        Correlation matrices. Shape is (..., N, N).
    std_diagonal: bool
        Whether the standard deviation is in the form of diagonal matrices

    Returns
    -------
    covariances: np.ndarray
        covariance matrices. Shape is (..., N, N)
    """
    if std_diagonal:
        stds = np.diagonal(stds, axis1=-2, axis2=-1)
    return corrs * stds[..., None] * stds[..., None, :]


def cov2stdcorr(covs):
    """
    Converts batches of covariance matrices into batches of standard deviation vectors
    and correlation matrices.

    Parameters
    ----------
    covs: np.ndarray
         Covariance matrices. Shape must be (..., N, N).

    Returns
    -------
    stds: np.ndarray
        Standard deviations. Shape is (..., N).
    corrs: np.ndarray
        Correlation matrices. Shape is (..., N, N).
    """
    # Validation
    if covs.ndim < 2:
        raise ValueError("input covariances must have more than 1 dimension.")

    # Extract batches of standard deviations
    stds = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1))
    normalisation = np.expand_dims(stds, -1) @ np.expand_dims(stds, -2)
    return stds, covs / normalisation


def sliding_window_view(
        x,
        window_shape,
        axis=None,
        *,
        subok=False,
        writeable=False,
):
    """Create a sliding window over an array in arbitrary dimensions.

    Unceremoniously ripped from numpy 1.20,
    `np.lib.stride_tricks.sliding_window_view \
    <https://numpy.org/doc/1.20/reference/generated/\
    numpy.lib.stride_tricks.sliding_window_view.html>`_.
    """
    if np.iterable(window_shape):
        window_shape = tuple(window_shape)
    else:
        window_shape = (window_shape,)

    # First convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = np.core.numeric.normalize_axis_tuple(
            axis,
            x.ndim,
            allow_duplicate=True,
        )
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            msg = "window shape cannot be larger than input array shape"
            raise ValueError(msg)
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return np.lib.stride_tricks.as_strided(
        x,
        strides=out_strides,
        shape=out_shape,
        subok=subok,
        writeable=writeable,
    )


def validate(array, correct_dimensionality, allow_dimensions, error_message):
    """Checks if the dimensionality of an array is correct.

    Parameters
    ----------
    array : np.ndarray
        Array to be checked.
    correct_dimensionality : int
        The desired number of dimensions in the array.
    allow_dimensions : int
        The number of dimensions that is acceptable for the passed array to
        have.
    error_message : str
        Message to print if the array is not valid.

    Returns
    -------
    array : np.ndarray
        Array with the correct dimensionality.
    """
    array = np.array(array)

    # Add dimensions to ensure array has the correct dimensionality
    for dimensionality in allow_dimensions:
        if array.ndim == dimensionality:
            for i in range(correct_dimensionality - dimensionality):
                array = array[np.newaxis, ...]

    # Check no other dimensionality has been passed
    if array.ndim != correct_dimensionality:
        raise ValueError(error_message)

    return array


def check_symmetry(mat, precision=1e-6):
    """Checks if one or more matrices are symmetric.

    Parameters
    ----------
    mat : np.ndarray or list of np.ndarray
        Matrices to be checked. Shape of a matrix should be (..., N, N).
    precision : float, optional
        Precision for comparing values. Corresponds to an absolute tolerance
        parameter. Default is :code:`1e-6`.

    Returns
    -------
    symmetry : np.ndarray of bool
        Array indicating whether matrices are symmetric.
    """
    mat = np.array(mat)
    if mat.ndim < 2:
        msg = "Input matrix must be an array with shape (..., N, N)."
        raise ValueError(msg)
    transpose_axes = np.concatenate((np.arange(mat.ndim - 2), [-1, -2]))
    symmetry = np.all(
        np.isclose(
            mat,
            np.transpose(mat, axes=transpose_axes),
            rtol=0,
            atol=precision,
            equal_nan=True,
        ),
        axis=(-1, -2),
    )
    return symmetry


def ezclump(binary_array):
    """Find the clumps (groups of data with the same values) for a 1D bool
    array.

    Taken wholesale from :code:`numpy.ma.extras.ezclump`.
    """
    if binary_array.ndim > 1:
        binary_array = binary_array.ravel()
    idx = (binary_array[1:] ^ binary_array[:-1]).nonzero()
    idx = idx[0] + 1

    if binary_array[0]:
        if len(idx) == 0:
            return [slice(0, binary_array.size)]

        r = [slice(0, idx[0])]
        r.extend((slice(left, right) for left, right in zip(idx[1:-1:2], idx[2::2])))
    else:
        if len(idx) == 0:
            return []

        r = [slice(left, right) for left, right in zip(idx[:-1:2], idx[1::2])]

    if binary_array[-1]:
        r.append(slice(idx[-1], binary_array.size))

    return r


def slice_length(slice_):
    """Return the length of a slice.

    Parameters
    ----------
    slice_ : slice
        Slice.

    Returns
    -------
    length : int
        Length.
    """
    return slice_.stop - slice_.start


def apply_to_lists(list_of_lists, func, check_empty=True):
    """Apply a function to each list in a list of lists.

    Parameters
    ----------
    list_of_lists : list of list
        List of lists.
    func : callable
        Function to apply to each list.
    check_empty : bool, optional
        Return :code:`0` for empty lists if set as :code:`True`.
        If :code:`False`, the function will be applied to an empty list.

    Returns
    -------
    result : np.ndarray
        Numpy array with the function applied to each list.
    """
    if check_empty:
        return np.array(
            [
                [
                    func(inner_list) if np.any(inner_list) else 0
                    for inner_list in array_list
                ]
                for array_list in list_of_lists
            ],
        )

    return np.array(
        [
            [func(inner_list) for inner_list in array_list]
            for array_list in list_of_lists
        ],
    )


def list_means(list_of_lists):
    """Calculate the mean of each list in a list of lists.

    Parameters
    ----------
    list_of_lists : list of list
        List of lists.

    Returns
    -------
    result : np.ndarray
        Numpy array with the mean of each list.
    """
    return apply_to_lists(list_of_lists, func=np.mean)


def list_stds(list_of_lists):
    """Calculate the standard deviation of each list in a list of lists.

    Parameters
    ----------
    list_of_lists : list of list
        List of lists.

    Returns
    -------
    result : np.ndarray
        Numpy array with the standard deviation of each list.
    """
    return apply_to_lists(list_of_lists, func=np.std)


def npz2list(array):
    '''
    Convert npz instance to a list of numpy arrays.
    Return a list of length one if array is a np.ndarray
    Parameters
    ----------
    array: np.ndarray or np.lib.npyio.NpzFile
        the input array to be converted

    Returns
    -------
    list_of_array: list
        the returned list of np.ndarrays
    '''
    if isinstance(array, np.ndarray):
        return [array]
    elif isinstance(array, np.lib.npyio.NpzFile):
        return [array[key] for key in array.keys()]


def demean_list(data):
    '''
    demean across a list of lists.
    Return the de-meaned list
    Parameters
    ----------
    data: list
        the input list to be demeaned

    Returns
    -------
    demeaned_data: list
        the demeaned data. Nan values are chopped out.
    '''
    # Check whether the input is a list
    if not isinstance(data, list):
        raise TypeError('The input should be a list of lists!')
    data = np.array(data)
    if not data.ndim == 2:
        raise ValueError('The input should be a list of lists!')

    data -= np.nanmean(data, axis=0, keepdims=True)

    return [[elem for elem in row if not np.isnan(elem)] for row in data]


def filter_nan_values(data):
    '''
    Filter out the nan values across a list of lists.
    Return the filtered list
    Parameters
    ----------
    data: list
        the input list to be filtered

    Returns
    -------
    filtered_data: list
        the filtered data. Nan values are chopped out.
        '''
    # Check whether the input is a list
    if not isinstance(data, list):
        raise TypeError('The input should be a list of lists!')
    filtered_data = []
    for sublist in data:
        if not isinstance(sublist, list):
            raise TypeError('The input should be a list of lists!')
        filtered_sublist = [item for item in sublist if not np.isnan(item)]
        filtered_data.append(filtered_sublist)

    return filtered_data


def convert_arrays_to_dtype(arrays, dtype):
    """
    Convert a list of numpy arrays to the specified data type.

    Parameters
    ----------
    arrays: np.ndarray | list of np.ndarray
        the arrays  to convert
    dtype: Desired data type (e.g., np.float16, np.float32, np.float64)

    Returns
    -------
    arrays: np.ndarray | list of np.ndarray
        numpy arrays with the specified data type
    """
    if isinstance(arrays, np.ndarray):
        return arrays.astype(dtype)

    converted_arrays = []
    for arr in arrays:
        converted_arrays.append(arr.astype(dtype))
    return converted_arrays


def estimate_gaussian_distribution(data, nonzero_means=False, keepdims=True, bias=True):
    """
    Estimate the mean and covariance of a Gaussian distribution from given data.

    Parameters
    ----------
    data: np.ndarray
        the (N,M) data to estimate. N is the #samples, and M is #dimensions.
    nonzero_means: bool,optional
        Whether we would like the output means to be zero.
    keepdims: bool,optional
        Whether to keep the first dimension in the output
    bias: bool,optional
        Used in the covariance estimation.
    """
    if nonzero_means:
        mean = np.mean(data, axis=0, keepdims=keepdims)
    else:
        mean = np.zeros((1, data.shape[1])) if keepdims else np.zeros(data.shape[1])

    # Subtract the mean from the data to center it
    centered_data = data - mean

    # Calculate the covariance matrix
    cov = np.cov(centered_data, rowvar=False, bias=bias)
    if keepdims:
        cov = np.expand_dims(cov, axis=0)

    return mean, cov

def estimate_gaussian_log_likelihood(data, means, covs, average=True):
    """
    Calculate the Gaussian log-likelihood given the data, means, and covariances.

    Parameters
    ----------
    data : np.ndarray
        The data to estimate the log-likelihood for. The last dimension should be the number of channels.
        All previous dimensions will be reshaped to form a 2D array where each row is a data point.
    means : np.ndarray
        The means of the Gaussian distribution. Should be of shape (channels,) or (1, channels).
    covs : np.ndarray
        The covariances of the Gaussian distribution. Should be of shape (channels, channels) or (1, channels, channels).
    average : bool, optional
        Whether to average the log-likelihood across all data points (default is True).

    Returns
    -------
    float
        The log-likelihood of the data under the Gaussian distribution.
        If `average` is True, returns the average log-likelihood per data point.
        Otherwise, returns the total log-likelihood.

    Raises
    ------
    ValueError
        If the last dimension (number of channels) is not compatible across data, means, and covs.
    """
    from scipy.stats import multivariate_normal

    # Check if the last dimension is compatible
    data_channels = data.shape[-1]
    means_channels = means.shape[-1] if len(means.shape) > 1 else means.shape[0]
    covs_channels = covs.shape[-1]

    if data_channels != means_channels or data_channels != covs_channels:
        raise ValueError("The number of channels in data, means, and covs must be compatible.")

    # Reshape data to 2D array (num_samples, num_channels)
    data = data.reshape(-1, data_channels)

    # Handle broadcasting of means and covs
    if means.ndim == 1:
        means = means.reshape(1, -1)
    if covs.ndim == 2:
        covs = covs.reshape(1, data_channels, data_channels)

    # Calculate the log-likelihood using scipy's multivariate_normal
    log_likelihoods = multivariate_normal.logpdf(data, mean=means[0], cov=covs[0])

    if average:
        return np.mean(log_likelihoods)
    else:
        return np.sum(log_likelihoods)
