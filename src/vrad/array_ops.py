"""Helper functions using NumPy.

"""
import logging
from typing import List, Union

import numpy as np
from vrad.utils.decorators import transpose

_logger = logging.getLogger("VRAD")


def get_one_hot(values: np.ndarray, n_modes: int = None):
    """Expand a categorical variable to a series of boolean columns (one-hot encoding).

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
    values : numpy.ndarray
        Categorical variable in a 1D array. Values should be integers (i.e. mode 0, 1,
        2, 3, ... , `n_modes`).
    n_modes : int
        Total number of modes in `values`. Must be at least the number of modes
        present in `values`. Default is the number of unique values in `values`.

    Returns
    -------
    one_hot : numpy.ndarray
        A 2D array containing the one-hot encoded form of the input data.

    """
    if values.ndim == 2:
        _logger.info("argmax being taken on shorter axis.")
        values = values.argmax(axis=1)
    if n_modes is None:
        n_modes = values.max() + 1
    res = np.eye(n_modes)[np.array(values).reshape(-1)]
    return res.reshape(list(values.shape) + [n_modes])


@transpose(0, "sequence_1", 1, "sequence_2")
def align_arrays(*sequences, alignment: str = "left") -> List[np.ndarray]:
    """Given a list of sequences, return the sequences trimmed to equal length.

    Given a list of sequences of unequal length, remove either the start, end or a
    portion of both the start and end of the arrays such that their lengths are equal
    to the length of the shortest array.

    If alignment is "left", values will be trimmed from the ends of the arrays
    (i.e. the starts of the arrays will be aligned). If "right", values will be trimmed
    from the starts of the arrays (i.e. the ends will be aligned). If "center", an
    equal amount will be trimmed from the start and end of the array (i.e. the arrays
    are aligned by their middle values.


    Parameters
    ----------
    sequences: list of numpy.ndarray
        Time courses with differing lengths.
    alignment: str
        One of "left", "center" and "right".
    Returns
    -------
    aligned_arrays: list of numpy.ndarray
    """
    min_length = min(len(sequence) for sequence in sequences)

    if alignment == "left":
        return [sequence[:min_length] for sequence in sequences]

    elif alignment == "right":
        return [sequence[-min_length:] for sequence in sequences]
    elif alignment == "center":
        half_length = int(min_length / 2)
        mids = [int(len(sequence) / 2) for sequence in sequences]

        return [
            sequence[mid - half_length : mid + half_length]
            for sequence, mid in zip(sequences, mids)
        ]

    else:
        raise ValueError("Alignment must be left, right or center.")


def from_cholesky(cholesky_matrix: np.ndarray):
    """Given a Cholesky matrix return the recomposed matrix.

    Operates on the assumption that cholesky_matrix is a valid Cholesky decomposition
    A = LL* and performs LL^T to recover A.

    Parameters
    ----------
    cholesky_matrix: numpy.ndarray
        A valid Cholesky decomposition.

    Returns
    -------
    full_matrix: numpy.ndarray
        A = LL^T where L is the Cholesky decomposition of A.
    """
    if cholesky_matrix.ndim == 2:
        return cholesky_matrix @ cholesky_matrix.transpose()
    return cholesky_matrix @ cholesky_matrix.transpose((0, 2, 1))


def mean_diagonal(array: np.ndarray):
    """Set the diagonal of a matrix to the mean of all non-diagonal elements.

    This is primarily useful for plotting without being concerned about the magnitude
    of diagonal values compressing the color scale.

    Parameters
    ----------
    array: numpy.ndarray

    Returns
    -------
    mean_diagonal_array: numpy.ndarray
        Array with diagonal set to mean of non-diagonal elements.

    """
    off_diagonals = ~np.eye(array.shape[0], dtype=bool)
    new_array = array.copy()
    np.fill_diagonal(new_array, array[off_diagonals].mean())
    return new_array


def trace_weights(covariance_matrices: Union[np.ndarray, List[np.ndarray]]):
    """Calculate a weight for each mode in a list of covariances, from their variance.

    Parameters
    ----------
    covariance_matrices: np.ndarray or list of np.ndarray
        A 3D matrix of dimensions [modes x chans x chans] ([modes x covariance]).

    Returns
    -------
    weights: np.ndarray
        The relative weights of each mode, from its variance.

    """
    covariance_matrices = np.array(covariance_matrices)
    if covariance_matrices.ndim == 2:
        covariance_matrices = covariance_matrices[None, ...]
    if np.not_equal(*covariance_matrices.shape[-2:]):
        raise ValueError(
            "Last two dimensions of matrix must be equal. Found {} and {}".format(
                *covariance_matrices.shape[-2:]
            )
        )
    weights = np.trace(covariance_matrices, axis1=1, axis2=2)
    normalized_weights = weights / weights.max()
    return normalized_weights


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """Create a sliding window over an array in arbitrary dimensions.

    Unceremoniously ripped from numpy 1.20, np.lib.stride_tricks.sliding_window_view.
    """
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
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
        axis = np.core.numeric.normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
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
            raise ValueError("window shape cannot be larger than input array shape")
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return np.lib.stride_tricks.as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


def validate(
    array: np.ndarray,
    correct_dimensionality: int,
    allow_dimensions: list,
    error_message: str,
) -> np.ndarray:
    """Checks if an array has been passed correctly.

    This function checks the dimensionality of the array is correct.

    Parameters
    ----------
    array : np.ndarray
        Array to be checked.
    correct_dimensionality : int
        The desired number of dimensions in the array.
    allow_dimensions : int
        The number of dimensions that is acceptable for the passed array to have.
    error_message : str
        Message to print if the array is not valid.

    Returns:
    np.ndarray
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
