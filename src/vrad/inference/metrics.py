"""metrics to analyse model performance.

"""
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.linalg import eigvalsh
from sklearn.metrics import confusion_matrix as sklearn_confusion
from tqdm import trange
from vrad.utils.decorators import transpose


def alpha_correlation(alpha_1: np.ndarray, alpha_2: np.ndarray) -> np.ndarray:
    """Calculates the correlation between modes of two alpha time series.

    Parameters
    ----------
    alpha_1 : np.ndarray
        First alpha time series. Shape is (n_samples, n_modes).
    alpha_2 : np.ndarray
        Second alpha time series. Shape is (n_samples, n_modes).

    Returns
    -------
    np.ndarray
        Correlation of each mode in the corresponding alphas.
        Shape is (n_modes,).
    """
    if alpha_1.shape[1] != alpha_2.shape[1]:
        raise ValueError(
            "alpha_1 and alpha_2 shapes are incomptible. "
            + f"alpha_1.shape={alpha_1.shape}, alpha_2.shape={alpha_2.shape}."
        )
    n_modes = alpha_1.shape[1]
    corr = np.corrcoef(alpha_1, alpha_2, rowvar=False)
    corr = np.diagonal(corr[:n_modes, n_modes:])
    return corr


@transpose("mode_time_course_1", 0, "mode_time_course_2", 1)
def confusion_matrix(
    mode_time_course_1: np.ndarray, mode_time_course_2: np.ndarray
) -> np.ndarray:
    """Calculate the confusion matrix of two mode time courses.

    For two mode-time-courses, calculate the confusion matrix (i.e. the
    disagreement between the mode selection for each sample). If either sequence is
    two dimensional, it will first have argmax(axis=1) applied to it. The produces the
    expected result for a one-hot encoded sequence but other inputs are not guaranteed
    to behave.

    This function is a wrapper for sklearn.metrics.confusion_matrix.

    Parameters
    ----------
    mode_time_course_1: numpy.ndarray
    mode_time_course_2: numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Confusion matrix
    """
    if mode_time_course_1.ndim == 2:
        mode_time_course_1 = mode_time_course_1.argmax(axis=1)
    if mode_time_course_2.ndim == 2:
        mode_time_course_2 = mode_time_course_2.argmax(axis=1)
    if not ((mode_time_course_1.ndim == 1) and (mode_time_course_2.ndim == 1)):
        raise ValueError("Both mode time courses must be either 1D or 2D.")

    return sklearn_confusion(mode_time_course_1, mode_time_course_2)


def dice_coefficient_1d(sequence_1: np.ndarray, sequence_2: np.ndarray) -> float:
    """Calculate the Dice coefficient of a discrete array

    Given two sequences containing a number of discrete elements (i.e. a
    categorical variable), calculate the Dice coefficient of those sequences.

    The Dice coefficient is 2 times the number of equal elements (equivalent to
    true-positives) divided by the sum of the total number of elements.

    Parameters
    ----------
    sequence_1 : numpy.ndarray
        A sequence containing discrete elements.
    sequence_2 : numpy.ndarray
        A sequence containing discrete elements.

    Returns
    -------
    float
        The Dice coefficient of the two sequences.

    Raises
    ------
    ValueError
        If either sequence is not one dimensional.
    """
    if (sequence_1.ndim, sequence_2.ndim) != (1, 1):
        raise ValueError(
            f"sequences must be 1D: {(sequence_1.ndim, sequence_2.ndim)} != (1, 1)."
        )
    if (sequence_1.dtype, sequence_2.dtype) != (int, int):
        raise TypeError("Both sequences must be integer (categorical).")

    return 2 * ((sequence_1 == sequence_2).sum()) / (len(sequence_1) + len(sequence_2))


@transpose(0, 1, "sequence_1", "sequence_2")
def dice_coefficient(sequence_1: np.ndarray, sequence_2: np.ndarray) -> float:
    """Wrapper method for `dice_coefficient`.

    If passed a one-dimensional array, it will be sent straight to `dice_coefficient`.
    Given a two-dimensional array, it will perform an argmax calculation on each sample.
    The default axis for this is zero, i.e. each row represents a sample.

    Parameters
    ----------
    sequence_1 : numpy.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
    sequence_2 : numpy.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
    axis_1 : int
        For a 2D sequence_1, the axis on which to perform argmax. Default is 0.
    axis_2 : int
        For a 2D sequence_2, the axis on which to perform argmax. Default is 0.

    Returns
    -------
    float
        The Dice coefficient of the two sequences.

    See Also
    --------
    dice_coefficient_1D : Dice coefficient of 1D categorical sequences.
    """
    if (len(sequence_1.shape) not in [1, 2]) or (len(sequence_2.shape) not in [1, 2]):
        raise ValueError("Both sequences must be either 1D or 2D")
    if (len(sequence_1.shape) == 1) and (len(sequence_2.shape) == 1):
        return dice_coefficient_1d(sequence_1, sequence_2)
    if len(sequence_1.shape) == 2:
        sequence_1 = sequence_1.argmax(axis=1)
    if len(sequence_2.shape) == 2:
        sequence_2 = sequence_2.argmax(axis=1)
    return dice_coefficient_1d(sequence_1, sequence_2)


def frobenius_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Calculates the frobenius norm of the difference of two matrices.

    The Frobenius norm is calculated as sqrt( Sum_ij abs(a_ij - b_ij)^2 ).

    Parameters
    ----------
    A : np.ndarray
        First matrix. Shape must be (n_modes, n_channels, n_channels) or
        (n_channels, n_channels).
    B : np.ndarray
        Second matrix. Shape must be (n_modes, n_channels, n_channels) or
        (n_channels, n_channels).

    Returns
    -------
    float
        The Frobenius norm of the difference of A and B. If A and B are
        stacked matrices, we sum the Frobenius norm of each sub-matrix.
    """
    if A.ndim == 2 and B.ndim == 2:
        norm = np.linalg.norm(A - B, ord="fro")
    elif A.ndim == 3 and B.ndim == 3:
        norm = np.linalg.norm(A - B, ord="fro", axis=(1, 2))
        norm = np.sum(norm)
    else:
        raise ValueError(
            f"Shape of A and/or B is incorrect. A.shape={A.shape}, B.shape={B.shape}."
        )
    return norm


def log_likelihood(
    time_series: np.ndarray,
    alpha: np.ndarray,
    covariances: np.ndarray,
    sequence_length: int,
    means: np.ndarray = None,
):
    """Calculate the negative log-likelihood.

    We calculate the negative log-likelihood using a tensorflow implementation by
    recursively calling tf_nll because it's quicker.

    Parameters
    ----------
    time_series : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    alpha : np.ndarray
        Inferred mode mixing factors. Shape must be (n_samples, n_modes).
    covariances : np.ndarray
        Inferred mode covariances. Shape must be (n_modes, n_channels, n_channels).
    means : np.ndarray
        Inferred mean vectors. Shape must be (n_modes, n_channels).
    sequence_length : int
        Length of time series to recursively pass to tf_nll.
    """
    # Validation
    if time_series.shape[0] != alpha.shape[0]:
        raise ValueError("time_series and alpha must have the same length.")

    if time_series.shape[0] % sequence_length != 0:
        raise ValueError("time_series and alpha must be divisible by sequence_length.")

    time_series = time_series.astype(np.float32)
    alpha = alpha.astype(np.float32)
    covariances = covariances.astype(np.float32)
    if means is not None:
        means = means.astype(np.float32)

    #  Convert means and covariances to tensors
    if means is None:
        m = tf.zeros([covariances.shape[0], covariances.shape[1]])
    else:
        m = tf.constant(means)
    C = tf.constant(covariances)

    # Number times to call tf_nll
    n_calls = time_series.shape[0] // sequence_length

    nll = []
    for i in trange(n_calls, desc="Calculating log-likelihood", ncols=98):

        # Convert data to tensors
        x = tf.constant(time_series[i * sequence_length : (i + 1) * sequence_length])
        a = tf.constant(alpha[i * sequence_length : (i + 1) * sequence_length])

        # Calculate the negative log-likelihood for each sequence
        nll.append(tf_nll(x, a, m, C))

    # Return the sum for all sequences
    return np.sum(nll)


@tf.function
def tf_nll(x: tf.constant, alpha: tf.constant, mu: tf.constant, D: tf.constant):
    """Calculates the negative log likelihood using a tensorflow implementation.

    Parameters
    ----------
    x : tf.constant
        Time series data. Shape must be (sequence_length, n_channels).
    alpha : tf.constant
        Mode mixing factors. Shape must be (sequence_length, n_modes).
    mu : tf.constant
        Mode mean vectors. Shape must be (n_modes, n_channels).
    D : tf.constant
        Mode covariances. Shape must be (n_modes, n_channels, n_channels).
    """
    # Calculate the mean: m = Sum_j alpha_jt mu_j
    alpha = tf.expand_dims(alpha, axis=-1)
    mu = tf.expand_dims(mu, axis=0)
    m = tf.reduce_sum(tf.multiply(alpha, mu), axis=1)

    # Calculate the covariance: C = Sum_j alpha_jt D_j
    alpha = tf.expand_dims(alpha, axis=-1)
    D = tf.expand_dims(D, axis=0)
    C = tf.reduce_sum(tf.multiply(alpha, D), axis=1)

    # Calculate the log-likelihood at each time point
    mvn = tfp.distributions.MultivariateNormalTriL(
        loc=m,
        scale_tril=tf.linalg.cholesky(C + 1e-6 * tf.eye(C.shape[-1])),
    )
    ll = mvn.log_prob(x)

    # Sum over time and return the negative log-likelihood
    return -tf.reduce_sum(ll, axis=0)


def mode_covariance_correlations(
    mode_covariances: np.ndarray, remove_diagonal: bool = True
) -> np.ndarray:
    """Calculate the correlation between elements of the mode covariances.

    Parameters
    ----------
    mode_covariances : np.ndarray
        Mode covariances matrices.
        Shape must be (n_modes, n_channels, n_channels).

    Returns
    -------
    np.ndarray
        Correlation between elements of each mode covariance.
        Shape is (n_modes, n_modes).
    """
    n_modes = mode_covariances.shape[0]
    mode_covariances = mode_covariances.reshape(n_modes, -1)
    correlations = np.corrcoef(mode_covariances)
    correlations -= np.eye(n_modes)
    return correlations


def riemannian_distance(M1: np.ndarray, M2: np.ndarray) -> float:
    """Calculate the Riemannian distance between two matrices.

    The Riemannian distance is defined as: d = (sum log(eig(M_1 * M_2))) ^ 0.5

    Parameters
    ----------
    M1 : np.ndarray
    M2 : np.ndarray

    Returns
    -------
    np.ndarray
    """
    d = np.sqrt(np.sum((np.log(eigvalsh(M1, M2)) ** 2)))
    return d


def mode_covariance_riemannian_distances(mode_covariances: np.ndarray) -> np.ndarray:
    """Calculate the Riemannian distance between mode covariances.

    Parameters
    ----------
    mode_covariances : np.ndarray
        Mode covariances. Shape must be (n_modes, n_channels, n_channels).

    Returns
    -------
    np.ndarray
        Matrix containing the Riemannian distances between modes.
        Shape is (n_modes, n_modes).
    """
    n_modes = mode_covariances.shape[0]
    riemannian_distances = np.empty([n_modes, n_modes])
    for i in range(n_modes):
        for j in range(n_modes):
            riemannian_distances[i][j] = riemannian_distance(
                mode_covariances[i], mode_covariances[j]
            )
    return riemannian_distances


def rv_coefficient(M: list) -> float:
    """Calculate the RV coefficient for two matrices.

    Parameters
    ----------
    M : list of np.ndarray
        List of matrices.

    Returns
    -------
    float
        RV coefficient.
    """
    # First compute the scalar product matrices for each data set X
    scal_arr_list = []

    for arr in M:
        scal_arr = np.dot(arr, np.transpose(arr))
        scal_arr_list.append(scal_arr)

    # Now compute the 'between study cosine matrix' C
    C = np.zeros((len(M), len(M)), float)

    for index, element in np.ndenumerate(C):
        nom = np.trace(
            np.dot(np.transpose(scal_arr_list[index[0]]), scal_arr_list[index[1]])
        )
        denom1 = np.trace(
            np.dot(np.transpose(scal_arr_list[index[0]]), scal_arr_list[index[0]])
        )
        denom2 = np.trace(
            np.dot(np.transpose(scal_arr_list[index[1]]), scal_arr_list[index[1]])
        )
        Rv = nom / np.sqrt(np.dot(denom1, denom2))
        C[index[0], index[1]] = Rv

    return C
