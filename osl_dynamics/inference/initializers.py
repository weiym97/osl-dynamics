"""Initializers for TensorFlow layers.

"""

from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers, initializers
from tensorflow.keras.initializers import Initializer

from osl_dynamics import inference

tfb = tfp.bijectors


class WeightInitializer(Initializer):
    """Initialize weights to given value.

    Parameters
    ----------
    initial_value : np.ndarray
        Value to initialise weights to. Note, the shape is not checked.
    """

    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __call__(self, shape, dtype=None):
        return self.initial_value


class RandomWeightInitializer(Initializer):
    """Initialize weights to given value with random noise added.

    Parameters
    ----------
    initial_value : np.ndarray
        Value to initialise weights to. Note, the shape is not checked.
    std : float
        Standard deviation of the noise to add.
    """

    def __init__(self, initial_value, std):
        self.initial_value = tf.cast(initial_value, tf.float32)
        self.std = std

    def __call__(self, shape, dtype=None):
        e = initializers.TruncatedNormal(mean=0.0, stddev=self.std).__call__(
            shape=shape, dtype=tf.float32
        )
        return self.initial_value + e


class IdentityCholeskyInitializer(Initializer):
    """Initialize weights to a flattened cholesky factor of identity
    matrices."""

    def __init__(self):
        # Bijector used to transform learnable vectors to covariance matrices
        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

    def __call__(self, shape, dtype=None):
        n = shape[0]  # n_modes
        m = int(np.sqrt(1 + 8 * shape[1]) / 2 - 0.5)  # n_channels
        diagonals = np.ones([n, m])
        matrices = np.array([np.diag(d) for d in diagonals], dtype=np.float32)
        return self.bijector.inverse(matrices)


class NormalIdentityCholeskyInitializer(Initializer):
    """Initialize weights to a flattened cholesky factor of identity
    matrices with a normal error added to the diagonal.

    Parameters
    ----------
    std : float
        Standard deviation of the error to add.
    """

    def __init__(self, std):
        self.std = std

        # Bijector used to transform learnable vectors to covariance matrices
        self.bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])

    def __call__(self, shape, dtype=None):
        n = shape[0]  # n_modes
        m = int(np.sqrt(1 + 8 * shape[1]) / 2 - 0.5)  # n_channels
        diagonals = initializers.TruncatedNormal(mean=1, stddev=self.std).__call__(
            shape=(n, m), dtype=tf.float32
        )
        matrices = np.array([np.diag(d) for d in diagonals], dtype=np.float32)
        return self.bijector.inverse(matrices)


class NormalCorrelationCholeskyInitializer(Initializer):
    """Initialize weights to a flattened cholesky factor of correlation
    matrices with a normal error added to the flattened cholesky factor.

    Parameters
    ----------
    mean : float
        Mean of the error to add.
    std : float
        Standard deviation of the error to add.
    """

    def __init__(self, std):
        self.std = std

        # Bijector used to transform learnable vectors to covariance matrices
        self.bijector = tfb.Chain(
            [tfb.CholeskyOuterProduct(), tfb.CorrelationCholesky()]
        )

    def __call__(self, shape, dtype=None):
        n = shape[0]  # n_modes
        m = int(np.sqrt(1 + 8 * shape[1]) / 2 + 0.5)  # n_channels
        diagonals = np.ones([n, m])
        matrices = np.array([np.diag(d) for d in diagonals], dtype=np.float32)
        cholesky_factors = self.bijector.inverse(matrices)
        cholesky_factors += initializers.TruncatedNormal(
            mean=0, stddev=self.std
        ).__call__(shape=cholesky_factors.shape, dtype=tf.float32)
        return cholesky_factors


class NormalDiagonalInitializer(Initializer):
    """Initializer for diagonal matrices with a normal error added.

    Parameters
    ----------
    std : float
        Standard deviation of the error to add.
    """

    def __init__(self, std):
        self.std = std

        # Softplus transformation to ensure diagonal is positive
        self.bijector = tfb.Softplus()

    def __call__(self, shape, dtype=None):
        n = shape[0]  # n_modes
        m = shape[1]  # n_channels
        diagonals = initializers.TruncatedNormal(mean=1, stddev=self.std).__call__(
            shape=(n, m), dtype=tf.float32
        )
        return self.bijector.inverse(diagonals)


class CopyTensorInitializer(Initializer):
    """Initialize weights to another Tensor's value.

    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to copy.
    """

    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, shape, dtype=None):
        return self.tensor.read_value()


def reinitialize_layer_weights(layer):
    """Re-initializes the weights in a particular layer.

    Parameters
    ----------
    layer: tf.keras.layers.Layer
        Layer to initialize weights for.

    Note
    ----
    This function relies on each layer having an attribute for the initializer.
    Standard TensorFlow layers have this. You must specify a
    :code:`self.*_initializer` attribute in any custom layer, otherwise this
    function will break.
    """

    # Get the initialisation container
    if hasattr(layer, "cell"):
        init_container = layer.cell
    else:
        init_container = layer

    # Loop through the attributes of the container
    for key in init_container.__dict__:
        if "initializer" not in key:
            # This attribute's not an initializer
            continue

        # Get the initializer object
        initializer = init_container.__dict__[key]
        initializer_type = type(initializer)

        if initializer_type.__name__ in dir(inference.initializers):
            # We have an osl-dynamics initializer
            #
            # By default these will return new random values when
            # called, so we don't need to create a new initializer
            new_initializer = initializer

        elif isinstance(init_container.__dict__[key], Initializer):
            # We have a standard TensorFlow initializer
            #
            # We need to create a new initializer to get new
            # random values
            new_initializer = deepcopy(initializer)

        # Get the variable (i.e. weights) we want to re-initialize
        if key == "recurrent_initializer":
            var = getattr(init_container, "recurrent_kernel")
        else:
            var = getattr(init_container, key.replace("_initializer", ""))

        # Assign new random values to the variable
        if var is not None:
            var.assign(new_initializer(var.shape, var.dtype))


def reinitialize_model_weights(model, keep=None):
    """Re-initialize the weights in a model.

    Parameters
    ----------
    model : tf.keras.Model
        Model to re-initialize weights for.
    keep : list, optional
        List of :code:`str` containing names for layers to not reinitialize.
    """
    if keep is None:
        keep = []

    for layer in model.layers:
        # Skip layers that we want to keep
        if layer.name in keep:
            continue

        # if this is just a single layer
        if not isinstance(layer, Model) and not ("layers" in dir(layer)):
            # If the layer in bidirectional we need to re_initialise the
            # forward and backward layers.
            if isinstance(layer, layers.Bidirectional):
                reinitialize_layer_weights(layer.forward_layer)
                reinitialize_layer_weights(layer.backward_layer)
            else:
                reinitialize_layer_weights(layer)
        # If the layer consists of multiple layers pass the layer back
        # to this function recursively
        else:
            reinitialize_model_weights(layer)


def nfindr_covariances(
    time_series,
    n_modes,
    window_length=20,
    n_restarts=10,
    eigenvalue_floor=1e-2,
    random_seed=None,
):
    """Estimate mode covariances for a soft mixture with N-FINDR.

    Under :code:`C_t = sum_k alpha_tk C_k` (zero mean), the second moment of a
    window is linear in the window-mean alpha, which lies on the simplex:
    :code:`E[S_w] = sum_k a_wk C_k`. So the window second moments lie in a
    simplex whose vertices are the mode covariances (the linear mixing model
    of hyperspectral unmixing). This function:

    1. Computes second moments of non-overlapping windows.
    2. Projects them onto their top :code:`n_modes - 1` principal components.
    3. Finds the :code:`n_modes` windows spanning the largest simplex
       (N-FINDR, Winter 1999).
    4. Computes each window's non-negative, sum-to-one coordinates relative
       to those vertices and fits the covariances to all windows by least
       squares.
    5. Floors the eigenvalues so each covariance is positive definite.

    Note, if no window is close to pure for a mode, the vertex found for it
    is a mixture; the covariances are then a denoised estimate of the purest
    windows rather than the pure modes.

    Parameters
    ----------
    time_series : list of np.ndarray or np.ndarray
        Time series for each session. Each array has shape
        (n_samples, n_channels). Assumed to have zero mean.
    n_modes : int
        Number of modes.
    window_length : int, optional
        Number of samples in each window.
    n_restarts : int, optional
        Number of random starts for N-FINDR.
    eigenvalue_floor : float, optional
        Eigenvalues are floored at this fraction of their mean.
    random_seed : int, optional
        Seed for the N-FINDR starts.

    Returns
    -------
    covariances : np.ndarray
        Mode covariances. Shape is (n_modes, n_channels, n_channels).
    """
    from scipy.optimize import nnls

    if isinstance(time_series, np.ndarray) and time_series.ndim == 2:
        time_series = [time_series]
    n_channels = time_series[0].shape[-1]
    iu = np.triu_indices(n_channels)

    # Window second moments (upper triangles)
    S = []
    for x in time_series:
        x = np.asarray(x, dtype=np.float64)
        n_windows = x.shape[0] // window_length
        w = x[: n_windows * window_length].reshape(n_windows, window_length, n_channels)
        S.append(np.einsum("wti,wtj->wij", w, w)[:, iu[0], iu[1]] / window_length)
    S = np.concatenate(S)
    if len(S) < n_modes:
        raise ValueError("Fewer windows than modes, use a shorter window_length.")

    # Principal components (signal subspace of the simplex)
    mu = S.mean(axis=0)
    Sc = S - mu
    _, eigvecs = np.linalg.eigh(Sc.T @ Sc)
    Y = np.c_[np.ones(len(S)), Sc @ eigvecs[:, ::-1][:, : n_modes - 1]]

    # N-FINDR. The simplex volume is proportional to |det(Y[idx])|, which is
    # linear in each row, so all replacements of row j are scored at once
    # with the j-th column of the adjugate.
    rng = np.random.default_rng(random_seed)
    best_idx, best_volume = None, -np.inf
    for _ in range(n_restarts):
        idx = rng.choice(len(Y), n_modes, replace=False)
        for _ in range(100):
            improved = False
            for j in range(n_modes):
                M = Y[idx]
                det = np.linalg.det(M)
                if det == 0:
                    break
                volumes = np.abs(Y @ (det * np.linalg.inv(M)[:, j]))
                w = np.argmax(volumes)
                if volumes[w] > abs(det) * (1 + 1e-9):
                    idx[j] = w
                    improved = True
            if not improved:
                break
        volume = abs(np.linalg.det(Y[idx]))
        if volume > best_volume:
            best_idx, best_volume = idx.copy(), volume

    # Sum-to-one, non-negative coordinates of every window (the sum-to-one
    # constraint is the heavily weighted first column of Y)
    weight = 100
    vertices = Y[best_idx].T.copy()
    vertices[0] *= weight
    A = np.empty((len(Y), n_modes))
    for i, y in enumerate(Y):
        target = y.copy()
        target[0] *= weight
        A[i] = nnls(vertices, target)[0]
    A /= A.sum(axis=1, keepdims=True)

    # Least-squares covariances from all windows
    D = np.linalg.lstsq(A, S, rcond=None)[0]
    covariances = np.zeros([n_modes, n_channels, n_channels])
    covariances[:, iu[0], iu[1]] = D
    covariances += np.triu(covariances, 1).transpose(0, 2, 1)

    # Positive definite with an eigenvalue floor
    eigvals, eigvecs = np.linalg.eigh(covariances)
    eigvals = np.maximum(eigvals, eigenvalue_floor * eigvals.mean(axis=-1, keepdims=True))
    covariances = np.einsum("kij,kj,klj->kil", eigvecs, eigvals, eigvecs)

    return covariances.astype(np.float32)
