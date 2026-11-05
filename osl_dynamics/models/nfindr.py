"""N-FINDR initialisation of soft-mixture mode matrices and means.

Copied from branch precision_mode (osl_dynamics/models/initialization.py,
commit c835acd2) for the PROFUMO DyNeMo path, which predates that module.
"""

from typing import List, Optional, Union

import numpy as np


def nfindr_covariances(
    time_series: Union[np.ndarray, List[np.ndarray]],
    n_modes: int,
    window_length: int = 20,
    n_restarts: int = 10,
    eigenvalue_floor: float = 1e-2,
    random_seed: Optional[int] = None,
    precisions: bool = False,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Estimate mode covariances for a soft mixture with N-FINDR.

    Under :code:`C_t = sum_k alpha_tk C_k` (zero mean), the second moment of
    a window is linear in the window-mean alpha, which lies on the simplex:
    :code:`E[S_w] = sum_k a_wk C_k`. So the window second moments lie in a
    simplex whose vertices are the mode covariances (the linear mixing model
    of hyperspectral unmixing). This function:

    1. Computes second moments of non-overlapping windows.
    2. Projects them onto their top :code:`n_modes - 1` principal
       components.
    3. Finds the :code:`n_modes` windows spanning the largest simplex
       (N-FINDR, Winter 1999).
    4. Computes each window's non-negative, sum-to-one coordinates relative
       to those vertices and fits the covariances to all windows by least
       squares.
    5. Floors the eigenvalues so each covariance is positive definite.

    If the modes mix precisions (:code:`P_t = sum_k alpha_tk P_k`), steps
    1-5 use regularised window precisions,
    :code:`inv((1 - shrinkage) S_w + shrinkage tr(S_w) / n_channels I)`,
    which are roughly linear in the mode precisions. Because the inverse of
    a short-window covariance is noisy, the mode precisions are then refitted
    to the window second moments by maximum likelihood with the coordinates
    fixed (:code:`sum_w -log det P_w + tr(S_w P_w)`, which is convex in the
    mode precisions).

    Note, if no window is close to pure for a mode, the vertex found for it
    is a mixture; the covariances are then a denoised estimate of the purest
    windows rather than the pure modes.

    Parameters
    ----------
    time_series : list of np.ndarray or np.ndarray
        Time series for each session (or sequence). Each array has shape
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
    precisions : bool, optional
        Do the modes mix precisions instead of covariances?
    shrinkage : float, optional
        Shrinkage of the window covariances towards a multiple of the
        identity before they are inverted (only if :code:`precisions=True`).

    Returns
    -------
    covariances : np.ndarray
        Mode covariances. Shape is (n_modes, n_channels, n_channels).

    References
    ----------
    M. E. Winter, "N-FINDR: an algorithm for fast autonomous spectral
    end-member determination in hyperspectral data", Proc. SPIE 3753 (1999).
    """
    if isinstance(time_series, np.ndarray) and time_series.ndim == 2:
        time_series = [time_series]
    n_channels = time_series[0].shape[-1]
    iu = np.triu_indices(n_channels)

    # Window second moments
    S = []
    for x in time_series:
        x = np.asarray(x, dtype=np.float64)
        n_windows = x.shape[0] // window_length
        w = x[: n_windows * window_length].reshape(n_windows, window_length, n_channels)
        S.append(np.einsum("wti,wtj->wij", w, w) / window_length)
    S = np.concatenate(S)

    # Features of each window (upper triangles)
    if precisions:
        scale = np.trace(S, axis1=1, axis2=2)[:, None, None] / n_channels
        F = np.linalg.inv((1 - shrinkage) * S + shrinkage * scale * np.eye(n_channels))
    else:
        F = S
    F = F[:, iu[0], iu[1]]
    A = _nfindr_coordinates(F, n_modes, n_restarts, random_seed)

    # Least-squares mode matrices from all windows
    D = np.linalg.lstsq(A, F, rcond=None)[0]
    matrices = np.zeros([n_modes, n_channels, n_channels])
    matrices[:, iu[0], iu[1]] = D
    matrices += np.triu(matrices, 1).transpose(0, 2, 1)

    # Positive definite with an eigenvalue floor
    eigvals, eigvecs = np.linalg.eigh(matrices)
    eigvals = np.maximum(
        eigvals, eigenvalue_floor * eigvals.mean(axis=-1, keepdims=True)
    )
    matrices = np.einsum("kij,kj,klj->kil", eigvecs, eigvals, eigvecs)

    if precisions:
        matrices = np.linalg.inv(
            _refit_precisions(S, A, matrices, np.random.default_rng(random_seed))
        )
    return matrices.astype(np.float32)


def _refit_precisions(
    S: np.ndarray,
    A: np.ndarray,
    precisions: np.ndarray,
    rng: np.random.Generator,
    n_iter: int = 400,
    learning_rate: float = 0.02,
    max_windows: int = 20000,
) -> np.ndarray:
    """Maximum likelihood mode precisions given the window coordinates.

    Minimises :code:`sum_w -log det P_w + tr(S_w P_w)` with
    :code:`P_w = sum_k a_wk P_k`, using Adam on the Cholesky factors of the
    :code:`P_k`.

    Parameters
    ----------
    S : np.ndarray
        Window second moments. Shape is (n_windows, n_channels, n_channels).
    A : np.ndarray
        Window coordinates. Shape is (n_windows, n_modes).
    precisions : np.ndarray
        Starting mode precisions. Shape is (n_modes, n_channels, n_channels).
    rng : np.random.Generator
        Used to subsample the windows if there are more than
        :code:`max_windows`.
    n_iter : int, optional
        Number of Adam iterations.
    learning_rate : float, optional
        Learning rate, relative to the average size of the Cholesky factors.
    max_windows : int, optional
        Maximum number of windows to use.

    Returns
    -------
    precisions : np.ndarray
        Mode precisions. Shape is (n_modes, n_channels, n_channels).
    """
    if len(S) > max_windows:
        keep = rng.choice(len(S), max_windows, replace=False)
        S, A = S[keep], A[keep]
    L = np.linalg.cholesky(precisions)
    lower = np.tril(np.ones(L.shape[-2:]))
    step = learning_rate * np.abs(L).mean(axis=(1, 2), keepdims=True)
    AS = np.einsum("wk,wij->kij", A, S)
    m, v = np.zeros_like(L), np.zeros_like(L)
    for i in range(1, n_iter + 1):
        P = np.einsum("wk,kij->wij", A, L @ L.transpose(0, 2, 1))
        grad = (AS - np.einsum("wk,wij->kij", A, np.linalg.inv(P))) / len(S)
        grad = 2 * (grad @ L) * lower
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        L = L - step * (m / (1 - 0.9**i)) / (np.sqrt(v / (1 - 0.999**i)) + 1e-12)
    return L @ L.transpose(0, 2, 1)


def nfindr_means(
    time_series: Union[np.ndarray, List[np.ndarray]],
    n_modes: int,
    window_length: int = 25,
    n_restarts: int = 10,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Estimate mode means for a soft mixture with N-FINDR.

    Under :code:`m_t = sum_k alpha_tk mu_k`, the mean of a window is a
    sum-to-one mixture of the mode means, so the window means lie in a
    simplex whose vertices are the mode means. Each session's average is
    removed first, so that offsets shared by all modes of a session do not
    spread out the simplex, and the average of the data is added back to
    the mode means. See :func:`nfindr_covariances` for the method.

    Parameters
    ----------
    time_series : list of np.ndarray or np.ndarray
        Time series for each session. Each array has shape
        (n_samples, n_channels).
    n_modes : int
        Number of modes.
    window_length : int, optional
        Number of samples in each window.
    n_restarts : int, optional
        Number of random starts for N-FINDR.
    random_seed : int, optional
        Seed for the N-FINDR starts.

    Returns
    -------
    means : np.ndarray
        Mode means. Shape is (n_modes, n_channels).
    """
    if isinstance(time_series, np.ndarray) and time_series.ndim == 2:
        time_series = [time_series]
    n_channels = time_series[0].shape[-1]

    # Window means relative to the session average
    W = []
    for x in time_series:
        x = np.asarray(x, dtype=np.float64)
        n_windows = x.shape[0] // window_length
        w = x[: n_windows * window_length].reshape(n_windows, window_length, n_channels)
        W.append(w.mean(axis=1) - x.mean(axis=0))
    W = np.concatenate(W)
    A = _nfindr_coordinates(W, n_modes, n_restarts, random_seed)

    # Least-squares means from all windows
    means = np.linalg.lstsq(A, W, rcond=None)[0]
    means += np.mean(np.concatenate(time_series), axis=0)
    return means.astype(np.float32)


def _nfindr_coordinates(
    S: np.ndarray, n_modes: int, n_restarts: int, random_seed: Optional[int]
) -> np.ndarray:
    """Coordinates of feature vectors in the simplex found by N-FINDR.

    Parameters
    ----------
    S : np.ndarray
        Feature vector for each window. Shape is (n_windows, n_features).
    n_modes : int
        Number of vertices.
    n_restarts : int
        Number of random starts for N-FINDR.
    random_seed : int
        Seed for the N-FINDR starts.

    Returns
    -------
    A : np.ndarray
        Non-negative, sum-to-one coordinates of each window.
        Shape is (n_windows, n_modes).
    """
    from scipy.optimize import nnls

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
    return A
