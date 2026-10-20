"""Reference implementations from the bi-cross-validation research code.

These are copies of functions from the WIP branch of
github.com/weiym97/osl-dynamics (commit 8e2aa858), which was used for the
published results. They are only changed to run without the rest of that
code base (e.g. :code:`self.config.n_channels` is passed as an argument).
The tests check that :py:mod:`osl_dynamics.evaluate` gives the same results.
"""

import os
import json
import warnings

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal
from sklearn.model_selection import ShuffleSplit, KFold


# osl_dynamics/evaluate/cross_validation.py
class CVSplit:
    def __init__(
        self, split_row=None, split_column=None, strategy="combination", **kwargs
    ):
        self.split_row = split_row
        self.split_column = split_column
        self.strategy = strategy
        self.kwargs = kwargs
        if strategy not in ["combination", "pairing"]:
            raise ValueError("Strategy must be 'combination' or 'pairing'.")
        if not (split_row or split_column):
            raise ValueError(
                "At least one of split_row or split_column must be provided."
            )
        self.row_splitter = self._init_splitter(split_row) if split_row else None
        self.column_splitter = (
            self._init_splitter(split_column) if split_column else None
        )

    def _init_splitter(self, split_config):
        n_samples = split_config["n_samples"]
        method = split_config.get("method", "ShuffleSplit")
        method_kwargs = split_config.get("method_kwargs", {})
        if method == "ShuffleSplit":
            splitter = ShuffleSplit(
                n_splits=method_kwargs["n_splits"],
                train_size=method_kwargs["train_size"],
                **self.kwargs,
            )
        elif method == "KFold":
            splitter = KFold(n_splits=method_kwargs["n_splits"], **self.kwargs)
        else:
            raise ValueError(
                "Unsupported cross-validation method: use 'ShuffleSplit' or 'KFold'."
            )
        return splitter, n_samples

    def split(self):
        row_splits = (
            list(self.row_splitter[0].split(range(self.row_splitter[1])))
            if self.row_splitter
            else [([], [])]
        )
        column_splits = (
            list(self.column_splitter[0].split(range(self.column_splitter[1])))
            if self.column_splitter
            else [([], [])]
        )
        if self.strategy == "combination":
            for row_train, row_test in row_splits:
                for col_train, col_test in column_splits:
                    yield row_train, row_test, col_train, col_test
        elif self.strategy == "pairing":
            n_splits = min(len(row_splits), len(column_splits))
            for i in range(n_splits):
                yield (*row_splits[i], *column_splits[i])

    def get_n_splits(self):
        row_splits = (
            len(list(self.row_splitter[0].split(range(self.row_splitter[1]))))
            if self.row_splitter
            else 1
        )
        column_splits = (
            len(list(self.column_splitter[0].split(range(self.column_splitter[1]))))
            if self.column_splitter
            else 1
        )
        if self.strategy == "combination":
            return row_splits * column_splits
        elif self.strategy == "pairing":
            return min(row_splits, column_splits)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, split in enumerate(self.split()):
            if self.row_splitter and self.column_splitter:
                row_train, row_test, col_train, col_test = split
                split_dict = {
                    "row_train": sorted(row_train.tolist()),
                    "row_test": sorted(row_test.tolist()),
                    "column_X": sorted(col_train.tolist()),
                    "column_Y": sorted(col_test.tolist()),
                }
            elif self.row_splitter:
                row_train, row_test, _, _ = split
                split_dict = {
                    "row_train": sorted(row_train.tolist()),
                    "row_test": sorted(row_test.tolist()),
                }
            elif self.column_splitter:
                _, _, col_train, col_test = split
                split_dict = {
                    "column_X": sorted(col_train.tolist()),
                    "column_Y": sorted(col_test.tolist()),
                }
            else:
                raise ValueError("No row or column splitter is defined.")
            file_path = os.path.join(save_dir, f"fold_indices_{i + 1}.json")
            with open(file_path, "w") as f:
                json.dump(split_dict, f, indent=4)


# osl_dynamics/inference/modes.py
def argmax_time_courses(alpha):
    n_modes = alpha.shape[1]
    tc = alpha.argmax(axis=1)
    return np.eye(n_modes)[tc]


# osl_dynamics/models/hmm.py: Model.dual_estimation (concatenate=True) with
# the nested _single_dual_estimation
def hmm_dual_estimation(
    data,
    alpha,
    n_channels,
    learn_means,
    initial_means,
    covariances_epsilon,
    method="sample",
):
    data = [np.concatenate(data, axis=0)]
    alpha = [np.concatenate(alpha, axis=0)]
    n_states = alpha[0].shape[1]

    def _single_dual_estimation(a, x, method="sample"):
        if method == "sample":
            a = argmax_time_courses(a)
        sum_a = np.sum(a, axis=0)
        if learn_means:
            session_means = np.empty((n_states, n_channels))
            for state in range(n_states):
                if method == "sample":
                    session_means[state] = (
                        np.sum(x * a[:, state, None], axis=0) / sum_a[state]
                    )
                elif method == "mle":
                    session_means[state] = np.average(x, axis=0, weights=a[:, state])
        else:
            session_means = initial_means
        session_covariances = np.empty((n_states, n_channels, n_channels))
        for state in range(n_states):
            diff = x - session_means[state]
            if method == "sample":
                if sum_a[state] == 0:
                    session_covariances[state] = np.eye(n_channels)
                else:
                    session_covariances[state] = (
                        np.sum(
                            diff[:, :, None]
                            * diff[:, None, :]
                            * a[:, state, None, None],
                            axis=0,
                        )
                        / sum_a[state]
                    )
            elif method == "mle":
                weights = a[:, state]
                weighted_diff = diff * np.sqrt(weights[:, None])
                session_covariances[state] = np.dot(
                    weighted_diff.T, weighted_diff
                ) / np.sum(weights)
            session_covariances[state] += covariances_epsilon * np.eye(n_channels)
        return session_means, session_covariances

    return _single_dual_estimation(alpha[0], data[0], method=method)


# osl_dynamics/models/hmm.py: Model.get_posterior_expected_log_likelihood
# with average=True. log_likelihood has shape (n_sessions, n_samples,
# n_states) and was calculated by the model's get_log_likelihood.
def hmm_average_log_likelihood(log_likelihood, gamma):
    gamma = np.reshape(gamma, log_likelihood.shape)
    return np.mean(log_likelihood * gamma) * gamma.shape[2]


# osl_dynamics/models/dynemo.py: Model.get_posterior_expected_log_likelihood
# with average=True
def dynemo_average_log_likelihood(x, alpha, means, covs):
    alpha = np.reshape(alpha, (alpha.shape[0] * alpha.shape[1], -1))
    x = np.reshape(x, (x.shape[0] * x.shape[1], -1))
    m_t = np.einsum("ns,sc->nc", alpha, means)
    C_t = np.einsum("ns,sij->nij", alpha, covs)
    log_likelihood = np.array(
        [
            multivariate_normal.logpdf(x[i], mean=m_t[i], cov=C_t[i])
            for i in range(len(x))
        ]
    )
    return np.average(log_likelihood)


# osl_dynamics/array_ops.py
def estimate_gaussian_distribution(data, nonzero_means=False, keepdims=True, bias=True):
    if nonzero_means:
        mean = np.mean(data, axis=0, keepdims=keepdims)
    else:
        mean = np.zeros((1, data.shape[1])) if keepdims else np.zeros(data.shape[1])
    centered_data = data - mean
    cov = np.cov(centered_data, rowvar=False, bias=bias)
    if keepdims:
        cov = np.expand_dims(cov, axis=0)
    return mean, cov


def estimate_gaussian_log_likelihood(data, means, covs, average=True):
    data_channels = data.shape[-1]
    data = data.reshape(-1, data_channels)
    if means.ndim == 1:
        means = means.reshape(1, -1)
    if covs.ndim == 2:
        covs = covs.reshape(1, data_channels, data_channels)
    log_likelihoods = multivariate_normal.logpdf(data, mean=means[0], cov=covs[0])
    if average:
        return np.mean(log_likelihoods)
    else:
        return np.sum(log_likelihoods)


# osl_dynamics/models/swc.py: Model.infer_temporal and
# Model.get_posterior_expected_log_likelihood (average=True)
def swc_infer_temporal(ts, covs, window_length, window_offset):
    n_states = covs.shape[0]
    all_labels = []
    for subject_ts in ts:
        subject_labels = []
        for start in range(0, subject_ts.shape[0] - window_length + 1, window_offset):
            window_ts = subject_ts[start : start + window_length]

            def flatten_upper_triangle(matrix):
                return matrix[np.triu_indices_from(matrix)]

            window_cov = np.cov(window_ts, rowvar=False)
            window_cov_vector = flatten_upper_triangle(window_cov)
            distances = [
                euclidean(window_cov_vector, flatten_upper_triangle(covs[state]))
                for state in range(n_states)
            ]
            subject_labels.append(np.argmin(distances))
        all_labels.append(np.array(subject_labels))
    return all_labels


def swc_average_log_likelihood(ts, alpha, means, covs, window_length, window_offset):
    total_log_likelihood = 0
    total_window_number = 0
    for subject_ts, subject_alpha in zip(ts, alpha):
        for start, label in zip(
            range(0, subject_ts.shape[0] - window_length + 1, window_offset),
            subject_alpha,
        ):
            window_ts = subject_ts[start : start + window_length]
            total_log_likelihood += estimate_gaussian_log_likelihood(
                window_ts, means[label], covs[label], average=True
            )
            total_window_number += 1
    return total_log_likelihood / total_window_number


# osl_dynamics/inference/metrics.py
def riemannian_distance(M1, M2, threshold=1e-3):
    eigvals = eigvalsh(M1, M2, driver="gv")
    if np.any(eigvals < 0):
        eigvals = np.maximum(eigvals, threshold)
    d = np.sqrt((np.log(eigvals) ** 2).sum())
    return d


def regularisation(matrices, eps=1e-6):
    if matrices.ndim == 2:
        N = len(matrices)
    elif matrices.ndim == 3:
        _, N, _ = matrices.shape
    else:
        return ValueError("Matrices dimension is incorrect!")
    return matrices + np.eye(N) * eps


def twopair_riemannian_distance(
    matrices_1, matrices_2, eps_value=[0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5], threshold=1e-3
):
    M1, N1, P1 = matrices_1.shape
    M2, N2, P2 = matrices_2.shape
    riemannian_distances = np.zeros([M1, M2])
    for eps in eps_value:
        try:
            for i in range(M1):
                for j in range(M2):
                    riemannian_distances[i][j] = riemannian_distance(
                        regularisation(matrices_1[i], eps),
                        regularisation(matrices_2[j], eps),
                        threshold=threshold,
                    )
            return riemannian_distances
        except np.linalg.LinAlgError:
            warnings.warn(f"Calculation of Riemannian distance failed for eps={eps}")
    raise ValueError("Riemannian distance failed for all eps values!")


# osl_dynamics/inference/modes.py
def hungarian_pair(matrix, distance=False):
    if distance:
        row_indices, col_indices = linear_sum_assignment(matrix)
    else:
        row_indices, col_indices = linear_sum_assignment(np.max(matrix) - matrix)
    indices = {"row": row_indices.tolist(), "col": col_indices.tolist()}
    matrix_reordered = matrix[row_indices, :]
    matrix_reordered = matrix_reordered[:, col_indices]
    return indices, matrix_reordered


# osl_dynamics/config_api/batch.py: BatchAnalysis._reproducibility_analysis
def reproducibility_analysis(cov_1, cov_2):
    riem = twopair_riemannian_distance(cov_1, cov_2)
    indice, riem_reorder = hungarian_pair(riem, distance=True)
    return np.mean(np.diagonal(riem_reorder))


# osl_dynamics/config_api/batch.py: IndexParser._make_list (job order)
def job_order(models, n_states, mode_index):
    from itertools import product

    return list(product(models, n_states, mode_index))
