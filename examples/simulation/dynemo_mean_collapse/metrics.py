"""Score a fitted observation model + inferred mode time courses against the simulation truth."""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

N_SUB = 50000


def r2(target, regressors):
    """Fraction of the variance of `target` explained by a linear fit on `regressors`."""
    X = np.column_stack([regressors, np.ones(len(regressors))])
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    tc = target - target.mean(0)
    if np.sum(tc ** 2) < 1e-12:
        return float("nan")
    return 1 - np.sum(resid ** 2) / np.sum(tc ** 2)


def gaussian_kl(m0, C0, m1, C1):
    """Mean over samples of KL( N(m0, C0) || N(m1, C1) ); batched over the first axis."""
    n = m0.shape[-1]
    L1 = np.linalg.cholesky(C1)
    C1inv = np.linalg.inv(C1)
    d = m1 - m0
    tr = np.einsum("tij,tji->t", C1inv, C0)
    maha = np.einsum("ti,tij,tj->t", d, C1inv, d)
    logdet1 = 2 * np.sum(np.log(np.diagonal(L1, axis1=1, axis2=2)), axis=1)
    logdet0 = np.linalg.slogdet(C0)[1]
    return 0.5 * (tr + maha - n + logdet1 - logdet0)


def score(means, covs, alpha, truth, seed=0, moments=None):
    """means (K, n), covs (K, n, n), alpha (n_sessions, T, K); truth from simulate.py.

    moments, optional: function of flat sample indices returning the model's per-sample
    mean (N, n) and covariance (N, n, n), for models whose covariance is not sum alpha C_k
    (e.g. M-DyNeMo). alpha is then only used for mode time-course recovery.
    """
    rng = np.random.default_rng(seed)
    K, n = means.shape
    T = alpha.shape[1]
    a = alpha.reshape(-1, K).astype(np.float64)
    a_true = truth["alpha"][:, :T].reshape(-1, truth["alpha"].shape[-1]).astype(np.float64)
    m_true = truth["m"][:, :T].reshape(-1, n).astype(np.float64)
    s_true = truth["s"][:, :T].reshape(-1, n).astype(np.float64)
    e_true = truth["e"][:, :T].reshape(-1, n).astype(np.float64)
    psi = np.diag(truth["private_var"])

    idx = rng.choice(len(a), size=min(N_SUB, len(a)), replace=False)
    if moments is None:
        def moments(rows):
            return a[rows] @ means, np.einsum("tk,kij->tij", a[rows], covs)
    m_model = moments(idx)[0]

    # 1. Collapse: model's time-averaged covariance vs the true conditional covariance
    if covs is not None:
        C_model = np.einsum("k,kij->ij", a.mean(0), covs)
    else:
        C_model = moments(idx[:20000])[1].mean(0)
    C_true = np.einsum("k,kij->ij", a_true.mean(0), truth["covs"]) + psi
    lam, V = eigh(C_model, C_true)
    cov_m = np.cov(m_model.T)
    mean_share = np.einsum("ij,jk,ki->i", V.T, cov_m, V)  # v'C_true v = 1

    # 2. What the model mean follows: true mean dynamics, shared white FC noise, private noise
    r2_mean = r2(m_model, m_true[idx])
    r2_shared = r2(m_model, s_true[idx])
    r2_private = r2(m_model, e_true[idx])
    recover_mean = r2(m_true[idx], m_model)

    # 3. Calibration to the truth: KL(true conditional given alpha_true || model given alpha)
    sub = idx[:5000]
    C_t_true = np.einsum("tk,kij->tij", a_true[sub], truth["covs"]) + psi
    m_t_model, C_t_model = moments(sub)
    kl = gaussian_kl(m_true[sub], C_t_true, m_t_model, C_t_model)

    # 4. Mode time-course recovery (matched by correlation)
    corr = np.corrcoef(a[idx].T, a_true[idx].T)[:K, K:]
    corr = np.nan_to_num(corr)
    rows, cols = linear_sum_assignment(-corr)
    alpha_corr = corr[rows, cols]

    return {
        "lam_min": float(lam[0]),
        "lam_sorted_5": [float(x) for x in lam[:5]],
        "n_lam_below_0.1": int(np.sum(lam < 0.1)),
        "n_lam_below_0.5": int(np.sum(lam < 0.5)),
        "mean_share_at_lam_min": float(mean_share[0]),
        "r2_model_mean_on_true_mean": float(r2_mean),
        "r2_model_mean_on_shared_fc_noise": float(r2_shared),
        "r2_model_mean_on_private_noise": float(r2_private),
        "r2_true_mean_recovered": float(recover_mean),
        "kl_truth_mean": float(kl.mean()),
        "kl_truth_median": float(np.median(kl)),
        "alpha_corr_matched": [float(x) for x in alpha_corr],
        "alpha_mean_max": float(a.max(1).mean()),
    }
