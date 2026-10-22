"""Simulate data where DyNeMo with learned means is expected to copy private noise.

Generative model (per session):

    x_t = m_t + s_t + e_t
    m_t = sum_k alpha_tk mu_k                  (mean activity, shared)
    s_t ~ N(0, sum_k alpha_tk D_k)             (time-varying FC, white given alpha)
    e_t: private noise, independent across channels, AR(1) with coefficient phi
    alpha_t = softmax(beta * theta_t), theta: slow AR(1) per mode

With phi > 0 the private noise is temporally smooth (like fMRI), so a mode time
course inferred from x_t itself can follow it cheaply. phi = 0 is the white-noise
control. Data are standardised per channel with the population mean/std and the
ground-truth parameters are transformed accordingly, so the truth is exact.

Usage: python simulate.py <output_dir> [--phi 0.9] [--seed 0]
"""

import argparse
import os

import numpy as np


def ar1(rng, n_samples, n_series, phi, std):
    """Stationary AR(1) series with the given marginal std."""
    innov_std = std * np.sqrt(1 - phi ** 2)
    out = np.empty((n_samples, n_series))
    out[0] = rng.normal(0, std, n_series)
    for t in range(1, n_samples):
        out[t] = phi * out[t - 1] + rng.normal(0, innov_std, n_series)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--phi", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_sessions", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=1200)
    parser.add_argument("--n_channels", type=int, default=20)
    parser.add_argument("--n_modes", type=int, default=4)
    parser.add_argument("--theta_phi", type=float, default=0.97)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--mean_scale", type=float, default=1.0)
    parser.add_argument("--cov_scale", type=float, default=0.6)
    parser.add_argument("--fc_rank", type=int, default=3)
    parser.add_argument("--private_std", type=float, default=0.7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n, K = args.n_channels, args.n_modes

    # Observation model parameters (shared by all sessions)
    means = rng.normal(0, args.mean_scale, (K, n))
    covs = np.empty((K, n, n))
    for k in range(K):
        W = rng.normal(0, args.cov_scale / np.sqrt(args.fc_rank), (n, args.fc_rank))
        covs[k] = W @ W.T + 0.1 * args.cov_scale ** 2 * np.eye(n)
    private_var = np.full(n, args.private_std ** 2)

    alpha, m_true, s_true, e_true, x = [], [], [], [], []
    for _ in range(args.n_sessions):
        theta = ar1(rng, args.n_samples, K, args.theta_phi, 1.0)
        a = np.exp(args.beta * theta)
        a /= a.sum(1, keepdims=True)
        m = a @ means
        C = np.einsum("tk,kij->tij", a, covs)
        s = np.einsum("tij,tj->ti", np.linalg.cholesky(C), rng.normal(size=(args.n_samples, n)))
        e = ar1(rng, args.n_samples, n, args.phi, args.private_std)
        alpha.append(a)
        m_true.append(m)
        s_true.append(s)
        e_true.append(e)
        x.append(m + s + e)
    alpha, m_true, s_true, e_true, x = map(np.array, (alpha, m_true, s_true, e_true, x))

    # Standardise with population statistics and transform the truth exactly
    c = x.reshape(-1, n).mean(0)
    d = x.reshape(-1, n).std(0)
    x = (x - c) / d
    m_true = (m_true - c) / d
    s_true = s_true / d
    e_true = e_true / d
    means_std = (means - c) / d
    covs_std = covs / np.outer(d, d)
    private_var_std = private_var / d ** 2

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(f"{args.output_dir}/data.npy", x.astype(np.float32))
    np.savez(
        f"{args.output_dir}/truth.npz",
        alpha=alpha.astype(np.float32),
        means=means_std,
        covs=covs_std,
        private_var=private_var_std,
        m=m_true.astype(np.float32),
        s=s_true.astype(np.float32),
        e=e_true.astype(np.float32),
        phi=args.phi,
    )

    # Summary of the simulated variance budget (per channel, averaged)
    v_m = m_true.reshape(-1, n).var(0).mean()
    v_s = s_true.reshape(-1, n).var(0).mean()
    v_e = e_true.reshape(-1, n).var(0).mean()
    R = np.corrcoef(x.reshape(-1, n).T)
    off = np.abs(R[np.triu_indices(n, 1)])
    print(f"saved to {args.output_dir}: data {x.shape}, phi={args.phi}")
    print(f"variance shares: mean {v_m:.2f}, shared FC {v_s:.2f}, private {v_e:.2f}")
    print(f"alpha: mean max {alpha.max(-1).mean():.2f}, FO {np.round(alpha.reshape(-1, K).mean(0), 2)}")
    print(f"static |r|: median {np.median(off):.2f}, 95% {np.percentile(off, 95):.2f}")
    lag1 = np.mean([np.corrcoef(e_true[i, :-1, 0], e_true[i, 1:, 0])[0, 1] for i in range(10)])
    print(f"private noise lag-1 autocorrelation: {lag1:.2f}")


if __name__ == "__main__":
    main()
