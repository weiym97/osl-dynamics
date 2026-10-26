"""Train DyNeMo on a subjects x channels subset of a soft-mixing simulation and score
mode recovery against the ground truth.

Usage: python run.py <out_dir> [--sim dynemo_fair] [--regime fold] [--fold 1] [--init rsi]
                     [--seed 0] [--n_modes 6] [--n_epochs 200]

Regimes (rows/columns from the BCV partition of the paper runs):
- fold: train subjects, Y channels (what Y_train sees in a BCV fold)
- subj: all subjects, Y channels
- chan: train subjects, all channels
- full: all subjects, all channels (what the repeat runs see)

Initialisations of the mode covariances (random subset initialisation runs on top of each):
- rsi: default (identity plus noise), as in the paper runs
- truth: the true covariances
- hmm: covariances of a K-state HMM trained on the same data
- als: soft unmixing of short-window second moments, S_w ~ sum_k a_wk C_k with a_w on the
  simplex (alternating least squares / NNLS). Unlike the HMM it can find a mode that never
  dominates.
"""

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = "/gpfs3/well/win-fmrib-analysis/users/uap971/osl-dynamics"

# Same as results_final/simulation/dynemo_fair/config_simulation.yaml
DYNEMO_BASE = dict(
    sequence_length=200,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    batch_size=16,
    learning_rate=0.01,
    lr_decay=0.1,
)


def load_subset(sim, regime, fold, n_samples):
    sim_dir = f"{REPO}/data/node_timeseries/simulation_final/{sim}"
    files = sorted(glob.glob(f"{sim_dir}/*.txt"))
    truth_files = sorted(glob.glob(f"{sim_dir}/truth/*_mode_time_course.npy"))
    idx = json.load(open(f"{REPO}/results_final/simulation/{sim}/bcv_partition/fold_indices_{fold}.json"))
    rows = idx["row_train"] if regime in ["fold", "chan"] else list(range(len(files)))
    cols = idx["column_Y"] if regime in ["fold", "subj"] else list(range(np.loadtxt(files[0], max_rows=1).size))
    x = [np.loadtxt(files[i])[:n_samples, cols].astype(np.float32) for i in rows]
    alpha = [np.load(truth_files[i])[:n_samples] for i in rows]
    covs = np.load(f"{sim_dir}/truth/state_covariances.npy")[:, cols][:, :, cols]
    return x, alpha, covs


def hmm_covariances(data, n_channels, n_modes, n_epochs):
    from osl_dynamics.models.hmm import Config, Model

    config = Config(
        n_states=n_modes,
        n_channels=n_channels,
        sequence_length=200,
        learn_means=False,
        learn_covariances=True,
        learn_trans_prob=True,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=n_epochs,
    )
    model = Model(config)
    model.set_regularizers(data)
    model.random_state_time_course_initialization(data, n_epochs=2, n_init=3)
    model.fit(data)
    return model.get_covariances()


def als_covariances(x, n_modes, window, n_iter=100, n_restarts=5, seed=0):
    from scipy.optimize import nnls

    n_channels = x[0].shape[1]
    iu = np.triu_indices(n_channels)
    S = np.array([
        (xs[i:i + window].T @ xs[i:i + window] / window)[iu]
        for xs in x for i in range(0, len(xs) - window + 1, window)
    ])
    # Weight of the sum-to-one row in the NNLS step
    w = 10 * np.linalg.norm(S, axis=1).mean()

    def to_matrices(D):
        C = np.zeros((n_modes, n_channels, n_channels))
        C[:, iu[0], iu[1]] = D
        C = C + np.triu(C, 1).transpose(0, 2, 1)
        # Project to PSD with a floor, so the DyNeMo Cholesky start is well conditioned
        e, v = np.linalg.eigh(C)
        e = np.maximum(e, 1e-2 * e.mean(-1, keepdims=True))
        return np.einsum("kij,kj,klj->kil", v, e, v)

    rng = np.random.default_rng(seed)
    best_err, best_C = np.inf, None
    for _ in range(n_restarts):
        A = rng.dirichlet(np.ones(n_modes), size=len(S))
        for _ in range(n_iter):
            C = to_matrices(np.linalg.lstsq(A, S, rcond=None)[0])
            D = C[:, iu[0], iu[1]]
            M = np.r_[D.T, w * np.ones((1, n_modes))]
            A = np.array([nnls(M, np.r_[s, w])[0] for s in S])
            A /= A.sum(1, keepdims=True)
        err = np.linalg.norm(S - A @ D)
        if err < best_err:
            best_err, best_C = err, C
    return best_C.astype(np.float32)


def nfindr_covariances(x, n_modes, window, n_restarts=10, seed=0):
    """Endmember extraction, as in hyperspectral unmixing.

    Window second moments are linear in the window-mean alpha, vec(S_w) = sum_k a_wk vec(C_k),
    so in the top n_modes-1 principal components the windows fill a simplex whose vertices are
    the mode covariances. N-FINDR picks the n_modes windows spanning the largest simplex; the
    covariances are then a least-squares fit given the simplex (NNLS) abundances.
    """
    from scipy.optimize import nnls

    n_channels = x[0].shape[1]
    iu = np.triu_indices(n_channels)
    S = np.array([
        (xs[i:i + window].T @ xs[i:i + window] / window)[iu]
        for xs in x for i in range(0, len(xs) - window + 1, window)
    ])
    mu = S.mean(0)
    Y = (S - mu) @ np.linalg.svd(S - mu, full_matrices=False)[2][:n_modes - 1].T

    def volume(idx):
        return abs(np.linalg.det(np.c_[np.ones(n_modes), Y[idx]]))

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        idx = rng.choice(len(Y), n_modes, replace=False)
        improved = True
        while improved:
            improved = False
            for j in range(n_modes):
                vols = []
                for w in range(len(Y)):
                    trial = idx.copy()
                    trial[j] = w
                    vols.append(volume(trial))
                if max(vols) > volume(idx) * (1 + 1e-9):
                    idx[j] = int(np.argmax(vols))
                    improved = True
        if best is None or volume(idx) > volume(best):
            best = idx.copy()

    M = np.r_[Y[best].T, 100 * np.ones((1, n_modes))]
    A = np.array([nnls(M, np.r_[y, 100])[0] for y in Y])
    A /= A.sum(1, keepdims=True)
    D = np.linalg.lstsq(A, S, rcond=None)[0]
    C = np.zeros((n_modes, n_channels, n_channels))
    C[:, iu[0], iu[1]] = D
    C = C + np.triu(C, 1).transpose(0, 2, 1)
    return psd_floor(C).astype(np.float32)


def score(alpha, covs, alpha_true, covs_true):
    """Recovery metrics, one value per TRUE mode (Hungarian matching on alpha correlation)."""
    k_true = alpha_true.shape[1]
    corr = np.nan_to_num(np.corrcoef(alpha_true.T, alpha.T)[:k_true, k_true:])
    r, c = linear_sum_assignment(-corr)
    alpha_r = np.full(k_true, np.nan)
    alpha_r[r] = corr[r, c]

    # R^2 of each true alpha from a linear readout of all fitted alphas
    X = np.c_[alpha, np.ones(len(alpha))]
    res = alpha_true - X @ np.linalg.lstsq(X, alpha_true, rcond=None)[0]
    alpha_r2 = 1 - res.var(0) / alpha_true.var(0)

    # Mode-specific covariance (minus the mean over modes), off-diagonal, matched as above
    iu = np.triu_indices(covs_true.shape[1], 1)
    spec_true = (covs_true - covs_true.mean(0))[:, iu[0], iu[1]]
    spec = (covs - covs.mean(0))[:, iu[0], iu[1]]
    cov_r = np.full(k_true, np.nan)
    for i, j in zip(r, c):
        cov_r[i] = np.corrcoef(spec_true[i], spec[j])[0, 1]

    # Time-varying covariance C(t) = sum_k alpha_k C_k, relative to its dynamic part
    t = np.random.default_rng(0).choice(len(alpha), min(2000, len(alpha)), replace=False)
    ct_true = np.einsum("tk,kij->tij", alpha_true[t], covs_true)
    ct = np.einsum("tk,kij->tij", alpha[t], covs)
    ct_err = np.linalg.norm(ct - ct_true) / np.linalg.norm(ct_true - ct_true.mean(0))

    usage = alpha.mean(0)
    return {
        "alpha_r": alpha_r.tolist(),
        "alpha_r2": alpha_r2.tolist(),
        "cov_r": cov_r.tolist(),
        "ct_rel_err": float(ct_err),
        "n_dead": int(((usage < 0.02) | (alpha.std(0) < 0.01)).sum()),
        "alpha_mean": usage.tolist(),
    }


def make_tracker(data, alpha_true, covs_true, every):
    """Keras callback that scores recovery every `every` epochs of the main fit.

    `train` sets the callback's `osld_model` before training starts.
    """
    import tensorflow as tf

    class Tracker(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.records = []

        def record(self, epoch, loss):
            alpha = np.concatenate(self.osld_model.get_alpha(data))
            s = score(alpha, self.osld_model.get_covariances(), alpha_true, covs_true)
            self.records.append({"epoch": epoch, "loss": loss, **{k: s[k] for k in ["alpha_r", "cov_r", "alpha_mean"]}})

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % every == 0:
                self.record(epoch + 1, float(logs["loss"]))

    return Tracker()


def make_beta_schedule(model, beta0, n_epochs):
    """KL weight beta0 at the start, decaying linearly to 1 over n_epochs (use without KL annealing).

    beta > 1 keeps alpha close to the (initially uninformative) prior, so the modes separate
    gradually, as in deterministic annealing.
    """
    import tensorflow as tf

    class BetaSchedule(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            beta = 1 + (beta0 - 1) * max(0.0, 1 - epoch / n_epochs)
            model.model.get_layer("kl_loss").annealing_factor.assign(beta)

    return BetaSchedule()


def initialise(model, data, args):
    """Random initialisation of the RNNs (and covariances, unless they were given)."""
    if args.n_init == 0:
        return {"loss": [np.nan]}
    if args.init_select == "loss":
        return model.random_subset_initialization(
            data, n_init=args.n_init, n_epochs=args.init_epochs, take=1
        )

    # Keep the initialisation whose mean alpha is most balanced (highest entropy) instead of
    # the lowest loss: after a couple of epochs the lowest loss tends to be one where a few
    # modes have taken over
    best = (-np.inf, None, None)
    for _ in range(args.n_init):
        model.reset()
        history = model.fit(data, epochs=args.init_epochs)
        usage = np.concatenate(model.get_alpha(data)).mean(0)
        entropy = -(usage * np.log(usage + 1e-12)).sum()
        if entropy > best[0]:
            best = (entropy, model.get_weights(), history)
    model.set_weights(best[1])
    model.reset_kl_annealing_factor()
    return best[2]


def sim_training_config(sim):
    """DyNeMo config_kwargs, init_kwargs and session length of the original results folder."""
    import yaml

    with open(f"{REPO}/results_final/simulation/{sim}/config_simulation.yaml") as f:
        config = yaml.safe_load(f)
    dynemo = config["model"]["dynemo"]
    config_kwargs = {k: v for k, v in dynemo["config_kwargs"].items() if k != "n_channels"}
    n_samples = config["load_data"]["prepare"]["select"]["timepoints"][1]
    return config_kwargs, dynemo.get("init_kwargs", {}), n_samples


def resolve_training_config(args):
    """DYNEMO_BASE (the dynemo_fair settings), or the simulation's own settings with
    --sim_config, then any training option given on the command line."""
    base = {**DYNEMO_BASE, "n_epochs": 200, "n_init": 10, "init_epochs": 2}
    n_kl = None
    if args.sim_config:
        config_kwargs, init_kwargs, args.n_samples = sim_training_config(args.sim)
        base.update(config_kwargs)
        base["n_init"] = init_kwargs.get("n_init", 10)
        base["init_epochs"] = init_kwargs.get("n_epochs", 2)
        n_kl = base.pop("n_kl_annealing_epochs", None)
    for k in ["n_epochs", "learning_rate", "lr_decay", "batch_size", "n_init", "init_epochs"]:
        if getattr(args, k) is not None:
            base[k] = getattr(args, k)
        setattr(args, k, base[k])
    if args.n_kl_annealing_epochs is None:
        args.n_kl_annealing_epochs = n_kl or args.n_epochs // 2
    args.train_config = {
        k: v for k, v in base.items() if k not in ["n_epochs", "n_init", "init_epochs"]
    }


def train(args, data, n_channels, initial_covariances, callbacks=None, n_modes=None):
    from osl_dynamics.models.dynemo import Config, Model

    config = Config(
        n_modes=n_modes or args.n_modes,
        n_channels=n_channels,
        n_epochs=args.n_epochs,
        n_kl_annealing_epochs=args.n_kl_annealing_epochs,
        initial_covariances=initial_covariances,
        **{
            **args.train_config,
            "do_kl_annealing": not args.no_kl_annealing,
            "learn_alpha_temperature": not args.fixed_temperature,
        },
    )
    model = Model(config)
    model.set_regularizers(data)
    init_history = initialise(model, data, args)
    for c in callbacks or []:
        c.osld_model = model
        c.record(0, float(init_history["loss"][-1]))
    callbacks = list(callbacks or [])
    if args.kl_beta0 != 1:
        callbacks.append(make_beta_schedule(model, args.kl_beta0, args.kl_beta_epochs))
    history = model.fit(data, callbacks=callbacks)
    ll_loss, kl_loss = model.losses(model.make_dataset(data, concatenate=True))
    return model, {
        "alpha": np.concatenate(model.get_alpha(data)),
        "covs": model.get_covariances(),
        "free_energy": float(ll_loss + kl_loss),
        "ll_loss": float(ll_loss),
        "kl_loss": float(kl_loss),
        "final_loss": float(history["loss"][-1]),
        "init_loss": float(init_history["loss"][-1]),
        "history": {"init": init_history, "fit": history},
    }


def psd_floor(C, rel=1e-2):
    e, v = np.linalg.eigh(C)
    e = np.maximum(e, rel * e.mean(-1, keepdims=True))
    return np.einsum("...ij,...j,...kj->...ik", v, e, v)


def split_scores(x, alpha, covs):
    """Alpha-weighted mean Mahalanobis excess x'C_t^-1 x / p - 1 for each mode (0 if well fitted)."""
    n_channels = x.shape[1]
    excess = np.empty(len(x))
    for i in range(0, len(x), 4000):
        Ct = np.einsum("tk,kij->tij", alpha[i:i + 4000], covs)
        z = np.linalg.solve(np.linalg.cholesky(Ct), x[i:i + 4000, :, None])[..., 0]
        excess[i:i + 4000] = (z ** 2).sum(1) / n_channels - 1
    return (alpha * excess[:, None]).sum(0) / alpha.sum(0)


def split_mode(x_sessions, alpha_sessions, covs, k, window=20, top=0.2, seed=0):
    """Split mode k in two with 2-means on its local covariance in windows where it is strong.

    In window w the local covariance of mode k is (S_w - sum_{j != k} a_wj C_j) / a_wk,
    with S_w the window's second moment and a_w the window-mean alpha.
    """
    from sklearn.cluster import KMeans

    S, A = [], []
    for xs, a in zip(x_sessions, alpha_sessions):
        for i in range(0, len(xs) - window + 1, window):
            S.append(xs[i:i + window].T @ xs[i:i + window] / window)
            A.append(a[i:i + window].mean(0))
    S, A = np.array(S), np.array(A)
    keep = A[:, k] >= np.quantile(A[:, k], 1 - top)
    others = np.delete(np.arange(len(covs)), k)
    local = (S[keep] - np.einsum("wj,jab->wab", A[keep][:, others], covs[others])) / A[keep, k, None, None]
    iu = np.triu_indices(covs.shape[1])
    labels = KMeans(2, n_init=10, random_state=seed).fit_predict(local[:, iu[0], iu[1]])
    weights = A[keep, k]
    halves = [
        np.einsum("w,wab->ab", weights[labels == c], local[labels == c]) / weights[labels == c].sum()
        for c in range(2)
    ]
    return psd_floor(np.array(halves))


def smem_candidates(x_sessions, fit, n_merge=2, n_split=2):
    """Covariance sets with one merge + one split (or one split into a dead mode)."""
    alpha, covs = fit["alpha"], fit["covs"]
    lengths = [len(xs) for xs in x_sessions]
    alpha_sessions = np.split(alpha, np.cumsum(lengths)[:-1])
    usage = alpha.mean(0)
    dead = np.where((usage < 0.02) | (alpha.std(0) < 0.01))[0]
    alive = np.setdiff1d(np.arange(len(covs)), dead)

    scores = split_scores(np.concatenate(x_sessions), alpha, covs)
    scores[dead] = -np.inf
    split_order = np.argsort(scores)[::-1]

    candidates = []
    if len(dead):
        for k in split_order[:n_split]:
            new = covs.copy()
            new[[k, dead[0]]] = split_mode(x_sessions, alpha_sessions, covs, k)
            candidates.append({"merge": None, "split": int(k), "into": int(dead[0]), "covs": new})
        return candidates, scores

    corr = np.corrcoef(alpha[:, alive].T)
    pairs = sorted(
        [(corr[i, j], alive[i], alive[j]) for i in range(len(alive)) for j in range(i + 1, len(alive))],
        reverse=True,
    )[:n_merge]
    for _, a, b in pairs:
        merged = (usage[a] * covs[a] + usage[b] * covs[b]) / (usage[a] + usage[b])
        for k in [k for k in split_order if k not in (a, b)][:n_split]:
            new = covs.copy()
            new[a] = merged
            new[[k, b]] = split_mode(x_sessions, alpha_sessions, covs, k)
            candidates.append({"merge": [int(a), int(b)], "split": int(k), "into": int(b), "covs": new})
    return candidates, scores


def reset_candidates(x_sessions, fit, seed=0):
    """One candidate per mode: that mode's covariance reset to the static covariance (jittered).

    With the other modes kept where they converged, the freed mode can pick up whatever
    structure is left unexplained, e.g. a weak mode that the first fit merged or never found.
    """
    covs = fit["covs"]
    X = np.concatenate(x_sessions)
    static = X.T @ X / len(X)
    rng = np.random.default_rng(seed)
    candidates = []
    for k in range(len(covs)):
        new = covs.copy()
        jitter = np.diag(np.diag(static) * rng.normal(0, 0.1, len(static)))
        new[k] = psd_floor(static + jitter)
        candidates.append({"reset": k, "covs": new})
    return candidates


def split_candidates(fit, seed=0, eps=0.2):
    """Drop the least-used mode and split another mode k into C_k + D and C_k - D.

    D is a random symmetric perturbation with norm eps * ||C_k||; training decides the split
    direction. One candidate per k.
    """
    covs, usage = fit["covs"], fit["alpha"].mean(0)
    j = int(np.argmin(usage))
    rng = np.random.default_rng(seed)
    candidates = []
    for k in range(len(covs)):
        if k == j:
            continue
        D = rng.normal(size=covs[k].shape)
        D = (D + D.T) / 2
        D *= eps * np.linalg.norm(covs[k]) / np.linalg.norm(D)
        new = covs.copy()
        new[k] = psd_floor(covs[k] + D)
        new[j] = psd_floor(covs[k] - D)
        candidates.append({"split": k, "into": j, "covs": new})
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--sim", default="dynemo_fair")
    parser.add_argument("--regime", default="fold", choices=["fold", "subj", "chan", "full"])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--init", default="rsi", choices=["rsi", "truth", "hmm", "als", "nfindr", "nfindr_allch"])
    parser.add_argument("--init_select", default="loss", choices=["loss", "entropy"])
    parser.add_argument("--als_window", type=int, default=20)
    parser.add_argument("--nfindr_expand", type=float, default=1.0)
    parser.add_argument("--truth_scale", action="store_true")
    parser.add_argument("--revive", action="store_true")
    parser.add_argument("--sim_config", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_kl_annealing", action="store_true")
    parser.add_argument("--kl_beta0", type=float, default=1.0)
    parser.add_argument("--kl_beta_epochs", type=int, default=100)
    parser.add_argument("--fixed_temperature", action="store_true")
    parser.add_argument("--init_epochs", type=int, default=None)
    parser.add_argument("--track_every", type=int, default=0)
    parser.add_argument("--smem_rounds", type=int, default=0)
    parser.add_argument("--reset_rounds", type=int, default=0)
    parser.add_argument("--split_rounds", type=int, default=0)
    parser.add_argument("--prune_from_plus1", action="store_true")
    parser.add_argument("--prune_n", type=int, default=0, help="only try the n least-used removals (0 = all)")
    parser.add_argument("--split_eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_modes", type=int, default=6)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--n_kl_annealing_epochs", type=int, default=None)
    parser.add_argument("--n_init", type=int, default=None)
    parser.add_argument("--hmm_epochs", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=400)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    resolve_training_config(args)

    import tensorflow as tf

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    from osl_dynamics.data import Data

    x, alpha_true, covs_true = load_subset(args.sim, args.regime, args.fold, args.n_samples)
    alpha_true = np.concatenate(alpha_true)
    n_channels = x[0].shape[1]
    data = Data(x, store_dir=f"{args.out_dir}/tmp")
    print(f"{args.sim} {args.regime} fold {args.fold}: {len(x)} sessions x {x[0].shape}")

    t0 = time.time()

    def make_init(n_modes):
        """Initial covariances for n_modes modes with args.init (None = default initialiser)."""
        if args.init == "truth":
            if n_modes != len(covs_true):
                raise ValueError("The truth start needs n_modes equal to the true number of modes.")
            C = covs_true.astype(np.float32)
            if args.truth_scale:
                # Match the data's overall variance (e.g. after HRF filtering), shape unchanged
                static = sum(xs.T @ xs for xs in x) / sum(len(xs) for xs in x)
                expected = np.einsum("k,kij->ij", alpha_true.mean(0), covs_true)
                C *= np.float32(np.trace(static) / np.trace(expected))
            return C
        if args.init == "hmm":
            return hmm_covariances(data, n_channels, n_modes, args.hmm_epochs)
        if args.init == "als":
            return als_covariances(x, n_modes, args.als_window, seed=args.seed)
        if args.init in ["nfindr", "nfindr_allch"]:
            from osl_dynamics.inference.initializers import nfindr_covariances as library_nfindr

            if args.init == "nfindr":
                C = library_nfindr(x, n_modes, window_length=args.als_window, random_seed=args.seed)
            else:
                # N-FINDR on all channels of the same (training) sessions, then keep the block of
                # the channels this model is trained on. Only training sessions are used.
                if args.regime not in ["fold", "subj"]:
                    raise ValueError("nfindr_allch is for the fold/subj regimes (a subset of channels).")
                x_all, _, _ = load_subset(args.sim, "chan" if args.regime == "fold" else "full", args.fold, args.n_samples)
                C = library_nfindr(x_all, n_modes, window_length=args.als_window, random_seed=args.seed)
                del x_all
                cols = json.load(open(f"{REPO}/results_final/simulation/{args.sim}/bcv_partition/fold_indices_{args.fold}.json"))["column_Y"]
                C = C[:, cols][:, :, cols]
            if args.nfindr_expand != 1:
                # Push each mode away from the mean of the modes: the vertices N-FINDR finds are
                # the purest windows, which are still mixtures when no window is pure
                C = psd_floor(C.mean(0) + args.nfindr_expand * (C - C.mean(0)))
            return C.astype(np.float32)
        return None

    initial_covariances = make_init(args.n_modes)

    callbacks = None
    if args.track_every:
        callbacks = [make_tracker(data, alpha_true, covs_true, args.track_every)]
    model, fit = train(args, data, n_channels, initial_covariances, callbacks)

    revive_log = None
    if args.revive:
        # Reset every dead mode to the static covariance and retrain once; keep if FE is lower
        dead = np.where((fit["alpha"].mean(0) < 0.02) | (fit["alpha"].std(0) < 0.01))[0]
        if len(dead):
            del model
            tf.keras.backend.clear_session()
            static = sum(xs.T @ xs for xs in x) / sum(len(xs) for xs in x)
            new = fit["covs"].copy()
            rng = np.random.default_rng(args.seed)
            for k in dead:
                new[k] = psd_floor(static + np.diag(np.diag(static) * rng.normal(0, 0.1, n_channels)))
            model, rfit = train(args, data, n_channels, new.astype(np.float32))
            s = score(rfit["alpha"], rfit["covs"], alpha_true, covs_true)
            revive_log = {"dead": dead.tolist(), "before": fit["free_energy"],
                          "after": rfit["free_energy"], "alpha_r_after": s["alpha_r"]}
            if rfit["free_energy"] < fit["free_energy"]:
                fit = rfit

    prune_log = None
    if args.prune_from_plus1:
        # Grow then prune: fit K+1 modes, then start K-mode fits from the K+1 covariances with one
        # mode removed. Every removal is trained; the lowest FE (including the direct K fit) wins.
        # Two cheap ranking signals are logged to see which would have picked the winner.
        from scipy.optimize import nnls

        del model
        tf.keras.backend.clear_session()
        K = args.n_modes
        model, pfit = train(args, data, n_channels, make_init(K + 1), n_modes=K + 1)
        del model
        tf.keras.backend.clear_session()
        usage = pfit["alpha"].mean(0)
        iu = np.triu_indices(n_channels)
        w = args.als_window
        S = np.array([(xs[i:i + w].T @ xs[i:i + w] / w)[iu] for xs in x for i in range(0, len(xs) - w + 1, w)])
        weight = 10 * np.linalg.norm(S, axis=1).mean()
        prune_log = {"plus1_free_energy": pfit["free_energy"], "plus1_usage": usage.tolist(),
                     "plus1_alpha_r": score(pfit["alpha"], pfit["covs"], alpha_true, covs_true)["alpha_r"],
                     "candidates": []}
        best = fit
        removals = np.argsort(usage)[: args.prune_n] if args.prune_n else range(K + 1)
        for j in removals:
            j = int(j)
            covs = np.delete(pfit["covs"], j, axis=0)
            D = covs[:, iu[0], iu[1]]
            M = np.r_[D.T, weight * np.ones((1, K))]
            A = np.array([nnls(M, np.r_[s, weight])[0] for s in S])
            residual = float(np.linalg.norm(S - A @ D))
            model, cfit = train(args, data, n_channels, covs.astype(np.float32))
            del model
            tf.keras.backend.clear_session()
            s = score(cfit["alpha"], cfit["covs"], alpha_true, covs_true)
            prune_log["candidates"].append({"removed": j, "usage": float(usage[j]), "residual": residual,
                                            "free_energy": cfit["free_energy"], "alpha_r": s["alpha_r"]})
            print(f"Prune: removed {j} (usage {usage[j]:.3f}, residual {residual:.1f}) -> FE {cfit['free_energy']:.3f}"
                  f" (direct K fit {fit['free_energy']:.3f}), alpha r {np.round(s['alpha_r'], 2)}")
            if cfit["free_energy"] < best["free_energy"]:
                best = cfit
        prune_log["direct_free_energy"] = fit["free_energy"]
        fit = best
        model = None

    # Search moves (split-and-merge, or mode resets): propose covariance sets, retrain from each,
    # keep the lowest free energy. The truth is only used to score candidates for the log.
    def logged(f, **extra):
        s = score(f["alpha"], f["covs"], alpha_true, covs_true)
        return {**extra, "free_energy": f["free_energy"], "alpha_r": s["alpha_r"], "cov_r": s["cov_r"]}

    smem_log = [logged(fit, round=0)]
    for r in range(1, max(args.smem_rounds, args.reset_rounds, args.split_rounds) + 1):
        del model
        tf.keras.backend.clear_session()
        if args.split_rounds:
            candidates, extra = split_candidates(fit, seed=args.seed + r, eps=args.split_eps), {}
        elif args.reset_rounds:
            candidates, extra = reset_candidates(x, fit, seed=args.seed + r), {}
        else:
            candidates, scores = smem_candidates(x, fit)
            extra = {"split_scores": scores.tolist()}
        best_new = None
        for c in candidates:
            model, cfit = train(args, data, n_channels, c["covs"].astype(np.float32))
            del model
            tf.keras.backend.clear_session()
            move = {k: v for k, v in c.items() if k != "covs"}
            entry = logged(cfit, round=r, **move, **extra, current=fit["free_energy"])
            smem_log.append(entry)
            print(f"Search round {r}: {move} -> FE {cfit['free_energy']:.3f}"
                  f" (current {fit['free_energy']:.3f}), alpha r {np.round(entry['alpha_r'], 2)}")
            if cfit["free_energy"] < (best_new or fit)["free_energy"]:
                best_new = cfit
        model = None
        if best_new is None:
            break
        fit = best_new

    metrics = {
        **vars(args),
        **score(fit["alpha"], fit["covs"], alpha_true, covs_true),
        **{k: fit[k] for k in ["free_energy", "ll_loss", "kl_loss", "final_loss", "init_loss"]},
        "smem_log": smem_log,
        "revive_log": revive_log,
        "prune_log": prune_log,
        "train_minutes": (time.time() - t0) / 60,
    }
    json.dump(metrics, open(f"{args.out_dir}/metrics.json", "w"), indent=1)
    np.save(f"{args.out_dir}/alpha.npy", fit["alpha"].astype(np.float32))
    np.save(f"{args.out_dir}/covs.npy", fit["covs"])
    pickle.dump(fit["history"], open(f"{args.out_dir}/history.pkl", "wb"))
    if args.track_every:
        json.dump(callbacks[0].records, open(f"{args.out_dir}/tracking.json", "w"), indent=1)
    data.delete_dir()
    print(json.dumps({k: metrics[k] for k in ["alpha_r", "cov_r", "free_energy", "n_dead"]}))


if __name__ == "__main__":
    main()


