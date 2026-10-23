"""Train one model variant on a simulation and score it against the ground truth.

Usage: python run.py <sim_dir> <variant> <out_dir> [--seed 0] [--n_modes 4] [--n_epochs 30]
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score  # noqa: E402

DYNEMO_BASE = dict(
    sequence_length=100,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=5,
    batch_size=32,
    learning_rate=1e-3,
)

# variant name -> (model family, config overrides)
VARIANTS = {
    "dynemo_mean": ("dynemo", dict(learn_means=True)),
    "dynemo_nomean": ("dynemo", dict(learn_means=False)),
    "dynemo_mean_cd10": ("dynemo", dict(learn_means=True, coordinated_dropout=0.1)),
    "dynemo_mean_cd25": ("dynemo", dict(learn_means=True, coordinated_dropout=0.25)),
    "dynemo_mean_cd50": ("dynemo", dict(learn_means=True, coordinated_dropout=0.5)),
    # Diagonal private-noise floor Psi added to every C_t ("oracle" = true private
    # variance, "faN" = static factor analysis with N factors on the simulated data)
    "dynemo_mean_psi_learn": ("dynemo", dict(learn_means=True, private_variance=0.3,
                                             learn_private_variance=True)),
    "dynemo_mean_psi_oracle": ("dynemo", dict(learn_means=True, private_variance="oracle")),
    "dynemo_mean_psi_fa4": ("dynemo", dict(learn_means=True, private_variance="fa4")),
    "dynemo_mean_psi_fa8": ("dynemo", dict(learn_means=True, private_variance="fa8")),
    "dynemo_mean_psi_fa12": ("dynemo", dict(learn_means=True, private_variance="fa12")),
    "dynemo_mean_cd50_psi_learn": ("dynemo", dict(learn_means=True, coordinated_dropout=0.5,
                                                  private_variance=0.3,
                                                  learn_private_variance=True)),
    "mdynemo_mean": ("mdynemo", dict(learn_means=True)),
    "mdynemo_mean_cd10": ("mdynemo", dict(learn_means=True, coordinated_dropout=0.1)),
    "mdynemo_mean_cd25": ("mdynemo", dict(learn_means=True, coordinated_dropout=0.25)),
    "statedynemo_mean": ("statedynemo", dict(learn_means=True)),
    "hmm_mean": ("hmm", dict(learn_means=True)),
}


def resolve_private_variance(overrides, x, truth):
    """Replace "oracle"/"faN" private_variance settings with numbers."""
    pv = overrides.get("private_variance")
    if not isinstance(pv, str):
        return overrides
    if pv == "oracle":
        value = truth["private_var"]
    else:
        from sklearn.decomposition import FactorAnalysis

        fa = FactorAnalysis(n_components=int(pv[2:]), random_state=0)
        value = fa.fit(x.reshape(-1, x.shape[-1])[::10]).noise_variance_
    return {**overrides, "private_variance": value.astype(np.float32)}


def train_dynemo(data, n_channels, n_modes, n_epochs, overrides, n_init=5, batch_size=32):
    from osl_dynamics.models.dynemo import Config, Model

    config = Config(
        n_modes=n_modes,
        n_channels=n_channels,
        n_epochs=n_epochs,
        n_kl_annealing_epochs=max(1, n_epochs // 2),
        **{**DYNEMO_BASE, "batch_size": batch_size, **overrides},
    )
    model = Model(config)
    model.set_regularizers(data)
    init_history = model.random_subset_initialization(data, n_init=n_init, n_epochs=2, take=1)
    history = model.fit(data)
    return model, {"init_loss": init_history["loss"], "loss": history["loss"]}


def train_mdynemo(data, n_channels, n_modes, n_epochs, overrides, n_init=5, batch_size=32):
    from osl_dynamics.models.mdynemo import Config, Model

    base = {k: v for k, v in DYNEMO_BASE.items() if k != "learn_covariances"}
    config = Config(
        n_modes=n_modes,
        n_fc_modes=n_modes,
        n_channels=n_channels,
        n_epochs=n_epochs,
        n_kl_annealing_epochs=max(1, n_epochs // 2),
        learn_stds=True,
        learn_fcs=True,
        **{**base, "batch_size": batch_size, **overrides},
    )
    model = Model(config)
    model.set_regularizers(data)
    init_history = model.random_subset_initialization(data, n_init=n_init, n_epochs=2, take=1)
    history = model.fit(data)
    return model, {"init_loss": init_history["loss"], "loss": history["loss"]}


def train_statedynemo(data, n_channels, n_modes, n_epochs, overrides, n_init=5, batch_size=32):
    from osl_dynamics.models.state_dynemo import Config, Model

    base = {k: v for k, v in DYNEMO_BASE.items() if "alpha_temperature" not in k}
    config = Config(
        n_states=n_modes,
        n_channels=n_channels,
        n_epochs=n_epochs,
        n_kl_annealing_epochs=max(1, n_epochs // 2),
        **{**base, "batch_size": batch_size, **overrides},
    )
    model = Model(config)
    model.set_regularizers(data)
    init_history = model.random_subset_initialization(data, n_init=n_init, n_epochs=2, take=1)
    history = model.fit(data)
    return model, {"init_loss": init_history["loss"], "loss": history["loss"]}


def train_hmm(data, n_channels, n_modes, n_epochs, overrides, n_init=5, batch_size=64):
    from osl_dynamics.models.hmm import Config, Model

    config = Config(
        n_states=n_modes,
        n_channels=n_channels,
        sequence_length=400,
        learn_covariances=True,
        learn_trans_prob=True,
        batch_size=64,
        learning_rate=1e-3,
        n_epochs=n_epochs,
        **overrides,
    )
    model = Model(config)
    model.set_regularizers(data)
    init_history = model.random_subset_initialization(data, n_init=n_init, n_epochs=2, take=1)
    history = model.fit(data)
    return model, {"init_loss": init_history["loss"], "loss": history["loss"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_dir")
    parser.add_argument("variant", choices=list(VARIANTS))
    parser.add_argument("out_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_modes", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    import tensorflow as tf

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    tf.config.threading.set_intra_op_parallelism_threads(n_cpus)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    from osl_dynamics.data import Data

    os.makedirs(args.out_dir, exist_ok=True)
    x = np.load(f"{args.sim_dir}/data.npy")
    truth = dict(np.load(f"{args.sim_dir}/truth.npz"))
    data = Data(list(x), store_dir=f"{args.out_dir}/tmp")

    family, overrides = VARIANTS[args.variant]
    overrides = resolve_private_variance(overrides, x, truth)
    trainer = {"dynemo": train_dynemo, "mdynemo": train_mdynemo,
               "statedynemo": train_statedynemo, "hmm": train_hmm}[family]
    t0 = time.time()
    model, history = trainer(data, x.shape[-1], args.n_modes, args.n_epochs, overrides,
                             n_init=args.n_init,
                             **({"batch_size": args.batch_size} if args.batch_size else {}))
    train_minutes = (time.time() - t0) / 60

    with open(f"{args.out_dir}/history.pkl", "wb") as f:
        pickle.dump(history, f)

    if family == "mdynemo":
        # C_t = G_t F_t G_t with G_t = sum alpha stds (means share alpha), F_t = sum gamma fcs
        alpha, gamma = map(np.array, model.get_alpha(data))
        means, stds, fcs = model.get_means_stds_fcs()
        for name, arr in [("alpha", alpha), ("gamma", gamma), ("means", means),
                          ("stds", stds), ("fcs", fcs)]:
            np.save(f"{args.out_dir}/{name}.npy", arr.astype(np.float32))
        a_flat = alpha.reshape(-1, alpha.shape[-1])
        g_flat = gamma.reshape(-1, gamma.shape[-1])

        def moments(rows):
            G = np.einsum("tk,kij->tij", a_flat[rows], stds)
            F = np.einsum("tk,kij->tij", g_flat[rows], fcs)
            return a_flat[rows] @ means, G @ F @ G

        result = score(means, None, alpha, truth, moments=moments)
        result.update(variant=args.variant, sim=os.path.basename(os.path.normpath(args.sim_dir)),
                      seed=args.seed, n_modes=args.n_modes, n_epochs=args.n_epochs,
                      n_init=args.n_init, final_loss=float(history["loss"][-1]),
                      train_minutes=train_minutes)
        with open(f"{args.out_dir}/metrics.json", "w") as f:
            json.dump(result, f, indent=1)
        print(json.dumps(result))
        return

    alpha = np.array(model.get_alpha(data))
    means, covs = model.get_means_covariances()
    np.save(f"{args.out_dir}/alpha.npy", alpha.astype(np.float32))
    np.save(f"{args.out_dir}/means.npy", means)
    np.save(f"{args.out_dir}/covs.npy", covs)

    # The private-noise floor is part of every C_t (alpha sums to one)
    has_psi = getattr(model.config, "private_variance", None) is not None
    psi = model.get_private_variance() if has_psi else None
    if psi is not None:
        np.save(f"{args.out_dir}/private_var.npy", psi)
        covs = covs + np.diag(psi)

    result = score(means, covs, alpha, truth)
    if psi is not None:
        result.update(private_var_mean=float(psi.mean()),
                      private_var_true_mean=float(truth["private_var"].mean()))
    result.update(variant=args.variant, sim=os.path.basename(os.path.normpath(args.sim_dir)),
                  seed=args.seed, n_modes=args.n_modes, n_epochs=args.n_epochs, n_init=args.n_init,
                  final_loss=float(history["loss"][-1]), train_minutes=train_minutes)
    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
