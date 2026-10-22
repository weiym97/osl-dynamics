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
    "dynemo_mean_cd25": ("dynemo", dict(learn_means=True, coordinated_dropout=0.25)),
    "dynemo_mean_cd50": ("dynemo", dict(learn_means=True, coordinated_dropout=0.5)),
    # Diagonal private-noise floor Psi added to every C_t ("oracle" = true private
    # variance, "faN" = static factor analysis with N factors on the simulated data)
    "dynemo_mean_psi_learn": ("dynemo", dict(learn_means=True, private_variance=0.3,
                                             learn_private_variance=True)),
    "dynemo_mean_psi_oracle": ("dynemo", dict(learn_means=True, private_variance="oracle")),
    "dynemo_mean_psi_fa8": ("dynemo", dict(learn_means=True, private_variance="fa8")),
    "dynemo_mean_cd50_psi_learn": ("dynemo", dict(learn_means=True, coordinated_dropout=0.5,
                                                  private_variance=0.3,
                                                  learn_private_variance=True)),
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


def train_dynemo(data, n_channels, n_modes, n_epochs, overrides):
    from osl_dynamics.models.dynemo import Config, Model

    config = Config(
        n_modes=n_modes,
        n_channels=n_channels,
        n_epochs=n_epochs,
        n_kl_annealing_epochs=max(1, n_epochs // 2),
        **{**DYNEMO_BASE, **overrides},
    )
    model = Model(config)
    model.set_regularizers(data)
    init_history = model.random_subset_initialization(data, n_init=5, n_epochs=2, take=1)
    history = model.fit(data)
    return model, {"init_loss": init_history["loss"], "loss": history["loss"]}


def train_hmm(data, n_channels, n_modes, n_epochs, overrides):
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
    init_history = model.random_subset_initialization(data, n_init=5, n_epochs=2, take=1)
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
    trainer = {"dynemo": train_dynemo, "hmm": train_hmm}[family]
    t0 = time.time()
    model, history = trainer(data, x.shape[-1], args.n_modes, args.n_epochs, overrides)
    train_minutes = (time.time() - t0) / 60

    alpha = np.array(model.get_alpha(data))
    means, covs = model.get_means_covariances()
    np.save(f"{args.out_dir}/alpha.npy", alpha.astype(np.float32))
    np.save(f"{args.out_dir}/means.npy", means)
    np.save(f"{args.out_dir}/covs.npy", covs)
    with open(f"{args.out_dir}/history.pkl", "wb") as f:
        pickle.dump(history, f)

    # The private-noise floor is part of every C_t (alpha sums to one)
    psi = model.get_private_variance() if hasattr(model, "get_private_variance") else None
    if psi is not None:
        np.save(f"{args.out_dir}/private_var.npy", psi)
        covs = covs + np.diag(psi)

    result = score(means, covs, alpha, truth)
    if psi is not None:
        result.update(private_var_mean=float(psi.mean()),
                      private_var_true_mean=float(truth["private_var"].mean()))
    result.update(variant=args.variant, sim=os.path.basename(os.path.normpath(args.sim_dir)),
                  seed=args.seed, n_modes=args.n_modes, n_epochs=args.n_epochs,
                  final_loss=float(history["loss"][-1]), train_minutes=train_minutes)
    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
