"""Run one BCV fold through the real pipeline (BCV.validate) with a chosen start for Y_train.

The config is the one IndexParser/BatchTrain would build from results_final/simulation/<sim>_nfindr/
(same folds, same DyNeMo settings). Only the Y_train training start differs:
- truth: the true covariances of the fold's Y channels (needs K = the true number of modes)
- nfindr_allch: N-FINDR on all channels of the training sessions, Y-channel block kept
- nfindr: N-FINDR, as in the _nfindr runs
- default: no N-FINDR, as in the original runs
Training overrides (--n_epochs, --lr_decay, ...) apply to every DyNeMo training in the fold
(Y_train and X_test), as they would in a pipeline config.

Usage (from the repository root):
    python examples/simulation/dynemo_mode_recovery/bcv_oracle.py <out_root> --sim dynemo_fair --fold 1 --K 6 --start truth
"""

import argparse
import copy
import json
import os

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_root")
    parser.add_argument("--sim", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--start", default="nfindr", choices=["truth", "nfindr", "nfindr_allch", "default"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--nfindr_window", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--n_kl_annealing_epochs", type=int, default=None)
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--prune", action="store_true",
                        help="Grow then prune: fit K+1 modes on the Y_train data (from the chosen "
                             "start at K+1), drop the least-used mode, start Y_train from the rest")
    args = parser.parse_args()

    from osl_dynamics.evaluate.cross_validation import BCV

    root = f"./results_final/simulation/{args.sim}_nfindr"
    with open(f"{root}/config_simulation.yaml") as f:
        cfg = yaml.safe_load(f)
    model_kwargs = copy.deepcopy(cfg["model"]["dynemo"])
    if args.start == "nfindr":
        model_kwargs["nfindr_kwargs"]["window_length"] = args.nfindr_window
    else:
        model_kwargs.pop("nfindr_kwargs", None)
    config_kwargs = model_kwargs["config_kwargs"]
    for k in ["n_epochs", "n_kl_annealing_epochs", "lr_decay"]:
        if getattr(args, k) is not None:
            config_kwargs[k] = getattr(args, k)
    config_kwargs["n_modes"] = args.K

    tag = args.tag or args.start
    save_dir = f"{args.out_root}/{args.sim}/{tag}/dynemo_state_{args.K}/bcv_{args.fold}/"
    indices = f"{root}/bcv_partition/fold_indices_{args.fold}.json"
    os.makedirs(save_dir, exist_ok=True)
    config = {
        "save_dir": save_dir,
        "indices": indices,
        "load_data": copy.deepcopy(cfg["load_data"]),
        "model": {"dynemo": model_kwargs},
        "mode": f"bcv_{args.fold}",
        "n_modes": args.K,
    }
    with open(f"{save_dir}/oracle_config.yaml", "w") as f:
        yaml.safe_dump({**config, "args": vars(args)}, f, sort_keys=False)

    if (args.start in ["truth", "nfindr_allch"] or args.prune) and args.K > 1:
        from osl_dynamics.config_api.wrappers import load_data
        from osl_dynamics.inference.initializers import nfindr_covariances

        fold = json.load(open(indices))
        cols = fold["column_Y"]
        start_path = f"{save_dir}Y_train_start_covs.npy"
        n_start = args.K + 1 if args.prune else args.K

        def training_sessions(channels, store):
            # Training sessions only (no test sessions), optionally a subset of channels
            load_kwargs = copy.deepcopy(cfg["load_data"])
            load_kwargs.setdefault("kwargs", {})["store_dir"] = f"{save_dir}/{store}/"
            if channels is not None:
                load_kwargs.setdefault("prepare", {}).setdefault("select", {})["channels"] = channels
            data = load_data(**load_kwargs)
            ts = data.time_series(prepared=True, concatenate=False)
            return data, [np.asarray(ts[i]) for i in fold["row_train"]]

        if args.start == "truth":
            # True covariances (simulation parameters) of this fold's Y channels
            C = np.load(f"./data/node_timeseries/simulation_final/{args.sim}/truth/state_covariances.npy")
            if C.shape[0] != n_start:
                raise ValueError("--start truth needs K equal to the true number of modes.")
            C = C[:, cols][:, :, cols]
        elif args.start == "nfindr_allch":
            # N-FINDR on all channels of the training sessions, Y block kept
            data, x = training_sessions(None, "tmp_allch")
            C = nfindr_covariances(x, n_start, window_length=args.nfindr_window, random_seed=args.fold)
            C = C[:, cols][:, :, cols]
            data.delete_dir()
        else:
            C = None

        if args.prune:
            # Fit K+1 modes on the Y_train data with the same settings, drop the least-used mode
            import tensorflow as tf
            from osl_dynamics.data import Data
            from osl_dynamics.models.dynemo import Config, Model

            data, x = training_sessions(cols, "tmp_prune")
            data.delete_dir()
            train_data = Data(x, store_dir=f"{save_dir}/tmp_prune_data/")
            kwargs = {k: v for k, v in config_kwargs.items() if k not in ["n_modes", "n_channels"]}
            model = Model(Config(n_modes=n_start, n_channels=len(cols), initial_covariances=C, **kwargs))
            model.set_regularizers(train_data)
            if C is None and args.start == "nfindr":
                model.nfindr_initialization(train_data, window_length=args.nfindr_window, random_seed=args.fold)
            init = model_kwargs.get("init_kwargs", {})
            model.random_subset_initialization(train_data, n_init=init.get("n_init", 5),
                                               n_epochs=init.get("n_epochs", 2), take=1)
            model.fit(train_data)
            usage = np.concatenate(model.get_alpha(train_data)).mean(0)
            covs = model.get_covariances()
            ll_loss, kl_loss = model.losses(model.make_dataset(train_data, concatenate=True))
            drop = int(np.argmin(usage))
            C = np.delete(covs, drop, axis=0)
            with open(f"{save_dir}/prune_log.json", "w") as f:
                json.dump({"plus1_free_energy": float(ll_loss + kl_loss), "usage": usage.tolist(),
                           "dropped": drop}, f)
            train_data.delete_dir()
            del model
            tf.keras.backend.clear_session()
        np.save(start_path, C.astype(np.float32))

        full_train = BCV.full_train

        def full_train_from_start(self, row, column, save_dir=None):
            # Only the Y_train training gets the start; later steps are unchanged
            if save_dir is not None and save_dir.rstrip("/").endswith("Y_train"):
                self.model_kwargs["config_kwargs"]["initial_covariances"] = start_path
                try:
                    return full_train(self, row, column, save_dir)
                finally:
                    self.model_kwargs["config_kwargs"].pop("initial_covariances")
            return full_train(self, row, column, save_dir)

        BCV.full_train = full_train_from_start

    BCV(config, n_temp_save=5).validate()


if __name__ == "__main__":
    main()
