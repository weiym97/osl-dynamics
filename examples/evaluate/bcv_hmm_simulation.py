"""Example script for choosing the number of HMM states with bi-cross-validation.

The data are simulated from an HMM with 5 states. The bi-cross-validated
log-likelihood should increase up to 5 states and then plateau.

The results are saved in results/summary.
"""

print("Importing packages")

from osl_dynamics import data, evaluate, simulation
from osl_dynamics.inference import tf_ops
from osl_dynamics.utils import set_random_seed

set_random_seed(0)

# GPU settings
tf_ops.gpu_growth()

# Simulate data: 20 sessions, 10 channels
n_sessions = 20
n_samples = 2000
n_channels = 10

sim = simulation.HMM_MVN(
    n_samples=n_sessions * n_samples,
    n_states=5,
    n_channels=n_channels,
    trans_prob="sequence",
    stay_prob=0.9,
    means="zero",
    covariances="random",
)
time_series = sim.time_series.reshape(n_sessions, n_samples, n_channels)

training_data = data.Data([x for x in time_series])
training_data.standardize()

# Bi-cross-validation for 1 to 8 states
summary = evaluate.run(
    {
        "data": training_data,
        "save_dir": "results",
        "model": {
            "hmm": {
                "config_kwargs": {
                    "learn_means": False,
                    "learn_covariances": True,
                    "sequence_length": 200,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "n_epochs": 10,
                },
                "init_kwargs": {"n_init": 3, "n_epochs": 1},
            },
        },
        "n_states": [1, 2, 3, 4, 5, 6, 7, 8],
        "modes": {
            "bcv": {
                "split_row": {"method_kwargs": {"n_splits": 5, "train_size": 0.8}},
                "split_column": {"method_kwargs": {"n_splits": 5, "train_size": 0.5}},
            },
        },
    }
)

# Average bi-cross-validated log-likelihood for each number of states
print(summary["bcv"].groupby("n_states")["log_likelihood"].mean())

# Delete the temporary directory created by the Data class
training_data.delete_dir()
