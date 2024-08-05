"""Example script for demonstrating DyNeMo's ability to infer a soft mixture of modes.

"""

print("Setting up")
import os
import pickle

import numpy as np
from tqdm.auto import trange

from osl_dynamics import data, simulation
from osl_dynamics.inference import metrics, modes, tf_ops
from osl_dynamics.models.dynemo import Config, Model
from osl_dynamics.utils import plotting

# Create directory to hold plots
os.makedirs("figures_init", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()


n_modes =6
n_channels = 80

print("Simulating data")
sim = simulation.MixedSine_MVN(
    n_samples=25600,
    n_modes=n_modes,
    n_channels=n_channels,
    relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.1],
    amplitudes=[6, 5, 4, 3, 2, 1],
    frequencies=[1, 2, 3, 4, 6, 8],
    sampling_frequency=250,
    means="zero",
    covariances="random",
    random_seed=123,
)
sim_alp = sim.mode_time_course
sim_cov = sim.covariances
training_data = data.Data(sim.time_series)

# Plot ground truth logits
plotting.plot_separate_time_series(
    sim.logits, n_samples=2000, filename="figures_init/sim_logits.png"
)

# Settings
config = Config(
    n_modes=n_modes,
    n_channels=n_channels,
    sequence_length=200,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    initial_means=np.zeros((n_modes,n_channels)),
    initial_covariances=sim_cov,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=1000,
    batch_size=16,
    learning_rate=0.001,
    n_epochs=2000,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_data,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="tmp/weights",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(training_data)
print(f"Free energy: {free_energy}")

# Inferred alpha and mode time course
inf_alp = model.get_alpha(training_data)
sim_alp, inf_alp = modes.match_modes(sim_alp, inf_alp)

with open('figures_init/sim_alp.pkl', 'wb') as f:
    pickle.dump(sim_alp, f)
with open('figures_init/inf_alp.pkl', 'wb') as f:
    pickle.dump(inf_alp, f)

# Compare the inferred mode time course to the ground truth
plotting.plot_alpha(
    sim_alp,
    n_samples=2000,
    title="Ground Truth",
    y_labels=r"$\alpha_{jt}$",
    filename="figures_init/sim_alp.png",
)
plotting.plot_alpha(
    inf_alp,
    n_samples=2000,
    title="DyNeMo",
    y_labels=r"$\alpha_{jt}$",
    filename="figures_init/inf_alp.png",
)

# Correlation between mode time courses
corr = metrics.alpha_correlation(inf_alp, sim_alp)
print("Correlation (DyNeMo vs Simulation):", corr)

# Reconstruction of the time-varying covariance
inf_cov = model.get_covariances()

np.save("figures_init/sim_cov.npy",sim_cov)
np.save("figures_init/inf_cov.npy",inf_cov)

sim_tvcov = np.sum(
    sim_alp[:, :, np.newaxis, np.newaxis] * sim_cov[np.newaxis, :, :, :], axis=1
)
inf_tvcov = np.sum(
    inf_alp[:, :, np.newaxis, np.newaxis] * inf_cov[np.newaxis, :, :, :], axis=1
)

# Calculate the Riemannian distance between the ground truth and inferred covariance
print("Calculating riemannian distances")
rd = np.empty(2000)
for i in trange(2000):
    rd[i] = metrics.riemannian_distance(sim_tvcov[i], inf_tvcov[i])

plotting.plot_line(
    [range(2000)],
    [rd],
    labels=["DyNeMo"],
    x_label="Sample",
    y_label="$d$",
    fig_kwargs={"figsize": (15, 1.5)},
    filename="figures_init/rd.png",
)

# Save trained model
from osl_dynamics.utils.misc import save
model_dir = 'figures_init/model/'
model.save(model_dir)
save(f"{model_dir}/history.pkl", history)

# Delete temporary directory
training_data.delete_dir()
