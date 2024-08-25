"""Train DyNeMo.

"""

print("Setting up")

import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.dynemo import Config, Model

tf_ops.gpu_growth()

data = Data("data/x.npy")

config = Config(
    n_modes=5,
    n_channels=data.n_channels,
    sequence_length=100,
    inference_n_units=32,
    inference_normalization="layer",
    model_n_units=32,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=50,
    batch_size=8,
    learning_rate=0.01,
    n_epochs=100,
)
model = Model(config)

model.multistart_initialization(data, n_init=5, n_epochs=10)
model.fit(data)
model.save("models_2/dynemo")

alp = model.get_alpha(data)
covs = model.get_covariances()

np.save("models_2/dynemo/alp.npy", alp)
np.save("models_2/dynemo/covs.npy", covs)
