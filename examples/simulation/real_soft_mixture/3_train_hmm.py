"""Train HMM.

"""

print("Setting up")

import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.hmm import Config, Model

tf_ops.gpu_growth()

data = Data("data/x.npy")

config = Config(
    n_states=3,
    n_channels=data.n_channels,
    sequence_length=1000,
    learn_means=False,
    learn_covariances=True,
    batch_size=8,
    learning_rate=0.01,
    n_epochs=40,
)
model = Model(config)
model.summary()

model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
model.fit(data)
model.save("models_2/hmm")

alp = model.get_alpha(data)
covs = model.get_covariances()

np.save("models_2/hmm/alp.npy", alp)
np.save("models_2/hmm/covs.npy", covs)

data.delete_dir()
