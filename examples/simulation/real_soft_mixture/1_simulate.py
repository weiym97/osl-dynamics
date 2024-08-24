"""Simulate data based on real alphas and covariances from DyNeMo.

"""

import numpy as np
from tqdm import trange

alp = np.load("data/alp.npy")
covs = np.load("data/covs.npy")

alp = alp[:25600]

C = np.sum(alp[:, :, np.newaxis, np.newaxis] * covs[np.newaxis, :, :], axis=1)

x = []
for i in trange(C.shape[0]):
    m = np.zeros(C[i].shape[0])
    c = C[i]
    x.append(np.random.multivariate_normal(m, c))
np.save("data/x.npy", x)
