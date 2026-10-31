import itertools
import numpy as np
import numpy.testing as npt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def _brute_force_log_evidence(x, means, covs, log_init, log_trans_prob):
    # log p(x) = log sum over all state paths of p(x, s)
    n_states = len(covs)
    log_b = np.stack([multivariate_normal(means[k], covs[k]).logpdf(x)
                      for k in range(n_states)], axis=1)
    log_joint = []
    for path in itertools.product(range(n_states), repeat=len(x)):
        log_joint.append(log_init[path[0]]
                         + sum(log_trans_prob[path[t - 1], path[t]] for t in range(1, len(x)))
                         + sum(log_b[t, path[t]] for t in range(len(x))))
    return logsumexp(log_joint)


def _build_model(covariances, trans_prob, sequence_length, batch_size):
    from osl_dynamics.models import hmm

    config = hmm.Config(
        n_states=covariances.shape[0],
        n_channels=covariances.shape[1],
        sequence_length=sequence_length,
        learn_means=False,
        learn_covariances=True,
        initial_covariances=covariances,
        initial_trans_prob=trans_prob,
        batch_size=batch_size,
        learning_rate=0.01,
        n_epochs=1,
    )
    return hmm.Model(config)


def test_hmm_evidence():
    from osl_dynamics.data.base import Data

    # 6 sequences with batch_size 4, so the last batch has 2 sequences
    sequence_length, batch_size = 5, 4
    covariances = np.array([[[1.0, 0.5], [0.5, 1.0]],
                            [[2.0, -0.3], [-0.3, 0.5]],
                            [[0.3, 0.0], [0.0, 0.3]]])
    trans_prob = np.array([[0.8, 0.15, 0.05],
                           [0.1, 0.7, 0.2],
                           [0.25, 0.25, 0.5]])
    model = _build_model(covariances, trans_prob, sequence_length, batch_size)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((6 * sequence_length, 2)).astype(np.float32)
    data = Data(x)

    # Each sequence starts from the stationary distribution
    means, covs = model.get_means_covariances()
    eigval, eigvec = np.linalg.eig(trans_prob.T)
    pi = np.real(eigvec[:, np.argmin(np.abs(eigval - 1))])
    pi /= pi.sum()
    log_evidence = np.array([
        _brute_force_log_evidence(x_seq, means, covs, np.log(pi), np.log(trans_prob))
        for x_seq in x.reshape(-1, sequence_length, 2).astype(np.float64)
    ])

    # Average over sequences, not over batches
    npt.assert_allclose(model.evidence(data), np.mean(log_evidence), atol=1e-3)
    batch_weighted = (np.mean(log_evidence[:4]) + np.mean(log_evidence[4:])) / 2
    assert abs(np.mean(log_evidence) - batch_weighted) > 1e-2
    data.delete_dir()


def test_hmm_free_energy():
    from osl_dynamics.data.base import Data

    # 3 sequences with batch_size 2, so the last batch has 1 sequence
    sequence_length, batch_size = 3, 2
    covariances = np.array([[[1.0, 0.5], [0.5, 1.0]],
                            [[0.3, 0.0], [0.0, 0.3]]])
    trans_prob = np.array([[0.9, 0.1],
                           [0.2, 0.8]])
    model = _build_model(covariances, trans_prob, sequence_length, batch_size)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((3 * sequence_length, 2)).astype(np.float32)
    data = Data(x)

    # With the exact posterior, the free energy of a batch is -log p(x) of the
    # batch's sequences as one chain, starting from state_probs_t0 (uniform)
    means, covs = model.get_means_covariances()
    log_init = np.log(np.ones(2) / 2)
    x = x.astype(np.float64)
    batch_free_energy = np.array([
        -_brute_force_log_evidence(x[:2 * sequence_length], means, covs, log_init, np.log(trans_prob)),
        -_brute_force_log_evidence(x[2 * sequence_length:], means, covs, log_init, np.log(trans_prob)),
    ])

    # Average over sequences, not over batches
    npt.assert_allclose(model.free_energy(data), np.sum(batch_free_energy) / 3, atol=1e-3)
    batch_weighted = (batch_free_energy[0] / 2 + batch_free_energy[1]) / 2
    assert abs(np.sum(batch_free_energy) / 3 - batch_weighted) > 1e-2
    data.delete_dir()
