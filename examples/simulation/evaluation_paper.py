import os
import pickle
import numpy as np
from osl_dynamics import simulation
from osl_dynamics.array_ops import apply_hrf


def hmm_iid(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/hmm_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.9,
        means='zero',
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)
    np.save(f'{save_dir}truth/tpm.npy', sim.hmm.trans_prob)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def hmm_iid_final(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/hmm_iid_final/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.9,
        means='zero',
        covariances='random',
        n_covariances_act=3
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)
    np.save(f'{save_dir}truth/tpm.npy', sim.hmm.trans_prob)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def hmm_iid_mean_only(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/hmm_iid_mean_only/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    covariances = np.array([np.eye(n_channels) for _ in range(n_states)])

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.9,
        means='random',
        covariances=covariances
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)
    np.save(f'{save_dir}truth/tpm.npy', sim.hmm.trans_prob)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def real_ground_truth(save_dir, model_name, repeat, n_states=6, real_dir='./results_final/real/ICA_50'):
    '''
    Save a real-data fit (alphas and covariances) as a simulation ground truth.
    Only repeat_1 of the ICA_50 fits saved alp.pkl, so here alp.pkl is inferred from the saved
    model of another repeat, using the same data and preparation as in training.
    '''
    from osl_dynamics import models
    from osl_dynamics.config_api.wrappers import load_data

    repeat_dir = f'{real_dir}/{model_name}_state_{n_states}/repeat_{repeat}'
    out_dir = f'{save_dir}/{model_name}_ICA_50_ground_truth_{repeat}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Same load_data settings as prepared_config.yaml of the real-data runs
    data = load_data(
        inputs='./data/node_timeseries/3T_HCP1200_MSMAll_d50_ts2/',
        kwargs={'load_memmaps': True, 'store_dir': f'{out_dir}tmp/'},
        prepare={'select': {'timepoints': [0, 4800]}, 'standardize': {'session_length': 1200}},
    )
    model = models.load(f'{repeat_dir}/model')
    alpha = model.get_alpha(data)
    data.delete_dir()

    print('Sanity check:')
    print(f'Length of alpha:{len(alpha)}')
    print(f'alpha[0] shape: {alpha[0].shape}')
    print(f'every row sums to 1: {all(np.allclose(a.sum(axis=1), 1, atol=1e-4) for a in alpha)}')

    with open(f'{out_dir}alp.pkl', 'wb') as f:
        pickle.dump(alpha, f)
    np.save(f'{out_dir}covs.npy', np.load(f'{repeat_dir}/inf_params/covs.npy'))

def hmm_iid_real(save_dir, n_subjects, n_samples, n_states, n_channels, tr,
                 ground_truth='hmm_ICA_50_ground_truth', name='hmm_iid_real'):
    import random
    from osl_dynamics.simulation.mvn import MVN
    from osl_dynamics.inference.modes import argmax_time_courses

    # Read in the ground-truth covariances and alphas.
    with open(f"{save_dir}/{ground_truth}/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha_truth = pickle.load(f)
    covariances = np.load(f'{save_dir}/{ground_truth}/covs.npy')

    # Step 1: Select 500 subjects for simulation
    random.seed(42)  # Optional: for reproducibility
    selected_indices = random.sample(range(len(alpha_truth)), n_subjects)
    alpha_selected = [alpha_truth[i] for i in selected_indices]

    # Step 2: Truncate each array to the first 1200 time points
    alpha_trimmed = [a[:n_samples] for a in alpha_selected]

    # Step 3: argmax time courses
    alpha = argmax_time_courses(alpha_trimmed)

    save_dir = f'{save_dir}/{name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    print('Sanity check:')
    print(f'Length of alpha:{len(alpha)}')
    print(f'alpha[0] shape: {alpha[0].shape}')
    print(f'every row is one-hot: {all(np.all(a.sum(axis=1) == 1) for a in alpha)}')

    np.save(f'{save_dir}/truth/state_covariances.npy', covariances)

    mvn = MVN(means='zero', covariances=covariances)

    for i in range(n_subjects):
        time_course = alpha[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course)

def hmm_iid_real_mean(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    import random
    from osl_dynamics.simulation.mvn import MVN
    from osl_dynamics.inference.modes import argmax_time_courses

    # Read in the ground-truth covariances and alphas.
    with open(f"{save_dir}/hmm_ICA_50_mean_ground_truth/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha_truth = pickle.load(f)
    means = np.load(f'{save_dir}/hmm_ICA_50_mean_ground_truth/means.npy')
    covariances = np.load(f'{save_dir}/hmm_ICA_50_mean_ground_truth/covs.npy')

    # Step 1: Select 500 subjects for simulation
    random.seed(42)  # Optional: for reproducibility
    selected_indices = random.sample(range(len(alpha_truth)), n_subjects)
    alpha_selected = [alpha_truth[i] for i in selected_indices]

    # Step 2: Truncate each array to the first 1200 time points
    alpha_trimmed = [a[:n_samples] for a in alpha_selected]

    # Step 3: argmax time courses
    alpha = argmax_time_courses(alpha_trimmed)

    save_dir = f'{save_dir}/hmm_iid_real_mean/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    print('Sanity check:')
    print(f'Length of alpha:{len(alpha)}')
    print(f'alpha[0] shape: {alpha[0].shape}')
    print(f'alpha[0][:20]: {alpha[0][:20]}')

    np.save(f'{save_dir}/truth/state_means.npy', means)
    np.save(f'{save_dir}/truth/state_covariances.npy', covariances)

    mvn = MVN(means=means, covariances=covariances)

    for i in range(n_subjects):
        time_course = alpha[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course)

def hmm_iid_meg_tde(save_dir,):
    from osl_dynamics.simulation.mvn import MVN
    from osl_dynamics.inference.modes import argmax_time_courses

    # Read in the ground-truth covariances and alphas.
    with open(f"{save_dir}/hmm_meg_tde_ground_truth/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha = pickle.load(f)
    means = np.load(f'{save_dir}/hmm_meg_tde_ground_truth/means.npy')
    covariances = np.load(f'{save_dir}/hmm_meg_tde_ground_truth/covs.npy')


    # Step 3: argmax time courses
    alpha = argmax_time_courses(alpha)

    save_dir = f'{save_dir}/hmm_iid_meg_tde/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    print('Sanity check:')
    print(f'Length of alpha:{len(alpha)}')
    print(f'alpha[0] shape: {alpha[0].shape}')
    print(f'alpha[0][:20]: {alpha[0][:20]}')

    np.save(f'{save_dir}/truth/state_covariances.npy', covariances)
    np.save(f'{save_dir}/truth/state_means.npy', means)

    mvn = MVN(means=means, covariances=covariances)

    for i in range(len(alpha)):
        time_course = alpha[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course)

def hmm_iid_meg_78(save_dir,):
    from osl_dynamics.simulation.mvn import MVN
    from osl_dynamics.inference.modes import argmax_time_courses

    # Read in the ground-truth covariances and alphas.
    with open(f"{save_dir}/hmm_meg_78_ground_truth/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha = pickle.load(f)
    means = np.load(f'{save_dir}/hmm_meg_78_ground_truth/means.npy')
    covariances = np.load(f'{save_dir}/hmm_meg_78_ground_truth/covs.npy')


    # Step 3: argmax time courses
    alpha = argmax_time_courses(alpha)

    save_dir = f'{save_dir}/hmm_iid_meg_78/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    print('Sanity check:')
    print(f'Length of alpha:{len(alpha)}')
    print(f'alpha[0] shape: {alpha[0].shape}')
    print(f'alpha[0][:20]: {alpha[0][:20]}')

    np.save(f'{save_dir}/truth/state_covariances.npy', covariances)
    np.save(f'{save_dir}/truth/state_means.npy', means)

    mvn = MVN(means=means, covariances=covariances)

    for i in range(len(alpha)):
        time_course = alpha[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course)

def hmm_hrf(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/hmm_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.9,
        means='zero',
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)
    np.save(f'{save_dir}truth/tpm.npy', sim.hmm.trans_prob)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i], tr))
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def hmm_hrf_noise(save_dir, noise_ratios, n_subjects=500, seed=42):
    '''
    Add white measurement noise to the hmm_hrf BOLD data (noise ratio 0 is hmm_hrf itself).
    For a noise-to-signal variance ratio r, channel c gets independent Gaussian noise with
    variance r * (variance of channel c in hmm_hrf, pooled over subjects).
    The same standard-normal draw is used for every r, so the datasets differ only in noise amplitude.
    truth/ is a link to the hmm_hrf one (state time courses and covariances are unchanged).
    '''
    data = np.array([np.loadtxt(f'{save_dir}/hmm_hrf/{10001 + i}.txt') for i in range(n_subjects)])
    signal_var = data.reshape(-1, data.shape[-1]).var(axis=0)

    for r in noise_ratios:
        noise_dir = f'{save_dir}/hmm_hrf_noise_{r}/'
        if not os.path.exists(noise_dir):
            os.makedirs(noise_dir)
        if not os.path.exists(f'{noise_dir}truth'):
            os.symlink('../hmm_hrf/truth', f'{noise_dir}truth')
        noise_std = np.sqrt(float(r) * signal_var)
        np.save(f'{noise_dir}noise_std.npy', noise_std)

        for i in range(n_subjects):
            z = np.random.default_rng([seed, i]).standard_normal(data[i].shape)
            np.savetxt(f'{noise_dir}{10001 + i}.txt', data[i] + z * noise_std)
        print(f'hmm_hrf_noise_{r}: wrote {n_subjects} subjects')


def dynemo_iid(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/dynemo_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.MixedSine_MVN(
        n_samples=n_subjects * n_samples,
        n_modes=n_states,
        n_channels=n_channels,
        relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.2],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1, 2, 3, 4, 6, 8],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])


def dynemo_iid_new(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/dynemo_iid_new/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.MixedSine_MVN(
        n_samples=n_subjects * n_samples,
        n_modes=n_states,
        n_channels=n_channels,
        relative_activation=[1, 1.75, 2.5, 3.25, 4.0, 4.75],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
        sampling_frequency=1.389,
        means="zero",
        covariances="random",
        n_covariances_act=3
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])


def dynemo_iid_final(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/dynemo_iid_final/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    #frequency_ratio = 1.389 / 250
    #frequencies = [1.2, 2.2, 3.2, 4.2, 5.2, 6.2]
    #new_frequencies = [i * frequency_ratio for i in frequencies]
    sim = simulation.MixedSine_MVN(
        n_samples=n_subjects * n_samples,
        n_modes=n_states,
        n_channels=n_channels,
        relative_activation=[1.0, 1.75, 2.5, 3.25, 4.0, 4.75],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1.3, 2.3, 3.3, 4.3, 5.3, 6.3],
        sampling_frequency=250,
        means="zero",
        covariances="random",
        n_covariances_act=3
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])

def dynemo_iid_real(save_dir, n_subjects, n_samples, n_states, n_channels, tr,
                    ground_truth='dynemo_ICA_50_ground_truth', name='dynemo_iid_real'):
    import random
    from osl_dynamics.simulation.mvn import MVN

    with open(f"{save_dir}/{ground_truth}/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha_truth = pickle.load(f)
    covariances = np.load(f'{save_dir}/{ground_truth}/covs.npy')


    save_dir = f'{save_dir}/{name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    # Step 1: Select 500 subjects for simulation
    random.seed(42)  # Optional: for reproducibility
    selected_indices = random.sample(range(len(alpha_truth)), n_subjects)
    alpha_selected = [alpha_truth[i] for i in selected_indices]

    # Step 2: Truncate each array to the first 1200 time points
    alpha = [a[:n_samples] for a in alpha_selected]

    print('Sanity check:')
    print(f'Length of alpha:{len(alpha)}')
    print(f'alpha[0] shape: {alpha[0].shape}')
    print(f'every row sums to 1: {all(np.allclose(a.sum(axis=1), 1, atol=1e-4) for a in alpha)}')

    np.save(f'{save_dir}truth/state_covariances.npy', covariances)

    mvn = MVN(means='zero', covariances=covariances)

    for i in range(n_subjects):
        time_course = alpha[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course)


def dynemo_hrf(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/dynemo_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.MixedSine_MVN(
        n_samples=n_subjects * n_samples,
        n_modes=n_states,
        n_channels=n_channels,
        relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.2],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1, 2, 3, 4, 6, 8],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i], tr))
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])


def swc_iid(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/swc_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.SWC_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        stay_time=100,
        means="zero",
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])


def swc_hrf(save_dir, n_subjects, n_samples, n_states, n_channels, tr):
    save_dir = f'{save_dir}/swc_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.SWC_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        stay_time=100,
        means="zero",
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i], tr))
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])


def soft_mixing(save_dir):
    save_dir = f'{save_dir}/soft_mixing/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    n_subjects = 64
    n_channels = 80
    n_modes = 6
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
    )

    time_course = sim.mode_time_course
    data = sim.time_series
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_modes)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])


def dynemo_fair(save_dir):
    save_dir = f'{save_dir}/dynemo_fair/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    n_subjects = 64
    n_channels = 80
    n_modes = 6
    sim = simulation.MixedSine_MVN(
        n_samples=25600,
        n_modes=n_modes,
        n_channels=n_channels,
        # relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.2],
        relative_activation=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_modes)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])


def dynemo_ukb(save_dir):
    from osl_dynamics.simulation.mvn import MVN

    save_dir = f'{save_dir}/dynemo_ukb/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    n_subjects = 500
    ukb_dir = './results_final/real/ICA_50_UKB/dynemo_state_6/repeat_1/inf_params/'
    with open(f"{ukb_dir}/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha = pickle.load(f)[:n_subjects]
    covariances = np.load(f'{ukb_dir}/covs.npy')

    np.save(f'{save_dir}truth/state_covariances.npy', covariances)

    mvn = MVN(means='zero', covariances=covariances)

    for i in range(n_subjects):
        time_course = alpha[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course)


def dynemo_meg(save_dir):
    import pickle
    from osl_dynamics.simulation.mvn import MVN

    save_dir = f'{save_dir}/dynemo_meg/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    n_subjects = 10
    meg_dir = './dynemo-paper-resting-state/data/'
    with open(f"{meg_dir}/alp.pkl", "rb") as f:  # "rb" means read binary mode
        alpha = pickle.load(f)
    alpha_trimmed = [a[:60000] for a in alpha]
    covariances = np.load(f'{meg_dir}/covs.npy')

    np.save(f'{save_dir}truth/state_covariances.npy', covariances)

    mvn = MVN(means='zero', covariances=covariances)

    for i in range(n_subjects):
        time_course = alpha_trimmed[i]
        data = mvn.simulate_data(time_course)
        np.savetxt(f'{save_dir}{10001 + i}.txt', data)
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course)


def main(simulation_list=None):
    save_dir = './data/node_timeseries/simulation_final/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config = {
        'save_dir': save_dir,
        'n_subjects': 500,
        'n_states': 6,
        'n_channels': 50,
        'n_samples': 1200,
        'tr': 0.72
    }

    if 'hmm_iid' in simulation_list:
        hmm_iid(**config)
    if 'hmm_hrf' in simulation_list:
        hmm_hrf(**config)
    # e.g. hmm_hrf_noise_0.25 hmm_hrf_noise_1.0 hmm_hrf_noise_4.0 (needs hmm_hrf to exist)
    noise_ratios = [s[len('hmm_hrf_noise_'):] for s in simulation_list if s.startswith('hmm_hrf_noise_')]
    if noise_ratios:
        hmm_hrf_noise(save_dir, noise_ratios)
    if 'hmm_iid_mean_only' in simulation_list:
        hmm_iid_mean_only(**config)
    if 'hmm_iid_final' in simulation_list:
        hmm_iid_final(**config)
    if 'hmm_iid_real' in simulation_list:
        hmm_iid_real(**config)
    if 'hmm_iid_real_mean' in simulation_list:
        hmm_iid_real_mean(**config)

    if 'dynemo_iid' in simulation_list:
        dynemo_iid(**config)
    if 'dynemo_iid_new' in simulation_list:
        dynemo_iid_new(**config)

    if 'dynemo_iid_final' in simulation_list:
        dynemo_iid_final(**config)
    if 'dynemo_iid_real' in simulation_list:
        dynemo_iid_real(**config)
    if 'dynemo_hrf' in simulation_list:
        dynemo_hrf(**config)
    if 'swc_iid' in simulation_list:
        swc_iid(**config)
    if 'swc_hrf' in simulation_list:
        swc_hrf(**config)

    if 'soft_mixing' in simulation_list:
        soft_mixing(save_dir)
    if 'dynemo_UKB' in simulation_list:
        dynemo_ukb(save_dir)

    if 'dynemo_fair' in simulation_list:
        dynemo_fair(save_dir)

    if 'dynemo_meg' in simulation_list:
        dynemo_meg(save_dir)

    if 'hmm_iid_meg_tde' in simulation_list:
        hmm_iid_meg_tde(save_dir)

    if 'hmm_iid_meg_78' in simulation_list:
        hmm_iid_meg_78(save_dir)

    # Repeats of hmm_iid_real / dynemo_iid_real, using the other real-data fits (repeat_2, repeat_3)
    for repeat in [2, 3]:
        if f'hmm_ICA_50_ground_truth_{repeat}' in simulation_list:
            real_ground_truth(save_dir, 'hmm', repeat)
        if f'dynemo_ICA_50_ground_truth_{repeat}' in simulation_list:
            real_ground_truth(save_dir, 'dynemo', repeat)
        if f'hmm_iid_real_{repeat}' in simulation_list:
            hmm_iid_real(**config, ground_truth=f'hmm_ICA_50_ground_truth_{repeat}', name=f'hmm_iid_real_{repeat}')
        if f'dynemo_iid_real_{repeat}' in simulation_list:
            dynemo_iid_real(**config, ground_truth=f'dynemo_ICA_50_ground_truth_{repeat}', name=f'dynemo_iid_real_{repeat}')


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

