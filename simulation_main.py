import os
import numpy as np
from rotation.simulation import HMM_single_subject_simulation, perturb_covariances

def create_covariance_matrix(n_channels, rho):
    # Create a matrix with all elements equal to rho
    cov = np.eye(n_channels)
    cov[np.arange(1, n_channels), np.arange(n_channels - 1)] = rho
    cov[np.arange(n_channels - 1), np.arange(1, n_channels)] = rho

    return cov

if __name__ == '__main__':
    '''
    save_dir ='./data/node_timeseries/simulation_202401/sigma_0.025/run_0/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    perturbation_factor = 0.025
    # Define the parameters
    n_scans = 100
    n_states = 8
    n_samples = 1200
    n_channels = 25

    # Read from early TPM
    #tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_50_state_4/trans_prob.npy')
    #tpm = np.array([[0.8,0.2],[0.2,0.8]])
    #tpm = np.array([[0.96,0.01,0.01,0.01,0.01],
    #                 [0.01,0.96,0.01,0.01,0.01],
    #                 [0.01,0.01,0.96,0.01,0.01],
    #                 [0.01,0.01,0.01,0.96,0.01],
    #                 [0.01,0.01,0.01,0.01,0.96]])
    #tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_25_state_8/trans_prob.npy')

    # tpm for simulation_toy_6
    #diagonal_value = 0.99
    #off_diagonal_value = (1 - diagonal_value) / (n_states - 1)
    #tpm = diagonal_value * np.eye(n_states) + off_diagonal_value * (1 - np.eye(n_states))

    # tpm for simulation_toy_9 (two subjects)
    tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_25_state_8/trans_prob.npy')


    #means = np.load('./results_202310/HMM_ICA_50_state_4/state_means.npy')
    means = np.zeros((n_states,n_channels))
    #covariances = np.load('./results_HCP_202311_no_mean/HMM_ICA_50_state_4/state_covariances.npy')
    #covariances = np.array([[[0.25,0.2],[0.2,0.25]],[[1.,0.8],[0.8,1.]]])
    #covariances = np.array([
    #    [[1.,-0.1],[-0.1,1.]],
    #    [[1.,-0.05],[-0.05,1.]],
    #    [[1.,0.],[0.,1.]],
    #    [[1.,0.05],[0.05,1.]],
    #    [[1.,0.1],[0.1,1.]]
    #])
    covariances = np.load('./results_HCP_202311_no_mean/HMM_ICA_25_state_8/state_covariances.npy')

    # Update 20240119: generate ten subjects for simulation
    for i in range(10001,10011):
        HMM_single_subject_simulation(save_dir=save_dir,
                                      n_scans=n_scans,
                                      n_states=n_states,
                                      n_samples=n_samples,
                                      n_channels=n_channels,
                                      trans_prob=tpm,
                                      means=means,
                                      covariances=perturb_covariances(covariances,perturbation_factor=perturbation_factor),
                                      subj_name = str(i))
    '''
    #############################################################
    # Update 25th March 2024: Generate bi-cross validation simulation
    '''
    from osl_dynamics import data, simulation

    # Case 1: All the data are weakly correlated
    save_dir = './data/node_timeseries/simulation_bicv/low_cor/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth')

    n_subjects = 500
    n_states = 5
    n_samples = 1200
    n_channels = 25
    stay_prob = 0.8

    cors = [-0.2,-0.1,0.0,0.1,0.2]

    covs = [create_covariance_matrix(n_channels,cor) for cor in cors]
    covs = np.stack(covs)

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob="uniform",
        stay_prob=stay_prob,
        means="zero",
        covariances=covs,
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

    # Case 2: All the data are strongly correlated
    save_dir = './data/node_timeseries/simulation_bicv/high_cor/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth')

    n_subjects = 500
    n_states = 5
    n_samples = 1200
    n_channels = 25
    stay_prob = 0.8

    cors = [-0.49, -0.3, 0.0, 0.3, 0.49]

    covs = [create_covariance_matrix(n_channels, cor) for cor in cors]
    covs = np.stack(covs)

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob="uniform",
        stay_prob=stay_prob,
        means="zero",
        covariances=covs,
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

    # Case 3: TPM and Covariances come from the real data
    save_dir = './data/node_timeseries/simulation_bicv/real/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth')

    n_subjects = 500
    n_states = 8
    n_samples = 1200
    n_channels = 25

    tpm = np.load('./results_HCP_202311_no_mean/HMM_ICA_25_state_8/trans_prob.npy')
    covs = np.load('./results_HCP_202311_no_mean/HMM_ICA_25_state_8/state_covariances.npy')
    for i in range(len(covs)):
        covs[i] = (covs[i] + covs[i].T)/2


    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob=tpm,
        means="zero",
        covariances=covs,
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

    # Case 4: Covariances random, generated by Chet & Rukuang's code
    save_dir = './data/node_timeseries/simulation_bicv/random/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth')

    n_subjects = 500
    n_states = 8
    n_samples = 1200
    n_channels = 25


    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.8,
        means="zero",
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
    '''
    #############################################################
    ### Update 31st March 2024
    ### This is a simulation for bi cross validation
    ### We try to answer the question: what happens when the state covariances are "sparse"
    from osl_dynamics import data, simulation
    from osl_dynamics.simulation.mvn import MVN

    # Case 1: Covariances can be divided into three blocks (8,8,9 channels),
    # Each block have two states, thus there are 8 states in total.

    '''
    save_dir = './data/node_timeseries/simulation_bicv/3_block/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth')

    n_subjects = 500
    n_states = 8
    n_samples = 1200
    n_channels = 25
    n_channels_split = [8,8,9]

    cov = []
    for i in range(3):
        mvn = MVN(means='zero',
                  covariances='random',
                  n_modes=2,
                  n_channels = n_channels_split[i])
        cov.append(mvn.covariances)
    # Build up the covariances using blocked ones.
    from scipy.linalg import block_diag
    covariances = np.zeros((n_states,n_channels,n_channels))
    for i in range(8):
        index = [int(digit) for digit in bin(i)[2:].zfill(3)]
        covariances[i,:,:] = block_diag(cov[0][index[0]],cov[1][index[1]],cov[2][index[2]])
    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.8,
        means="zero",
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
    '''
    # Case 2: Randomly generate 8 state covariances, but only
    # preserve the diagonal value
    '''
    save_dir = './data/node_timeseries/simulation_bicv/diagonal/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth')

    n_subjects = 500
    n_states = 8
    n_samples = 1200
    n_channels = 25

    mvn = MVN(means='zero',
              covariances='random',
              n_modes=n_states,
             n_channels=n_channels)
    covariances = mvn.covariances
    for i in range(n_states):
        temp = covariances[i]

        # Set off-diagonal elements to zero
        temp[~np.eye(temp.shape[0], dtype=bool)] = 0

        # Update the modified covariance matrix in the array
        covariances[i] = temp
    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.8,
        means="zero",
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
    '''
    # Case 3: 25 state covariances without correlation, all variances are unit
    # except for the ith compoment in the state i
    save_dir = './data/node_timeseries/simulation_bicv/diagonal_special/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth')

    n_subjects = 500
    n_states = 8
    n_samples = 1200
    n_channels = 25

    covariances = np.zeros((n_states,n_channels,n_channels))
    for i in range(n_states):
        covariances[i] = np.eye(n_channels)
        covariances[i,i,i] = 2

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.8,
        means="zero",
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

    #############################################################
    '''
    ### Update 6th Dec 2023
    ### This is a simulation from Chet, it's only for comparison purposes.
    ### Please comment out all previous codes when doing the following simulation
    save_dir = './data/node_timeseries/simulation_rukuang/state_100/'
    n_samples = 25600
    n_states = 100
    n_channels = 25
    n_subjects = 50
    stay_prob = 0.7
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    from osl_dynamics import data, simulation


    from osl_dynamics.data import Data
    # Create Data object for training

    sim = simulation.HMM_MVN(
        n_samples=n_samples*n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob="sequence",
        stay_prob=stay_prob,
        means="zero",
        covariances="random",
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects,-1,n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001+i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001+i}_state_time_course.npy', time_course[i])
    '''

