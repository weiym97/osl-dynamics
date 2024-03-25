import os
import numpy as np
from rotation.simulation import HMM_single_subject_simulation, perturb_covariances

def create_covariance_matrix(n_channels, rho):
    # Create a matrix with all elements equal to rho
    covariance_matrix = np.full((n_channels, n_channels), rho)

    # Set the diagonal elements to 1.0
    np.fill_diagonal(covariance_matrix, 1.0)

    return covariance_matrix
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
    from osl_dynamics import data, simulation

    # Case 1: All the data are loosely correlated
    save_dir = './data/node_timeseries/simulation_bicv/low_cor/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

