import os
import pickle

from osl_dynamics.inference.metrics import alpha_correlation

if __name__ == '__main__':
    save_dir= './results_HCP_yaml_202403/'
    N_states = list(range(2,37))
    N_splits = 5

    for N_state in N_states:
        for N_split in range(1,N_splits+1):
            split_1_dir = f'{save_dir}hmm_ICA_25_state_{N_state}/split_{N_split}/half_1/'
            split_2_dir = f'{save_dir}hmm_ICA_25_state_{N_state}/split_{N_split}/half_2/'

            with open(f'{split_1_dir}inf_params/alp.pkl','rb') as file:
                alpha_1 = pickle.load(file)
            with open(f'{split_2_dir}inf_params/alp.pkl','rb') as file:
                alpha_2 = pickle.load(file)

            temp_corr = alpha_correlation(alpha_1,alpha_2)


