import os
import sys
import json
import pickle
import pathlib

import numpy as np
from osl_dynamics.inference.metrics import alpha_correlation
from osl_dynamics.inference.modes import hungarian_pair
#from rotation.preprocessing import PrepareData
from osl_dynamics.data.base import Data
from osl_dynamics.models import load
from osl_dynamics.utils.plotting import plot_mode_pairing

if __name__ == '__main__':
    save_dir= './results_simulation_202403/sigma_0/'
    N_states = list(range(2,17))
    index = int(sys.argv[1]) - 1
    N_state = N_states[index]
    N_splits = 5
    data_dir = './data/node_timeseries/simulation_202402/sigma_0/'
    z_score_data = False



    for N_split in range(1, N_splits + 1):
        analysis_dir = f'{save_dir}hmm_ICA_25_state_{N_state}/split_{N_split}/'
        print(f'analysis_dir: {analysis_dir}')
        split_1_dir = f'{analysis_dir}half_1/'
        split_2_dir = f'{analysis_dir}half_2/'

        with open(f'{split_1_dir}inf_params/alp.pkl', 'rb') as file:
            alpha_1 = pickle.load(file)
        with open(f'{split_2_dir}inf_params/alp.pkl', 'rb') as file:
            alpha_2 = pickle.load(file)

        dataset = Data(data_dir)
        dataset.standardize()
        dataset.select(timepoints=[0,1200])

        model_1 = load(f'{split_1_dir}model')
        model_2 = load(f'{split_2_dir}model')

        with open(f'{analysis_dir}indices_1.json') as f:
            # Load the JSON data
            indice_1 = json.load(f)

        with open(f'{analysis_dir}/indices_2.json') as f:
            # Load the JSON data
            indice_2 = json.load(f)

        with dataset.set_keep(indice_1):
            alpha_1_cv = model_2.get_alpha(dataset)

        with dataset.set_keep(indice_2):
            alpha_2_cv = model_1.get_alpha(dataset)


        temp_cor_1 = alpha_correlation(alpha_1,alpha_1_cv,return_diagonal=False)
        temp_cor_2 = alpha_correlation(alpha_2,alpha_2_cv,return_diagonal=False)

        pair_1, temp_cor_1_pair = hungarian_pair(temp_cor_1)
        pair_2, temp_cor_2_pair = hungarian_pair(temp_cor_2)


        plot_mode_pairing(temp_cor_1_pair,pair_1,filename=f'{analysis_dir}temp_cor_1.jpg')
        plot_mode_pairing(temp_cor_2_pair, pair_2, filename=f'{analysis_dir}temp_cor_2.jpg')

        cov_1 = np.load(f'{split_1_dir}inf_params/covs.npy')
        cov_2 = np.load(f'{split_2_dir}inf_params/covs.npy')

        # Set the covariances after the pairing
        model_1.set_covariances(cov_2[pair_1['col']])
        model_2.set_covariances(cov_1[pair_2['col']])
        with dataset.set_keep(indice_1):
            free_energy_cv_1, log_likelihood_cv_1, entropy_cv_1, prior_cv_1 = model_1.free_energy(dataset,return_components=True)
            evidence_cv_1 = model_1.evidence(dataset)
        with dataset.set_keep(indice_2):
            free_energy_cv_2, log_likelihood_cv_2, entropy_cv_2, prior_cv_2 = model_2.free_energy(dataset,return_components=True)
            evidence_cv_2 = model_2.evidence(dataset)

        with open(f'{analysis_dir}free_energy_cv_1.json',"w") as file:
            metrics_cv_1 = {'free_energy': float(free_energy_cv_1),
                            'log_likelihood': float(log_likelihood_cv_1),
                            'entropy': float(entropy_cv_1),
                            'prior': float(prior_cv_1),
                            'evidence': float(evidence_cv_1),
                            }
            json.dump(metrics_cv_1, file)

        with open(f'{analysis_dir}free_energy_cv_2.json',"w") as file:
            metrics_cv_2 = {'free_energy': float(free_energy_cv_2),
                            'log_likelihood': float(log_likelihood_cv_2),
                            'entropy': float(entropy_cv_2),
                            'prior': float(prior_cv_2),
                            'evidence': float(evidence_cv_2),
                            }
            json.dump(metrics_cv_2, file)









