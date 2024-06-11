"""
Functions for batch training

In some use cases, e.g. compare and evaluate different models or hyperparameters,
we need to train many models and submit them to cluster using batch
This module contains useful functions to initialise proper batch training
See ./config_train_prototype.yaml for an example batch file.
"""
import os
import random
import pickle
import time
import json
from itertools import product

import yaml
import numpy as np
import pandas as pd
from .pipeline import run_pipeline_from_file
from ..data.base import Data
from ..evaluate.cross_validation import CVBase, CVHMM, CVSWC
from ..utils.misc import override_dict_defaults
from ..utils.plotting import plot_box


class IndexParser:
    """
    Parse the training config file with index for batch training.
    Typically, a root config YAML file looks like the following, where
    batch_variable contains list of variables for different configurations
    non_batch_variable contains all other hyperparameters for training
    Given a training index, we need to find the specific batch_variable
    and combine that with other non_batch variables
    header:
  # We assign a time stamp for each batch training
  time: 2024-02-02T16:05:00.000Z
  # Add custom notes, which will also be saved
  note: "test whether your yaml file works"
# where to read the data
load_data:
  inputs: './data/node_timeseries/simulation_202402/sigma_0.1/'
  prepare:
    select:
      timepoints:
        - 0
        - 1200
    standardize: {}
# Where to save model training results
save_dir: './results_yaml_test/'
# where to load the spatial map and spatial surface map
spatial_map: './data/spatial_maps/'
non_batch_variable:
  n_channels: 25
  sequence_length: 600
  learn_means: false
  learn_covariances: true
  learn_trans_prob: true
  learning_rate: 0.01
  n_epochs: 30
  split_strategy: random
  init_kwargs:
    n_init: 10
    n_epochs: 2
# The following variables have lists for batch training
batch_variable:
  model:
    - 'hmm'
    - 'dynemo'
    - 'mdyenmo'
    - 'swc'
  n_states: [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
  # Mode can be: train, repeat, split, cross_validation
  mode:
    - train
    - repeat_1
    - repeat_2
    - repeat_3
    - repeat_4
    - repeat_5
    - split_1
    - split_2
    - split_3
    - split_4
    - split_5
    - cv_1
    - cv_2
    - cv_3
    - cv_4
    - cv_5
    """

    def __init__(self, config: dict):

        # Sleep for random seconds, otherwise the batch job might contradict
        time.sleep(random.uniform(0., 2.))

        self.save_dir = config['save_dir']
        self.batch_variable = config['batch_variable']
        self.non_batch_variable = config['non_batch_variable']
        self.other_keys = {key: value for key, value in config.items()
                           if key not in ['batch_variable', 'non_batch_variable']}

        # Check the number of channels is set correctly.
        if 'cv_kwargs' in self.other_keys:
            assert self.non_batch_variable['n_channels'] == self.other_keys['cv_kwargs']['n_channels']

        # Check whether the save_dir exists, make directory if not
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        # Check whether the root configuration file exists, save if not
        if not os.path.exists(f'{self.save_dir}config_root.yaml'):
            with open(f'{self.save_dir}config_root.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

        # Check if the config list file exists, create if not
        if not os.path.exists(f'{self.save_dir}config_list.csv'):
            self._make_list()

    def parse(self, index: int = 0):
        """
        Given the index, parse the correct configuration file
        Parameters
        ----------
        index: the index passed in from batch.

        Returns
        -------
        config: dict
          the configuration file given the index
        """
        # Read in the list
        config_list = pd.read_csv(f'{self.save_dir}config_list.csv', index_col=0)

        # sv represents batch_variable given specific row
        bv = config_list.iloc[index].to_dict()

        # concatenate three parts of the dictionary
        new_config = {}
        new_config.update(self.other_keys)
        new_config.update(bv)
        new_config.update(self.non_batch_variable)

        root_save_dir = new_config['save_dir']

        new_config['save_dir'] = f'{new_config["save_dir"]}{new_config["model"]}' \
                                 f'_ICA_{new_config["n_channels"]}_state_{new_config["n_states"]}/{new_config["mode"]}/'

        # Deal with cross validation case
        if 'cv' in new_config['mode']:
            # Update the new_config['cv_kwargs']
            new_config['cv_kwargs'] = {
                'row_indices': f'{root_save_dir}/{new_config["mode"]}_partition/row_indices.npz',
                'column_indices': f'{root_save_dir}/{new_config["mode"]}_partition/column_indices.npz'
            }
            if 'row_fold' in new_config.keys() and 'column_fold' in new_config.keys():
                new_config[
                    'save_dir'] = f"{new_config['save_dir']}/fold_{new_config['row_fold']}_{new_config['column_fold']}/"
        return new_config

    def _make_list(self):
        """
        Make the list of batch variables with respect to index,
        and save them to f'{self.header["save_dir"]}config_list.xlsx'
        Returns
        -------
        """

        from itertools import product

        # Deal with cross validation
        cv_count = sum(1 for item in self.batch_variable['mode'] if 'cv' in item)
        if cv_count > 0:
            cv_kwargs = self.other_keys['cv_kwargs']

            for i in range(cv_count):
                cv_kwargs['save_dir'] = f'{self.save_dir}/cv_{i + 1}_partition/'
                cv = CVBase(**cv_kwargs)
            self.batch_variable['row_fold'] = list(range(1, cv.partition_rows + 1))
            self.batch_variable['column_fold'] = list(range(1, cv.partition_columns + 1))
        combinations = list(product(*self.batch_variable.values()))
        # Create a DataFrame
        df = pd.DataFrame(combinations, columns=self.batch_variable.keys())
        df.to_csv(f'{self.save_dir}config_list.csv', index=True)


class BatchTrain:
    """
    Convert a batch training configuration file to another config
    for training pipeline
    """
    mode_key_default = 'mode'

    train_keys_default = ['n_channels',
                          'n_states',
                          'learn_means',
                          'learn_covariances',
                          'learn_trans_prob',
                          'window_length',
                          'window_offset',
                          'initial_means',
                          'initial_covariances',
                          'initial_trans_prob',
                          'sequence_length',
                          'batch_size',
                          'learning_rate',
                          'n_epochs',
                          ]

    def __init__(self, config: dict, train_keys=None):
        self.train_keys = self.train_keys_default if train_keys is None else train_keys

        # Validate the configuration file
        if 'load_data' not in config:
            raise ValueError('No data directory specified!')
        # The default mode of 'mode' is train
        if 'mode' not in config:
            config['mode'] = 'train'
        if 'init_kwargs' not in config:
            config['init_kwargs'] = {}
        # Check whether save directory is specified
        if 'save_dir' not in config:
            raise ValueError('Saving directory not specified!')

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        if not os.path.isfile(f'{config["save_dir"]}batch_config.yaml'):
            with open(f'{config["save_dir"]}batch_config.yaml', 'w') as file:
                yaml.safe_dump(config, file, default_flow_style=False)
        self.config = config

    def model_train(self, cv_ratio=0.25):
        '''
        Batch model train method
        cv_ration: float,optional
           the proportion of sessions to use as the training data
        Returns
        -------
        '''
        prepare_config = {}
        prepare_config['load_data'] = self.config['load_data']

        prepare_config[f'train_{self.config["model"]}'] = {
            'config_kwargs':
                {key: self.config[key] for key in self.train_keys if key in self.config},
            'init_kwargs':
                self.config['init_kwargs']
        }

        if "split" in self.config["mode"]:
            # We need to know how many sessions in advance
            indice_1, indice_2 = self.select_indice()

            # Save the selected and remaining indices to JSON files
            with open(f'{self.config["save_dir"]}indices_1.json', 'w') as json_file:
                json.dump(indice_1, json_file)
            with open(f'{self.config["save_dir"]}indices_2.json', 'w') as json_file:
                json.dump(indice_2, json_file)

            for i in range(0, 2):
                temp_save_dir = f'{self.config["save_dir"]}half_{i + 1}/'
                if not os.path.exists(temp_save_dir):
                    os.makedirs(temp_save_dir)
                prepare_config['keep_list'] = f'{self.config["save_dir"]}indices_{i + 1}.json'
                with open(f'{temp_save_dir}prepared_config.yaml', 'w') as file:
                    yaml.safe_dump(prepare_config, file, default_flow_style=False)
                run_pipeline_from_file(f'{temp_save_dir}prepared_config.yaml',
                                       temp_save_dir)


        elif "cv" in self.config["mode"]:
            self.config['train_keys'] = self.train_keys
            if self.config['model'] == 'hmm':
                cv = CVHMM(**self.config['cv_kwargs'])
            elif self.config['model'] == 'swc':
                cv = CVSWC(**self.config['cv_kwargs'])
            cv.validate(self.config, self.config['row_fold'], self.config['column_fold'])
            '''
            indice_all = self.select_indice(ratio=cv_ratio)

            # Save the selected and remaining indices to JSON files
            for i in range(len(indice_all)):
                with open(f'{self.config["save_dir"]}indices_{i+1}.json', 'w') as json_file:
                    json.dump(indice_all[i], json_file)

            for i in range(0,2):
                temp_save_dir = f'{self.config["save_dir"]}half_{i+1}/'
                if not os.path.exists(temp_save_dir):
                    os.makedirs(temp_save_dir)
                prepare_config['keep_list'] = f'{self.config["save_dir"]}indices_{i+1}.json'
                with open(f'{temp_save_dir}prepared_config.yaml', 'w') as file:
                    yaml.safe_dump(prepare_config, file, default_flow_style=False)
                run_pipeline_from_file(f'{temp_save_dir}prepared_config.yaml',
                                      temp_save_dir)

            
            prepare_config['keep_list'] = f'{self.config["save_dir"]}indices_train.json'
            with open(f'{self.config["save_dir"]}prepared_config.yaml', 'w') as file:
                yaml.safe_dump(prepare_config, file, default_flow_style=False)
            run_pipeline_from_file(f'{self.config["save_dir"]}prepared_config.yaml',
                                   self.config['save_dir'])
            '''

        else:
            with open(f'{self.config["save_dir"]}prepared_config.yaml', 'w') as file:
                yaml.safe_dump(prepare_config, file, default_flow_style=False)
            run_pipeline_from_file(f'{self.config["save_dir"]}prepared_config.yaml',
                                   self.config["save_dir"])

    def select_indice(self, ratio=0.5):
        if "n_sessions" not in self.config:
            data = Data(self.config["load_data"]["inputs"])
            n_sessions = len(data.arrays)
        else:
            n_sessions = self.config["n_sessions"]

        all_indices = list(range(n_sessions))

        # Check if the ratio is 0.5
        if ratio == 0.5:
            # Randomly select indices without replacement
            selected_indices = random.sample(all_indices, int(n_sessions * ratio))
            # Calculate the remaining indices
            remaining_indices = list(set(all_indices) - set(selected_indices))
            return selected_indices, remaining_indices
        elif ratio == 0.25:
            # Randomly split indices into four chunks
            random.shuffle(all_indices)
            chunk_size = int(n_sessions * ratio)
            chunks = [all_indices[i:i + chunk_size] for i in range(0, n_sessions, chunk_size)]
            return chunks


def batch_check(config: dict):
    '''
    Check whether the batch training is successful, raise value Error
    and save the list if some batch training is not successful.
    Parameters
    ----------
    config: str
        configuration file of batch training
    '''
    # check the bad directories
    bad_dirs = []

    # Check whether the fail training list exists, delete if so.
    bad_dirs_save_path = f'{config["save_dir"]}failure_list.yaml'

    # Check if the file exists, delete if so
    if os.path.exists(bad_dirs_save_path):
        os.remove(bad_dirs_save_path)

    for values in product(*config['batch_variable'].values()):
        combination = dict(zip(config['batch_variable'].keys(), values))
        vars = override_dict_defaults(config['non_batch_variable'], combination)
        check_dir = f'{config["save_dir"]}{vars["model"]}_ICA' \
                    f'_{vars["n_channels"]}_state_{vars["n_states"]}/{vars["mode"]}/'

        # Check whether batch_config exists
        if not os.path.isfile(f'{check_dir}batch_config.yaml'):
            bad_dirs.append(check_dir)
        # Check whether prepared_config exists
        try:
            if "split" in vars['mode']:
                check_dir = f'{check_dir}half_2/'
            assert os.path.isfile(f'{check_dir}prepared_config.yaml')
            # Check whether model training is successful
            assert os.path.exists(f'{check_dir}model')
            # Check if the covs.npy file exists
            assert os.path.isfile(f'{check_dir}inf_params/covs.npy')
            # Check if the means.npy file exists
            assert os.path.isfile(f'{check_dir}inf_params/means.npy')
            # Check whether the alp.pkl exists
            assert os.path.isfile(f'{check_dir}inf_params/alp.pkl')
        except AssertionError:
            bad_dirs.append(check_dir)

    if len(bad_dirs) > 0:
        # Serialize and save the list to a file
        with open(bad_dirs_save_path, 'w') as file:
            yaml.safe_dump(bad_dirs, file, default_flow_style=False)
        raise ValueError(f'Some training cases failed, check {bad_dirs_save_path} for the list')
    else:
        print('All model training successful!')


class BatchAnalysis:
    '''
    Analysis code after batch training. The config path in initialisation should contain
    :code:`config_root.yaml` and :code:`config_list.csv`
    '''

    def __init__(self, config_path):
        self.config_path = config_path
        with open(os.path.join(config_path, 'config_root.yaml'), 'r') as file:
            self.config_root = yaml.safe_load(file)
        self.indexparser = IndexParser(self.config_root)
        self.config_list = pd.read_csv(os.path.join(config_path, 'config_list.csv'), index_col=0)
        self.analysis_path = os.path.join(config_path, 'analysis')
        if not os.path.exists(self.analysis_path):
            os.makedirs(self.analysis_path)

    def compare(self, demean=False, demean_index=-1,inset_start_index=None,folder='Y_test/',object='log_likelihood'):
        '''
        By default of bi-cross validation, we should compare the final log_likelihood on the Y_test.
        But for sanity check, and potentiall understand how the method work, we are also interested in
        the folder Y_train/metrics, X_test/metrics.
        '''
        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        metrics = {model: {str(int(num)): [] for num in n_states} for model in models}
        for i in range(len(self.config_list)):
            config = self.indexparser.parse(i)
            model = config['model']
            n_states = config['n_states']
            save_dir = config['save_dir']
            mode = config['mode']
            if 'cv' in mode:
                try:
                    with open(os.path.join(save_dir,folder, 'metrics.json'), 'r') as file:
                        metric = json.load(file)[object]
                    metrics[model][str(int(n_states))].append(metric)
                except Exception:
                    print(f'save_dir {save_dir} fails!')
                    metrics[model][str(int(n_states))].append(np.nan)

        # Plot
        for model in models:
            temp_keys = list(metrics[model].keys())
            temp_values = [metrics[model][key] for key in temp_keys]
            plot_box(data=temp_values,
                     labels=temp_keys,
                     demean=demean,
                     demean_index=demean_index,
                     x_label=r'$N_{states}$',
                     y_label='Demeaned log likelihood',
                     inset_start_index=inset_start_index,
                     filename=os.path.join(self.analysis_path, f'{model}_{folder.split("/")[0]}_{object}.pdf')
                     )


    def temporal_analysis(self,demean=False, inset_start_index=None,theme='reproducibility',normalisation=False):
        if theme == 'reproducibility':
            directory_list = [['fold_1_1/X_train/inf_params/alp.pkl','fold_1_2/X_train/inf_params/alp.pkl'],
                              ['fold_2_1/X_train/inf_params/alp.pkl','fold_2_2/X_train/inf_params/alp.pkl']]
        elif theme == 'compromise':
            directory_list = [['fold_1_1/X_train/inf_params/alp.pkl','fold_2_1/Y_test/inf_params/alp.pkl'],
                              ['fold_1_2/X_train/inf_params/alp.pkl','fold_2_2/Y_test/inf_params/alp.pkl'],
                              ['fold_2_1/X_train/inf_params/alp.pkl', 'fold_1_1/Y_test/inf_params/alp.pkl'],
                              ['fold_2_2/X_train/inf_params/alp.pkl', 'fold_1_2/Y_test/inf_params/alp.pkl']]
        elif theme == 'fixed':
            directory_list = [['fold_1_1/X_train/inf_params/alp.pkl', 'fold_2_2/Y_test/inf_params/alp.pkl'],
                              ['fold_1_2/X_train/inf_params/alp.pkl', 'fold_2_1/Y_test/inf_params/alp.pkl'],
                              ['fold_2_1/X_train/inf_params/alp.pkl', 'fold_1_2/Y_test/inf_params/alp.pkl'],
                              ['fold_2_2/X_train/inf_params/alp.pkl', 'fold_1_1/Y_test/inf_params/alp.pkl']]
        else:
            raise ValueError('Invalid theme presented!')

        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        modes = self.config_root['batch_variable']['mode']
        modes = [mode for mode in modes if 'cv' in mode]
        metrics = {model: {str(int(num)): [] for num in n_states} for model in models}
        temporal_directory = os.path.join(self.analysis_path, 'temporal_analysis')
        if not os.path.exists(temporal_directory):
            os.makedirs(temporal_directory)
        for model in models:
            for n_state in n_states:
                for mode in modes:
                    save_dir = (f"{self.config_root['save_dir']}/"
                                f"{model}_ICA_{self.config_root['non_batch_variable']['n_channels']}_state_{n_state}/"
                                f"{mode}/")
                    count = 1
                    for directory in directory_list:
                        try:
                            temp = self._temporal_reproducibility(
                                os.path.join(save_dir,directory[0]),
                                os.path.join(save_dir,directory[1]),
                                n_states = n_state,
                                normalisation=normalisation,
                                filename=os.path.join(temporal_directory,
                                    f"{model}_{n_state}_{mode}_{theme}_{count}.jpg"))
                            count += 1
                            metrics[model][str(int(n_state))].append(temp)
                        except Exception:
                            print(f'Case {model} {n_state} {mode} {theme} fails!')
                            metrics[model][str(int(n_state))].append(np.nan)
        # Save the dictionary to a file
        with open(os.path.join(self.analysis_path, f'{model}_temporal_analysis_{theme}.pkl'), 'wb') as file:
            pickle.dump(metrics, file)
        for model in models:
            temp_keys = list(metrics[model].keys())
            temp_values = [metrics[model][key] for key in temp_keys]
            plot_box(data=temp_values,
                     labels=temp_keys,
                     demean=demean,
                     inset_start_index=inset_start_index,
                     filename=os.path.join(self.analysis_path, f'{model}_temporal_analysis_{theme}{"_norm" if normalisation else ""}.jpg')
                     )

    def spatial_analysis(self,demean=False,inset_start_index=None,theme='reproducibility',normalisation=False):
        if theme == 'reproducibility':
            directory_list = [['fold_1_1/Y_train/inf_params/covs.npy','fold_2_1/Y_train/inf_params/covs.npy'],
                              ['fold_1_2/Y_train/inf_params/covs.npy','fold_2_2/Y_train/inf_params/covs.npy']]
        elif theme == 'fixed':
            directory_list = [['fold_1_1/Y_train/inf_params/covs.npy','fold_1_2/X_train/dual_estimates/covs.npy'],
                              ['fold_1_2/Y_train/inf_params/covs.npy','fold_1_1/X_train/dual_estimates/covs.npy'],
                              ['fold_2_1/Y_train/inf_params/covs.npy','fold_2_2/X_train/dual_estimates/covs.npy'],
                              ['fold_2_2/Y_train/inf_params/covs.npy','fold_2_1/X_train/dual_estimates/covs.npy']]
        else:
            raise ValueError('Invalid theme presented!')

        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        modes = self.config_root['batch_variable']['mode']
        modes = [mode for mode in modes if 'cv' in mode]
        metrics = {model: {str(int(num)): [] for num in n_states} for model in models}
        spatial_directory = os.path.join(self.analysis_path,'spatial_analysis')
        if not os.path.exists(spatial_directory):
            os.makedirs(spatial_directory)
        for model in models:
            for n_state in n_states:
                for mode in modes:
                    save_dir = (f"{self.config_root['save_dir']}/"
                                f"{model}_ICA_{self.config_root['non_batch_variable']['n_channels']}_state_{n_state}/"
                                f"{mode}/")
                    count = 1
                    for directory in directory_list:
                        try:
                            temp = self._spatial_reproducibility(
                                os.path.join(save_dir, directory[0]),
                                os.path.join(save_dir, directory[1]),
                                normalisation=normalisation,
                                filename=os.path.join(spatial_directory,
                                                      f"{model}_{n_state}_{mode}_{theme}_{count}.jpg"))
                            count += 1
                            metrics[model][str(int(n_state))].append(temp)
                        except Exception:
                            print(f'Case {model} {n_state} {mode} {theme} fails!')
                            metrics[model][str(int(n_state))].append(np.nan)
        for model in models:
            temp_keys = list(metrics[model].keys())
            temp_values = [metrics[model][key] for key in temp_keys]
            plot_box(data=temp_values,
                     labels=temp_keys,
                     demean=demean,
                     inset_start_index=inset_start_index,
                     filename=os.path.join(self.analysis_path, f'{model}_spatial_analysis_{theme}{"_norm" if normalisation else ""}.jpg')
                     )


    def plot_training_loss(self, metrics=['free_energy']):
        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        loss = {metric: {model: {str(int(num)): [] for num in n_states} for model in models} for metric in metrics}
        for i in range(len(self.config_list)):
            config = self.indexparser.parse(i)
            model = config['model']
            n_states = config['n_states']
            mode = config['mode']
            save_dir = config['save_dir']
            if 'repeat' in mode:
                try:
                    with open(f'{save_dir}/metrics/metrics.json',"r") as file:
                        data = json.load(file)
                    for metric in metrics:
                        loss[metric][model][str(int(n_states))].append(data[metric])
                except Exception:
                    print(f'save_dir {save_dir} fails!')
                    loss[metric][model][str(int(n_states))].append(np.nan)

        # Plot
        for metric in metrics:
            for model in models:
                temp_keys = list(loss[metric][model].keys())
                temp_values = [loss[metric][model][key] for key in temp_keys]
                plot_box(data=temp_values,
                         labels=temp_keys,
                         mark_best=False,
                         demean=False,
                         x_label='N_states',
                         y_label=metric,
                         title=f'{metric} VS N_states',
                         filename=os.path.join(self.analysis_path, f'{model}_{metric}.jpg')
                        )

    def plot_split_half_reproducibility(self):
        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        rep = {model: {str(int(num)): [] for num in n_states} for model in models}
        rep_path = os.path.join(self.analysis_path,'rep')
        if not os.path.exists(rep_path):
            os.makedirs(rep_path)
        for i in range(len(self.config_list)):
            config = self.indexparser.parse(i)
            model = config['model']
            n_states = config['n_states']
            save_dir = config['save_dir']
            mode = config['mode']
            if 'split' in mode:
                try:
                    cov_1 = np.load(f'{save_dir}/half_1/inf_params/covs.npy')
                    cov_2 = np.load(f'{save_dir}/half_2/inf_params/covs.npy')
                    rep[model][str(int(n_states))].append(self._reproducibility_analysis(cov_1,cov_2,
                                                          filename=os.path.join(rep_path,f'state_{n_states}_{mode}.jpg')))
                except Exception:
                    print(f'save_dir {save_dir} fails!')
                    rep[model][str(int(n_states))].append(np.nan)

        for model in models:
            temp_keys = list(rep[model].keys())
            temp_values = [rep[model][key] for key in temp_keys]
            plot_box(data=temp_values,
                     labels=temp_keys,
                     mark_best=False,
                     demean=False,
                     x_label='N_states',
                     y_label='reproducibility',
                     title=f'Average Riemannian distance VS N_states',
                     filename=os.path.join(self.analysis_path, f'{model}_reproducibility.jpg')
                     )

    def plot_fo(self, plot_mode='repeat_1'):
        from osl_dynamics.inference.modes import argmax_time_courses,fractional_occupancies
        from osl_dynamics.utils.plotting import plot_violin
        models = self.config_root['batch_variable']['model']
        n_states = self.config_root['batch_variable']['n_states']
        fo_path = os.path.join(self.analysis_path, 'fo')
        if not os.path.exists(fo_path):
            os.makedirs(fo_path)
        for i in range(len(self.config_list)):
            config = self.indexparser.parse(i)
            model = config['model']
            n_states = config['n_states']
            save_dir = config['save_dir']
            mode = config['mode']
            if plot_mode == mode:
                try:
                    with open(f'{save_dir}/inf_params/alp.pkl', "rb") as file:
                        alpha = pickle.load(file)
                    stc = argmax_time_courses(alpha)
                    fo = fractional_occupancies(stc)
                    plot_violin(fo.T, x_label="State", y_label="FO",title=f'Fractional Occupancy, {n_states} states',
                                filename=os.path.join(fo_path,f'state_{n_states}.jpg'))


                except Exception:
                    print(f'save_dir {save_dir} fails!')

    def _reproducibility_analysis(self,cov_1,cov_2,filename=None):
        if not os.path.exists(os.path.join(self.analysis_path,'rep')):
            os.makedirs(os.path.join(self.analysis_path,'rep'))
        from osl_dynamics.inference.metrics import twopair_riemannian_distance
        from osl_dynamics.inference.modes import hungarian_pair
        from osl_dynamics.utils.plotting import plot_mode_pairing
        riem = twopair_riemannian_distance(cov_1,cov_2)
        indice,riem_reorder = hungarian_pair(riem,distance=True)
        plot_mode_pairing(riem_reorder,indice,x_label='2nd half',y_label='1st half',
                          filename=filename)
        return np.mean(np.diagonal(riem_reorder))

    def _spatial_reproducibility(self,cov_1,cov_2,normalisation=False,filename=None):
        from osl_dynamics.inference.metrics import twopair_riemannian_distance
        from osl_dynamics.inference.modes import hungarian_pair
        from osl_dynamics.utils.plotting import plot_mode_pairing
        if isinstance(cov_1,str):
            cov_1 = np.load(cov_1)
        if isinstance(cov_2,str):
            cov_2 = np.load(cov_2)
        riem = twopair_riemannian_distance(cov_1, cov_2)
        indice, riem_reorder = hungarian_pair(riem, distance=True)
        plot_mode_pairing(riem_reorder, indice, x_label='2nd half', y_label='1st half',
                          filename=filename)
        mean_diagonal = np.mean(np.diagonal(riem_reorder))
        if normalisation:
            off_diagonal_indices = np.where(~np.eye(riem_reorder.shape[0], dtype=bool))
            mean_off_diagonal = np.mean(riem_reorder[off_diagonal_indices])
            var_off_diagonal = np.var(riem_reorder[off_diagonal_indices])

            if len(riem_reorder) == 2:
                var_off_diagonal = riem_reorder[0,1] ** 2

            # Return the mean diagonal value divided by the mean off-diagonal value
            return (mean_diagonal - mean_off_diagonal) / np.sqrt(var_off_diagonal)
        else:
            return mean_diagonal

    def _temporal_reproducibility(self,alpha_1,alpha_2,n_states,normalisation=False,filename=None):
        from osl_dynamics.inference.metrics import alpha_correlation
        from osl_dynamics.inference.modes import hungarian_pair,argmax_time_courses
        from osl_dynamics.utils.plotting import plot_mode_pairing
        from osl_dynamics.array_ops import get_one_hot

        if isinstance(alpha_1,str):
            with open(alpha_1, 'rb') as file:
                alpha_1 = pickle.load(file)

        if isinstance(alpha_2,str):
            with open(alpha_2, 'rb') as file:
                alpha_2 = pickle.load(file)

        # Get one-hot coding if the original one is not
        if alpha_1[0].ndim == 1:
            alpha_1 = get_one_hot(alpha_1,n_states)
        if alpha_2[0].ndim == 1:
            alpha_2 = get_one_hot(alpha_2,n_states)
        # Argmax time courses.
        alpha_1 = argmax_time_courses(alpha_1)
        alpha_2 = argmax_time_courses(alpha_2)

        corr = alpha_correlation(alpha_1,alpha_2,return_diagonal=False)
        indice, corr_reorder = hungarian_pair(corr, distance=False)
        plot_mode_pairing(corr_reorder, indice, x_label='2nd half', y_label='1st half',
                          filename=filename)
        mean_diagonal = np.mean(np.diagonal(corr_reorder))
        if normalisation:
            off_diagonal_indices = np.where(~np.eye(corr_reorder.shape[0], dtype=bool))
            mean_off_diagonal = np.mean(corr_reorder[off_diagonal_indices])
            var_off_diagonal = np.var(corr_reorder[off_diagonal_indices])

            if n_states == 2:
                var_off_diagonal = corr_reorder[0,1] ** 2

            # Return the mean diagonal value divided by the mean off-diagonal value
            return (mean_diagonal - mean_off_diagonal) / np.sqrt(var_off_diagonal)
        else:
            return mean_diagonal








