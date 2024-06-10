import numpy as np
import numpy.testing as npt
from osl_dynamics.config_api.wrappers import load_data

def test_load_data():
    import os
    import json
    save_dir = './test_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vector = np.array([-1.5 ** 0.5, 0, 1.5 ** 0.5])
    input_1 = np.array([vector, vector + 10.0]).T
    input_2 = np.array([vector * 0.5 + 1., vector * 100]).T
    np.savetxt(f'{save_dir}10001.txt', input_1)
    np.savetxt(f'{save_dir}10002.txt', input_2)
    prepare = {'standardize':{}}

    data = load_data(inputs=save_dir,prepare=prepare)
    npt.assert_almost_equal(data.arrays[0],np.array([vector,vector]).T)
    npt.assert_almost_equal(data.arrays[1],np.array([vector,vector]).T)
    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)

def test_train_swc():
    import os
    import pickle
    import shutil
    import yaml
    from osl_dynamics.config_api.wrappers import train_swc
    from osl_dynamics.config_api.pipeline import run_pipeline_from_file

    save_dir = './test_train_swc/'
    data_dir = f'{save_dir}/data/'
    output_dir = f'{save_dir}/result/'

    if os.path.isdir(save_dir):
        # Remove the directory
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)

    config = f"""
            load_data:
                inputs: {data_dir}
            train_swc:
                config_kwargs:
                    n_states: 3
                    learn_means: False
                    learn_covariances: True
                    window_length: 5
                    window_offset: 3
                """
    with open(f'{output_dir}train_config.yaml', "w") as file:
        yaml.safe_dump(yaml.safe_load(config), file, default_flow_style=False)
    run_pipeline_from_file(f'{output_dir}train_config.yaml', output_dir)

    # After running, check the files in the directory
    inf_params_dir = f'{output_dir}/inf_params/'
    means = np.load(f'{inf_params_dir}/means.npy')
    covs = np.load(f'{inf_params_dir}/covs.npy')
    with open(f'{inf_params_dir}/alp.pkl','rb') as file:
        labels = pickle.load(file)

    covs_answer = np.array([[[2.5, -2.5], [-2.5, 2.5]],
                            [[52., 48.], [48., 52.]],
                            [[54.5, 47.75], [47.75, 54.5]]])
    labels_answer = [np.array([0, 1]), np.array([1, 2])]

    # Reorder and covs and update the labels accordingly
    order = np.argsort(covs[:, 0, 0])
    reordered_covs = covs[order]
    label_mapping = {old: new for new, old in enumerate(order)}
    reordered_labels = [np.array([label_mapping[label] for label in session_labels]) for session_labels in labels]

    npt.assert_array_equal(means,np.zeros((3,2)))
    npt.assert_array_equal(covs_answer, reordered_covs)
    npt.assert_array_equal(labels_answer, reordered_labels)


