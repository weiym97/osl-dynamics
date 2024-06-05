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
    import yaml
    from osl_dynamics.config_api.wrappers import train_swc
    from osl_dynamics.config_api.pipeline import run_pipeline_from_file

    save_dir = './test_train_swc/'
    data_dir = f'{save_dir}/data/'
    output_dir = f'{save_dir}/result/'
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


