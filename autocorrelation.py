import sys
import yaml

import numpy as np

from osl_dynamics.config_api.wrappers import load_data

if __name__ == '__main__':
    data_kwargs_path = sys.argv[1]
    with open(data_kwargs_path,'r') as file:
        data_kwargs = yaml.safe_load(file)
    data = load_data(**data_kwargs)
    print(len(data.time_series()))
    print(data.time_series[0].shape)
    print(np.var(data.time_series[0], axis=0))

    data_time_series = data.time_series()

