import sys
import pickle
from osl_dynamics.config_api.wrappers import load_data
if __name__ == '__main__':
    data_kwargs_path = sys.argv[1]
    with open(data_kwargs_path,'rb') as file:
        data_kwargs = pickle.load(file)
    data = load_data(**data_kwargs)
    print(len(data.time_series()))
    print(data.time_series[0].shape)
