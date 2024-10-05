import sys
import yaml

import numpy as np
import statsmodels.api as sm

from osl_dynamics.config_api.wrappers import load_data


def fit_ar1_model(time_series):
    # Fit an AR(1) model to the time series using ARIMA with order (1,0,0)
    model = sm.tsa.ARIMA(time_series, order=(1, 0, 0)).fit()
    return model.params


if __name__ == '__main__':
    data_kwargs_path = sys.argv[1]
    save_path = sys.argv[2]
    with open(data_kwargs_path, 'r') as file:
        data_kwargs = yaml.safe_load(file)
    data = load_data(**data_kwargs)
    data_time_series = data.time_series()

    n_subjects = len(data_time_series)
    n_timepoints, n_channels = data_time_series[0].shape

    # To store AR(1) coefficients for each subject and channel
    ar1_intercepts = np.zeros((n_subjects, n_channels))
    ar1_coefficients = np.zeros((n_subjects, n_channels))
    ar1_noises = np.zeros((n_subjects, n_channels))

    # Iterate through subjects and channels
    for subject in range(n_subjects):
        for channel in range(n_channels):
            time_series = np.squeeze(data_time_series[subject][:, channel])
            params = fit_ar1_model(time_series)
            ar1_intercepts[subject, channel] = params[0]
            ar1_coefficients[subject, channel] = params[1]  # AR(1) coefficient
            ar1_noises[subject, channel] = params[2]
    np.save(f'{save_path}/ar1_intercepts.npy', ar1_intercepts)
    np.save(f'{save_path}/ar1_coefficients.npy', ar1_coefficients)
    np.save(f'{save_path}/ar1_noises.npy', ar1_noises)
