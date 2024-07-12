import numpy as np
from osl_dynamics.simulation.base import Simulation


class SWC(Simulation):
    def __init__(self, n_samples, n_states, n_channels, stay_time, means, covariances):
        super().__init__(n_samples)
        self.n_states = n_states
        self.n_channels = n_channels
        self.stay_time = stay_time
        self.means = self._initialize_means(means)
        self.covariances = covariances
        self.validate_covariances()
        self.generate_data()

    def _initialize_means(self, means):
        if means == "zero":
            return np.zeros((self.n_states, self.n_channels))
        else:
            return np.array(means)

    def validate_covariances(self):
        if not isinstance(self.covariances, np.ndarray) or self.covariances.shape != (
        self.n_states, self.n_channels, self.n_channels):
            raise ValueError("Covariances must be a numpy array of shape (n_states, n_channels, n_channels)")

    def generate_state_sequence(self):
        total_time_points = self.n_samples
        state_sequence = np.repeat(np.arange(self.n_states), self.stay_time)

        state_blocks = np.arange(self.n_states)
        np.random.shuffle(state_blocks)

        shuffled_state_sequence = np.concatenate([np.full(self.stay_time, state) for state in state_blocks])

        while len(shuffled_state_sequence) < total_time_points:
            np.random.shuffle(state_blocks)
            shuffled_state_sequence = np.concatenate(
                [shuffled_state_sequence] + [np.full(self.stay_time, state) for state in state_blocks])

        return shuffled_state_sequence[:total_time_points]

    def generate_data(self):
        total_time_points = self.n_samples
        data = np.zeros((total_time_points, self.n_channels))
        state_time_course = self.generate_state_sequence()

        for t in range(total_time_points):
            state = state_time_course[t]
            mean = self.means[state]
            cov = self.covariances[state]
            data[t, :] = np.random.multivariate_normal(mean, cov)

        self.time_series = data
        self.state_time_course = state_time_course