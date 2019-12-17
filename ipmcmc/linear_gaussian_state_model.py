# 4.1. Linear Gaussian State Space Model
import numpy as np
from ipmcmc.distribution import Distribution


class LinearMu(Distribution):
    def __init__(self, default_mean, default_cov):
        super().__init__()
        from scipy.stats import multivariate_normal
        self.distribution = multivariate_normal
        self.default_mean = default_mean
        self.default_cov = default_cov

    def rvs(self, given=None, **kwargs):
        return self.distribution.rvs(
            mean=self.default_mean,
            cov=self.default_cov,
            **kwargs)

    def logpdf(self, x, given=None, **kwargs):
        return self.distribution.logpdf(
            x,
            mean=self.default_mean,
            cov=self.default_cov,
            **kwargs)


class LinearTransition(Distribution):
    def __init__(self, default_mean, default_cov, default_alpha):
        super().__init__()
        from scipy.stats import multivariate_normal
        self.distribution = multivariate_normal
        self.default_mean = default_mean
        self.default_cov = default_cov
        self.default_alpha = default_alpha

    def rvs(self, given=None, **kwargs):
        if isinstance(given, type(None)):
            raise ValueError
        elif isinstance(given, list) and len(given) == 0:
            return self.distribution.rvs(
                mean=self.default_mean,
                cov=self.default_cov,
                **kwargs)
        return self.distribution.rvs(
            mean=self.default_alpha@given[-1] + self.default_mean,
            cov=self.default_cov,
            **kwargs)

    def logpdf(self, x,  given=None, **kwargs):
        if isinstance(given, type(None)):
            raise ValueError
        elif isinstance(given, list) and len(given) == 0:
            return self.distribution.logpdf(
                x,
                mean=self.default_mean,
                cov=self.default_cov,
                **kwargs)
        return self.distribution.logpdf(
            x,
            mean=self.default_alpha@given[-1] + self.default_mean,
            cov=self.default_cov,
            **kwargs)


class LinearObservation(Distribution):
    def __init__(self, default_mean, default_cov, default_beta):
        super().__init__()
        from scipy.stats import multivariate_normal
        self.distribution = multivariate_normal
        self.default_mean = default_mean
        self.default_cov = default_cov
        self.default_beta = default_beta

    def rvs(self, given=None, **kwargs):
        if isinstance(given, type(None)):
            raise ValueError
        elif isinstance(given, list) and len(given) == 0:
            return self.distribution.rvs(
                mean=self.default_mean,
                cov=self.default_cov,
                **kwargs)
        return self.distribution.rvs(
            mean=self.default_beta@given[-1] + self.default_mean,
            cov=self.default_cov,
            **kwargs)

    def logpdf(self, x,  given=None, **kwargs):
        if isinstance(given, type(None)):
            raise ValueError
        elif isinstance(given, list) and len(given) == 0:
            return self.distribution.logpdf(
                x,
                mean=self.default_mean,
                cov=self.default_cov,
                **kwargs)
        return self.distribution.logpdf(
            x,
            mean=self.default_beta@given[-1] + self.default_mean,
            cov=self.default_cov,
            **kwargs)


class LinearProposal(LinearTransition):
    pass


# TODO: Refactor to take Distribution objects into account
def linear_gaussian_state_space(t_max,
                                mu, start_var, transition_var, noise_var,
                                transition_coeffs, observation_coeffs):

    transition = np.random.multivariate_normal(
        np.zeros(mu.shape[0]), transition_var, t_max)
    noise = np.random.multivariate_normal(
        np.zeros(noise_var.shape[0]), noise_var, t_max)

    state = np.random.multivariate_normal(mu, start_var)
    observation = observation_coeffs @ state + noise[0]

    states, observations = [state], [observation]
    for i in range(1, t_max):
        state = transition_coeffs @ state + transition[i]
        observation = observation_coeffs @ state + noise[i]
        states.append(state)
        observations.append(observation)
    return np.array(states), np.array(observations)
