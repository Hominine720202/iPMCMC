# 4.2. Nonlinear State Space Model
import numpy as np
from .distribution import Distribution

# TODO: Clean kwargs handling

class NonLinearMu(Distribution):
    def __init__(self, default_mean, default_std):
        super().__init__()
        from scipy.stats import norm
        self.distribution = norm
        self.default_mean = default_mean
        self.default_std = default_std

    def rvs(self, given=None, **kwargs):
        if kwargs.setdefault('size', 1) == 1:
            return self.distribution.rvs(
                loc=self.default_mean,
                scale=self.default_std,
                **kwargs)
        return self.distribution.rvs(
            loc=self.default_mean,
            scale=self.default_std,
            **kwargs)[:, np.newaxis]

    def logpdf(self, x, given=None, **kwargs):
        return self.distribution.logpdf(
            x,
            loc=self.default_mean,
            scale=self.default_std,
            **kwargs
        )


class NonLinearTransition(Distribution):
    def __init__(self, default_mean, default_std):
        super().__init__()
        from scipy.stats import norm
        self.distribution = norm
        self.default_mean = default_mean
        self.default_std = default_std

    def rvs(self, given=None, **kwargs):
        if kwargs.setdefault('size', 1)==1:
            if isinstance(given, type(None)) or isinstance(given, list) and len(given) == 0:
                return self.distribution.rvs(
                    loc=self.default_mean,
                    scale=self.default_std,
                    **kwargs)
            return self.distribution.rvs(
                loc=8*np.cos(1.2*len(given)) *
                (given[-1]/2+25*(given[-1]/(1+given[-1]**2))+self.default_mean),
                scale=np.abs(8*np.cos(1.2*len(given)))*self.default_std,
                **kwargs)

        if isinstance(given, type(None)) or isinstance(given, list) and len(given) == 0:
            return self.distribution.rvs(
                loc=self.default_mean,
                scale=self.default_std,
                **kwargs)[:,np.newaxis]
        return self.distribution.rvs(
            loc=8*np.cos(1.2*len(given)) *
            (given[-1]/2+25*(given[-1]/(1+given[-1]**2))+self.default_mean),
            scale=np.abs(8*np.cos(1.2*len(given)))*self.default_std,
            **kwargs)[:, np.newaxis]

    def logpdf(self, x,  given=None, **kwargs):
        if isinstance(given, type(None)) or isinstance(given, list) and len(given) == 0:
            return self.distribution.logpdf(
                x,
                loc=self.default_mean,
                scale=self.default_std,
                **kwargs)
        return self.distribution.logpdf(
            x,
            loc=8*np.cos(1.2*len(given)) *
            (given[-1]/2+25*(given[-1]/(1+given[-1]**2))+self.default_mean),
            scale=np.abs(8*np.cos(1.2*len(given)))*self.default_std,
            **kwargs)


class NonLinearObservation(Distribution):
    def __init__(self, default_mean, default_std):
        super().__init__()
        from scipy.stats import norm
        self.distribution = norm
        self.default_mean = default_mean
        self.default_std = default_std

    def rvs(self, given=None, **kwargs):
        if kwargs.setdefault('size', 1)==1:
            if isinstance(given, type(None)) or isinstance(given, list) and len(given) == 0:
                return self.distribution.rvs(
                    loc=self.default_mean,
                    scale=self.default_std,
                    **kwargs)
            return self.distribution.rvs(
                loc=(given[-1]**2)/20 + self.default_mean,
                scale=self.default_std,
                **kwargs)
        if isinstance(given, type(None)) or isinstance(given, list) and len(given) == 0:
            return self.distribution.rvs(
                loc=self.default_mean,
                scale=self.default_std,
                **kwargs)[:,np.newaxis]
        return self.distribution.rvs(
            loc=(given[-1]**2)/20 + self.default_mean,
            scale=self.default_std,
            **kwargs)[:,np.newaxis]


    def logpdf(self, x,  given=None, **kwargs):
        if isinstance(given, type(None)) or isinstance(given, list) and len(given) == 0:
            return self.distribution.logpdf(
                x,
                loc=self.default_mean,
                scale=self.default_std,
                **kwargs)
        return self.distribution.logpdf(
            x,
            loc=(given[-1]**2)/20 + self.default_mean,
            scale=self.default_std,
            **kwargs)


class NonLinearProposal(NonLinearTransition):
    pass


# TODO: Refactor to take Distribution objects into account
def nonlinear_gaussian_state_space(t_max, mu,
                                   noise_std, transition_std, start_std):
    transition = np.random.normal(
        0, transition_std, t_max)
    noise = np.random.normal(
        0, noise_std, t_max)

    state = np.random.normal(mu, start_std)
    observation = (state**2)/20 + noise[0]

    states, observations = [state], [observation]
    for i in range(1, t_max):
        state = state/2 + 25*(state/(1+state**2)) + 8 * \
            np.cos(1.2*(i+1))*transition[i]
        observation = (state**2)/20 + noise[i]
        states.append(state)
        observations.append(observation)
    return np.array(states), np.array(observations)
