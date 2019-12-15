import numpy as np

def init_proposal(sampler, params):
    return lambda n: sampler(**params, size=n)


def gaussian_density(mu, var):
    assert (type(mu) == np.ndarray and var.shape == (len(mu), len(mu))) or (type(float(mu)) == float and type(float(var)) == float)
    def dens_func(x):
        if type(mu) == np.ndarray:
            N = len(mu)
            return 1/np.sqrt((2*np.pi)**N * np.linalg.det(var)) * np.exp(-1/2 * np.transpose(x - mu) @ np.linalg.inv(var) @ (x - mu))
        return 1/np.sqrt(2*np.pi * var) * np.exp(-1/(2*var) * (x - mu)**2)
    return dens_func
