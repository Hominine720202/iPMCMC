import numpy as np

def init_proposal(sampler, params):
    return lambda n: sampler(**params, size=n)


def gaussian_density(mu, var):
    assert (type(mu) == np.ndarray and var.shape == (len(mu), len(mu))) or (
        type(float(mu)) == float and type(float(var)) == float)
    def dens_func(x):
        if type(mu) == np.ndarray:
            N = len(mu)
            return 1/np.sqrt((2*np.pi)**N * np.linalg.det(var)) * np.exp(-1/2 * np.transpose(x - mu) @ np.linalg.inv(var) @ (x - mu))
        return 1/np.sqrt(2*np.pi * var) * np.exp(-1/(2*var) * (x - mu)**2)
    return dens_func


def gaussian_density_var(var, factor=1):
    assert (type(var) == np.ndarray and var.shape[0] == factor.shape[0]) or (
        type(float(var)) == float and type(float(factor)) == float)

    def dens_func(x, mu):
        mu = mu[-1]
        
        if type(mu) == np.ndarray:
            N = len(mu)
            return 1/np.sqrt((2*np.pi)**N * np.linalg.det(var)) * np.exp(-1/2 * np.transpose(x - factor@mu) @ np.linalg.inv(var) @ (x - factor@mu))
        return 1/np.sqrt(2*np.pi * var) * np.exp(-1/(2*var) * (x - factor*mu)**2)
    return dens_func
