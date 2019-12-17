from tqdm import tqdm
from typing import List, Callable
import numpy as np
import scipy

# Wrapper around scipy stats objects


class Distribution:
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def rvs(self, given=None, **kwargs):
        """Returns a random sample following the distribution"""
        raise NotImplementedError

    def pdf(self, x,  given=None, **kwargs):
        """Returns the value of the distribution in a given point x"""
        raise NotImplementedError

    def logpdf(self, x,  given=None, **kwargs):
        """Returns the value of the log-distribution in a given point x"""
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        return self.rvs(*args, **kwargs)
    
    def density(self, *args, **kwargs):
        return self.pdf(*args, **kwargs)

# Seqential Monte Carlo for HMM models
def seq_mc(observations: np.ndarray, 
           n_particles: int, 
           f_t: List[Distribution],
           q_t: List[Distribution],
           g_t: List[Distribution]
           ):

    t_max = len(observations)
    # We assume that the size of the state doesn't change with time
    s = q_t[0].sample()
    state_shape = s.shape if isinstance(s, np.ndarray) else ()
    particles = [[None for time in range(t_max)] for particle_idx in range(n_particles)]
    log_weights = np.zeros((n_particles, len(observations)))

    # Initialisation
    for i in range(n_particles):
        particle = q_t[0].sample()
        particles[i][0] = particle
        log_weights[i][0] = g_t[0].logpdf(observations[0], given=[particle])\
            + f_t[0].logpdf(particle, given=list())\
            - q_t[0].logpdf(particle, given=list())

    # Loop
    for t, observation in tqdm(enumerate(observations[1:]), total=t_max-1):  # t=time
        t_1 = t
        t += 1

        w_star = log_weights[:, t_1].max()
        normalisation_value = w_star + np.log(np.exp(log_weights[:, t_1] - w_star).sum())
        for i in range(n_particles):  # i=particle_id
            # Compute ancestor step
            p = np.exp(log_weights[:, t_1] - normalisation_value)
            ancestor = np.random.choice(range(n_particles), p=p)
            past = particles[ancestor][0:t]
            # Generate new particle
            particle = q_t[t].sample(given=past)
            trajectory = past + [particle]
            # Compute weight
            log_weight = g_t[t].logpdf(observations[t], given=trajectory)\
            + f_t[t].logpdf(particle, given=past)\
            - q_t[t].logpdf(particle, given=past)
            
            # Store results
            for old_t, elt in enumerate(trajectory):
                particles[i][old_t] = elt
            log_weights[i][t] = log_weight

    return np.array(particles), log_weights
