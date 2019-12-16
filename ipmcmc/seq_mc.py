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
    weights = np.zeros((n_particles, len(observations)))

    # Initialisation
    for i in range(n_particles):
        particle = q_t[0].sample()
        particles[i][0] = particle
        weights[i][0] = np.exp(np.log(g_t[0].density(observations[0], given=[particle]))\
            + np.log(f_t[0].density(particle, given=list()))\
            - np.log(q_t[0].density(particle, given=list())))

    # Loop
    for t, observation in tqdm(enumerate(observations[1:]), total=t_max-1):  # t=time
        t_1 = t
        t += 1
        for i in range(n_particles):  # i=particle_id
            # Compute ancestor step
            p = np.exp(np.log(weights[:, t_1]) - np.log(weights[:, t_1].sum()))
            ancestor = np.random.choice(range(n_particles), p=p)
            past = particles[ancestor][0:t]
            # Generate new particle
            particle = q_t[t].sample(given=past)
            trajectory = past + [particle]
            # Compute weight
            weight = np.exp(np.log(g_t[t].density(observations[t], given=trajectory))\
            + np.log(f_t[t].density(particle, given=past))\
            - np.log(q_t[t].density(particle, given=past)))
            
            # Store results
            for old_t, elt in enumerate(trajectory):
                particles[i][old_t] = elt
            weights[i][t] = weight

    return np.array(particles), weights
