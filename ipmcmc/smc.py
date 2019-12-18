from tqdm import tqdm
from typing import List, Callable
import numpy as np
import scipy
from .distribution import Distribution


# Seqential Monte Carlo for HMM models
def smc(observations: np.ndarray, 
        n_particles: int, 
        transition_model: List[Distribution],
        proposals: List[Distribution],
        observation_model: List[Distribution]
        ):
    # Values to fill
    t_max = observations.shape[0]
    particles = np.zeros((t_max, n_particles)+proposals[0].rvs().shape)
    log_weights = np.zeros((t_max, n_particles))
    ancestors = np.zeros((t_max-1, n_particles), dtype=int)

    # Initialisation 
    particles[0] = proposals[0].sample(size=n_particles)
    for i in range(n_particles):
        log_weights[0, i] = observation_model[0].logpdf(observations[0], [particles[0, i]])\
            + transition_model[0].logpdf([particles[0, i]])\
            - proposals[0].logpdf([particles[0, i]])

    # Loop
    #for t, observation in tqdm(enumerate(observations[1:]), total=t_max-1):  # t=time
    for t in range(t_max-1):
        t_1 = t
        t += 1
        
        w_star = log_weights[t_1].max()
        normalisation_value =  np.log(np.exp(log_weights[t_1] - w_star).sum()) + w_star
        p = np.exp(log_weights[t_1] - normalisation_value)

        new_ancestors_indices = np.searchsorted(p.cumsum(), np.random.rand(n_particles))
        ancestors[t-1] = np.array(list(range(n_particles)))[new_ancestors_indices]
        #ancestors[t_1] = np.random.choice(range(n_particles), size=n_particles, p=probabilities)

        for i in range(n_particles):  # i=particle_id
            # Compute ancestor step
            past = particles[0:t, ancestors[t_1, i]]
            # Generate new particle
            particle = proposals[t].sample(given=past)
            trajectory = np.append(past, particle[np.newaxis,:], axis=0)
            # Compute weight
            log_weight = observation_model[t].logpdf(observations[t], given=trajectory)\
            + transition_model[t].logpdf(particle, given=past)\
            - proposals[t].logpdf(particle, given=past)
            
            # Reset past
            particles[0:(t+1), i] = trajectory
            
            log_weights[t, i] = log_weight

    return particles, np.exp(log_weights), ancestors
