from tqdm import tqdm
from typing import List, Callable
import numpy as np
import scipy
from ipmcmc.distribution import Distribution


# Seqential Monte Carlo for HMM models
def smc(observations: np.ndarray, 
           n_particles: int, 
           transition_model: List[Distribution],
           proposals: List[Distribution],
           observation_model: List[Distribution]
           ):
    # Values to fill
    t_max = len(observations)
    particles = [[None for particle_idx in range(n_particles)] for time in range(t_max)]
    log_weights = np.zeros((t_max, n_particles))
    ancestors = np.zeros((t_max-1, n_particles))

    # Initialisation
    for i in range(n_particles):
        particle = proposals[0].sample()
        particles[0][i] = particle
        log_weights[0, i] = observation_model[0].logpdf(observations[0], given=[particle])\
            + transition_model[0].logpdf(particle, given=list())\
            - proposals[0].logpdf(particle, given=list())

    # Loop
    for t, observation in tqdm(enumerate(observations[1:]), total=t_max-1):  # t=time
        t_1 = t
        t += 1

        previous_log_weights = log_weights[t_1, :]
        w_star = previous_log_weights.max()
        normalisation_value =  np.log(np.exp(previous_log_weights - w_star).sum()) + w_star
        probabilities = np.exp(previous_log_weights - normalisation_value)
        for i in range(n_particles):  # i=particle_id
            # Compute ancestor step
            try:
                ancestor = np.random.choice(range(n_particles), p=probabilities)
            except ValueError as e:
                print(probabilities)
                print(probabilities.sum())
                raise e
            ancestors[t_1][i] = ancestor
            past = [elt[ancestor] for elt in particles[0:t]]
            
            # Generate new particle
            particle = proposals[t].sample(given=past)
            trajectory = past + [particle]
            # Compute weight
            log_weight = observation_model[t].logpdf(observations[t], given=trajectory)\
            + transition_model[t].logpdf(particle, given=past)\
            - proposals[t].logpdf(particle, given=past)
            
            # Reset past
            for old_t, elt in enumerate(trajectory):
                particles[old_t][i] = elt
            log_weights[t][i] = log_weight

    return np.array(particles), log_weights, ancestors
