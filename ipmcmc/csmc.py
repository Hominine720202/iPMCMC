import numpy as np
from typing import List
from distribution import Distribution


def csmc(observations: np.ndarray,
         n_particles: int,
         conditional_traj: np.ndarray,
         proposals: List[Distribution],
         transition_model: List[Distribution],
         observation_model: List[Distribution]
         ):

    T = observations.shape[0]
    particles = np.zeros((T, n_particles, conditional_traj.shape[1]))
    log_weights = np.zeros((T, n_particles))
    particles[0] = np.append(proposals[0].sample(
        size=n_particles-1), conditional_traj[0][np.newaxis, :], axis=0)
    for i in range(n_particles):
        weights = observation_model[0].logpdf(observations[0], particles[np.newaxis, 0, i])\
            + transition_model[0].logpdf(particles[0, i])\
            - proposals[0].logpdf(particles[0, i])
        log_weights[0] = weights
    ancestors = np.zeros((T-1, n_particles), dtype=int)
    for t in range(1, T):

        w_star = log_weights[t-1].max()
        normalisation_value = w_star + np.log(np.exp(log_weights[t-1] - w_star).sum())

        p = np.exp(log_weights[t-1] - normalisation_value)

        new_ancestors_indices = np.searchsorted(p.cumsum(), np.random.rand(n_particles-1))
        ancestors[t-1] = np.append(np.array(list(range(n_particles)))[new_ancestors_indices], n_particles-1)
        #ancestors[t-1] = np.append(np.random.choice(range(n_particles), size=n_particles-1, p=p), n_particles-1)
        
        for i in range(n_particles):
            if i == n_particles-1:
                particles[t, i] = conditional_traj[t]
            else:
                particles[t, i] = proposals[t].sample(particles[0:t, ancestors[t-1, i]])

            particles[0:t, i] = particles[0:t, ancestors[t-1, i]]

            weights = observation_model[t].logpdf(observations[t], particles[0:(t+1), i])\
                + transition_model[t].logpdf(particles[t, i], particles[0:t, i])\
                - proposals[t].logpdf(particles[t, i], particles[0:t, i])
            
            log_weights[t, i] = weights

    return particles, np.exp(log_weights), ancestors
