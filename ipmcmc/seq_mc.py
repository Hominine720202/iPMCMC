from typing import List
import numpy as np

def seq_mc(observations: list, states: list, n_particles: int, proposals: list, state_to_obs: list, mu):
    raise NotImplementedError('Work in progress')
    particles = np.zeros((n_particles, len(observations)))
    weights = np.zeros((n_particles, len(observations)))
    # Initialisation
    for i in range(n_particles):
        particles[i][0] = proposals[0](states[0])
        weights[i][0] = state_to_obs[0]([particles[i][0]])*mu(particles[i][0])/proposals[0](particles[i][0])

    for idx, observation in enumerate(observations[1:]):
        for i in range(n_particles):
            p = weights[:,idx] / weights[:,idx].sum()
            ancestor = np.random.choice(range(n_particles), p=p)
            past = particles[ancestor][0:idx]
            particle = proposals(past)
            trajectory = np.append(past, particle)
            weight = state_to_obs[idx](trajectory)*mu(particles[i][0])/proposals[0](particles[i][0])
            # TODO: set particles and weights
            particles[i][0:idx+1] = trajectory
            weights[i][idx+1] = weight
    return particles, weights

