import numpy as np
from typing import List
from seq_mc import Distribution


def CSMC(obs: np.ndarray,
         N: int,
         conditional_traj: np.ndarray,
         proposals: List[Distribution],
         transitions: List[Distribution],
         obs_models: List[Distribution]
         ):

    T = conditional_traj.shape[0]
    particles = np.zeros((T, N, conditional_traj.shape[1]))
    weights = [[]]
    particles[0] = np.append(proposals[0].sample(
        size=N-1), conditional_traj[0][np.newaxis, :], axis=0)
    for i in range(N):
        weights[0].append(obs_models[0].density(obs[0], particles[np.newaxis, 0, i]) *
                          transitions[0].density(particles[0, i]) / proposals[0].density(particles[0, i]))
    ancestors = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        ancestors[t-1] = np.append(np.random.choice(range(N),
                                                    size=N-1, p=np.array(weights[t-1])/sum(weights[t-1])), N-1)

        new_particles = []
        for i in range(N-1):
            new_particles.append(proposals[t].sample(particles[0:t, ancestors[t-1, i]]))
        new_particles.append(conditional_traj[t])
        particles[t] = np.array(new_particles)
        weights.append([])
        for i in range(N):
            particles[0:t, i] = particles[0:t, ancestors[t-1, i]]
            weights[t].append(obs_models[t].density(obs[t], particles[0:(t+1), i]) * transitions[t].density(
                particles[t, i], particles[0:t, i]) / proposals[t].density(
                    particles[t,i], particles[0:t, i]))

    return particles, np.array(weights)
