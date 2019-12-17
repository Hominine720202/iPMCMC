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
    log_weights = np.zeros((T, N))
    particles[0] = np.append(proposals[0].sample(
        size=N-1), conditional_traj[0][np.newaxis, :], axis=0)
    for i in range(N):
        weights = obs_models[0].logpdf(obs[0], particles[np.newaxis, 0, i])\
            + transitions[0].logpdf(particles[0, i])\
            - proposals[0].logpdf(particles[0, i])
        log_weights[0] = weights
    ancestors = np.zeros((T, N), dtype=int)
    for t in range(1, T):

        w_star = log_weights[t-1].max()
        normalisation_value = w_star + np.log(np.exp(log_weights[t-1] - w_star).sum())

        p = np.exp(log_weights[t-1] - normalisation_value)

        ancestors[t-1] = np.append(np.random.choice(range(N), size=N-1, p=p))

        for i in range(N):
            if i == N-1:
                particles[t, i] = conditional_traj[t]
            else:
                particles[t, i] = proposals[t].sample(particles[0:t, ancestors[t-1, i]])

            particles[0:t, i] = particles[0:t, ancestors[t-1, i]]

            weights = obs_models[t].logpdf(obs[t], particles[0:(t+1), i])\
                + transitions[t].logpdf(particles[t, i], particles[0:t, i])\
                - proposals[t].logpdf(particles[t, i], particles[0:t, i])

            weights[t, i] = weights

    return particles, weights
