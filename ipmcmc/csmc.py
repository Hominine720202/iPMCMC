import numpy as np


def CSMC(obs, N, conditional_traj, proposals_samplers, proposals_dens, transition_dens, obs_dens, init_dens):
    particles = np.array([[]])
    weights = [[]]
    particles[0] = np.append(proposals_samplers[0](
        N-1), conditional_traj[0])
    for i in range(N):
        weights[0].append(obs_dens[0](obs[0], particles[0, i]) *
                          init_dens(particles[0, i]) / proposals_dens[0](particles[0, i]))
    T = len(obs)
    ancestors = np.array([[]])
    for t in range(1, T):
        ancestors[t-1] = np.append(np.random.choice(range(1, N+1),
                                                    size=N-1, p=weights[t-1]), N)

        new_particles = []
        for i in range(N-1):
            new_particles.append(proposals_samplers[0](
                1, particles[0:t-1, ancestors[t-1, i]]))
        new_particles.append(conditional_traj[t])
        particles[t] = np.array(new_particles)
        weights.append([])
        for i in range(N):
            particles[0:t, i] = particles[0:t, ancestors[t-1, i]]
            weights[t].append(obs_dens[t](obs[t], particles[0:t+1, i]) * transition_dens[t-1](
                particles[0:t, ancestors[t-1, i]]) / proposals_dens[t](particles[0:t, ancestors[t-1, i]]))
