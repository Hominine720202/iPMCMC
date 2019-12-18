import numpy as np
from filterpy.kalman import KalmanFilter

def rao_blackwellisation(particles, weights, zetas, n_conditional_nodes):
    
    n_steps, n_nodes, t_max, n_particles, state_dim = particles.shape
    w_barre = weights.mean(axis=2)
    rao_black_traj = np.zeros((n_steps, n_conditional_nodes, t_max, state_dim))

    for r in range(n_steps):
        for p in range(n_conditional_nodes):
            for m in range(n_nodes):
                weighted_sum_parts = np.zeros((t_max, state_dim))
                for n in range(n_particles):
                    weighted_sum_parts += w_barre[r, m, n] * particles[r, m, :, n]
                weighted_zetas = zetas[r, p, m] * weighted_sum_parts
                rao_black_traj[r, p] += weighted_zetas
    
    return rao_black_traj


def compute_error(rao_black_traj, ground_truth, mcmc_step=100, state_step=None):
    if state_step is None:
        estimated_mean = np.mean(rao_black_traj[0:mcmc_step], axis=(0, 1))
        return np.mean((estimated_mean - ground_truth)**2)
    else:
        estimated_mean = np.mean(
            rao_black_traj[:, :, 0:state_step], axis=(0, 1))
        return np.mean((estimated_mean - ground_truth[0:state_step])**2)


def compute_ground_truth(observations, initial_mean, initial_var, transition_factor, transition_noise_var, observation_factor, observation_noise_var):
    
    fk = KalmanFilter(dim_x=len(initial_mean), dim_z=observations.shape[1])

    fk.x = initial_mean
    fk.P = initial_var

    fk.F = transition_factor
    fk.Q = transition_noise_var

    fk.H = observation_factor
    fk.R = observation_noise_var

    mu, cov, _, _ = fk.batch_filter(observations)
    means, covs, _, _ = fk.rts_smoother(mu, cov)

    return means, covs


