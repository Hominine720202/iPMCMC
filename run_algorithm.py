import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
from ipmcmc.generate_data import *
from ipmcmc.linear_gaussian_state_model import *
from ipmcmc.non_linear_gaussian_state_model import *
from ipmcmc.smc import *
from ipmcmc.csmc import *
from ipmcmc.ipmcmc import *
from ipmcmc.estimation import *


if __name__ == "__main__":
    if False:  # Linear case
        # 4.1. Linear Gaussian State Space Model
        np.random.seed(420)
        # Parameters
        r = R.from_rotvec(np.array([7*np.pi/10, 3*np.pi/10, np.pi/20]))
        rotation_matrix = r.as_dcm()
        scaling_matrix = 0.99*np.eye(3)
        beta = np.random.dirichlet(np.ones(20)*0.2, 3).transpose()
        alpha = scaling_matrix@rotation_matrix
        t_max = 50
        mu = np.array([0, 1, 1])
        start_var = 0.1*np.eye(3)
        omega = np.eye(3)
        sigma = 0.1*np.eye(20)

        n_particles = 100

        transition_model = [LinearMu(default_mean=mu, default_cov=start_var)]+[LinearTransition(
            default_mean=np.zeros(3), default_cov=omega, default_alpha=alpha) for t in range(1, t_max)]
        proposals = [LinearMu(default_mean=mu, default_cov=start_var)]+[LinearProposal(
            default_mean=np.zeros(3), default_cov=omega, default_alpha=alpha) for t in range(1, t_max)]
        observation_model = [LinearObservation(default_mean=np.zeros(
            20), default_cov=sigma, default_beta=beta) for t in range(0, t_max)]

        # If we want to change the parameters
        assert np.all(np.linalg.eigvals(start_var) > 0)
        assert np.all(np.linalg.eigvals(omega) > 0)
        assert np.all(np.linalg.eigvals(sigma) > 0)

        states, observations = linear_gaussian_state_space(
            t_max=t_max, mu=mu, start_var=start_var, transition_var=omega, noise_var=sigma,
            transition_coeffs=alpha, observation_coeffs=beta)
        print((states.shape, observations.shape))
    else:  # Non linear case
        # 4.2. Nonlinear State Space Model
        np.random.seed(420)
        t_max = 50
        mu = 0
        start_std = np.sqrt(5)
        omega = np.sqrt(10)
        sigma = np.sqrt(10)

        n_particles = 100

        transition_model = [NonLinearMu(default_mean=mu, default_std=start_std)]+[NonLinearTransition(
            default_mean=0, default_std=omega) for t in range(1, t_max)]
        proposals = [NonLinearMu(default_mean=mu, default_std=start_std)]+[
            NonLinearProposal(default_mean=0, default_std=omega) for t in range(1, t_max)]
        observation_model = [NonLinearObservation(
            default_mean=0, default_std=sigma) for t in range(0, t_max)]

        states, observations = nonlinear_gaussian_state_space(
            t_max=t_max, mu=mu, start_std=start_std, transition_std=omega, noise_std=sigma)

    # %%
    n_nodes = 32
    n_conditional_nodes = 16

    n_steps = 5
    init_conditional_traj = np.zeros((n_conditional_nodes, t_max)+proposals[0].rvs().shape)
    print('init_conditional_traj')
    for i in tqdm(range(n_conditional_nodes)):
        particles, weights, _ = smc(observations, n_particles,
                              transition_model, proposals, observation_model)
        init_conditional_traj[i] = particles[:,np.argmax(weights[-1])]
    print('running ipmcmc')
    particles, conditional_traj, weights, conditional_indices, zetas = ipmcmc(
        n_steps, n_nodes, n_conditional_nodes, observations, n_particles, init_conditional_traj,
        proposals, transition_model, observation_model)

    if False:
        print('Mean estimation')

        true_means, true_covs = compute_ground_truth(observations, mu, start_var, alpha, omega, beta, sigma)

        rao_black_traj = rao_blackwellisation(particles, weights, zetas, n_conditional_nodes)

        errors_function_of_mcmc_step = []
        errors_function_of_state_step = []
        for r in range(1, (n_steps+1)):
            errors_function_of_mcmc_step.append(compute_error(rao_black_traj, true_means, r))

        for t in range(1, (t_max+1)):
            errors_function_of_state_step.append(compute_error(rao_black_traj, true_means, state_step=t))
        

    
