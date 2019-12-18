import numpy as np
from typing import List
from .distribution import Distribution
from .csmc import csmc
from .smc import smc
from tqdm import tqdm

def ipmcmc(n_steps: int,
           n_nodes: int,
           n_conditional_nodes: int,
           observations: np.ndarray,
           n_particles: int,
           init_conditional_traj: np.ndarray,
           proposals: List[Distribution],
           transition_model: List[Distribution],
           observation_model: List[Distribution]
           ):
    
    trajectories_length, state_dim = init_conditional_traj.shape[1:]
    conditional_indices = np.zeros((n_steps+1, n_conditional_nodes), dtype=int)
    conditional_indices[0] = np.random.choice(range(n_nodes), n_conditional_nodes, replace=False)
    conditional_traj = np.zeros((n_steps+1, n_conditional_nodes, trajectories_length, state_dim))
    conditional_traj[0] = init_conditional_traj

    weights = np.zeros((n_steps, n_nodes, trajectories_length, n_particles))
    particles = np.zeros((n_steps, n_nodes, trajectories_length, n_particles, state_dim))
    zetas = np.zeros((n_steps, n_conditional_nodes, n_nodes))

    for r in tqdm(range(1, n_steps+1)):
        c_P = conditional_indices[r-1]
        m_no_cp = [i for i in range(n_nodes) if i not in c_P]
        Z = {}
        
        # Classic Sequential Monte Carlo
        for m in m_no_cp:  # TODO: Multiprocess here
            non_cond_particles, non_cond_weights, _ = smc(observations, n_particles, transition_model, proposals, observation_model)
            weights[r-1, m] = non_cond_weights
            particles[r-1, m] = non_cond_particles
            Z[m] = non_cond_weights.mean(axis=1).prod()

        # Conditional Sequential Monte Carlo
        for i, c in enumerate(c_P):  # TODO: Multiprocess here
            cond_particles, cond_weights, _ = csmc(observations, n_particles, conditional_traj[r-1, i], proposals, transition_model, observation_model)
            weights[r-1, c] = cond_weights
            particles[r-1, c] = cond_particles
            Z[c] = cond_weights.mean(axis=1).prod()
        
        new_c_P = []
        for j in range(n_conditional_nodes):
            c_P_no_j = [c for c in c_P if c != j]
            zeta = [z if index not in c_P_no_j else 0 for index, z in enumerate(Z)]
            zeta = np.array(zeta)/sum(zeta)
            zetas[r-1, j] = zeta
            new_c_P.append(np.random.choice(range(n_nodes), p=zeta))

            cond_weight = weights[r-1, c_P[j]].mean(axis=0)
            b_j = np.random.choice(range(n_particles), p=cond_weight/sum(cond_weight))
            conditional_traj[r, j] = particles[r-1, c_P[j], :, b_j]
        conditional_indices[r] = np.array(new_c_P)

    return particles, conditional_traj, weights, conditional_indices, zetas
