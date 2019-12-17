import numpy as np
from typing import List
from  csmc import csmc
from seq_mc import seq_mc, Distribution


def ipmcmc(n_nodes: int,
           n_conditional_nodes: int,
           n_steps: int,
           observations: np.ndarray,
           n_particles: int,
           init_conditional_traj: np.ndarray,
           proposals: List[Distribution],
           transition_model: List[Distribution],
           observation_model: List[Distribution]
           ):
    
    c_P = np.random.choice(range(n_nodes), n_conditional_nodes, replace=False)
    conditional_traj = np.zeros((n_steps, n_conditional_nodes))
    conditional_traj[0] = init_conditional_traj
    for r in range(1, n_steps+1):
        m_no_cp = [i for i in range(n_nodes) if i not in c_P]
        Z = {}
        for m in m_no_cp:
            _, weights = seq_mc(observations, n_particles, transition_model, proposals, observation_model)
            Z[m] = np.prod(np.mean(weights, axis=1))
        conditional_weights = {}
        conditional_particles = {}
        for c in c_P:
            cond_particles, cond_weights = csmc(observations, n_particles, conditional_traj[r-1], proposals, transition_model, observation_model)
            conditional_weights[c] = cond_weights
            conditional_particles[c] = cond_particles
        
        new_c_P = []
        for j in range(n_conditional_nodes):
            c_P_no_j = [c for c in c_P in c != j]
            zeta = [z if index not in c_P_no_j else 0 for z, index in enumerate(Z)]
            zeta = np.array(zeta)/sum(zeta)
            new_c_P.append(np.random.choice(range(n_nodes), p=zeta))

            cond_weight = conditional_weights[c_P[j]]
            b_j = np.random.choice(range(n_particles), p=cond_weight/sum(cond_weight))
            conditional_traj[r, j] = conditional_particles[c_P[j]][-1,b_j]
        c_P = np.array(new_c_P)
