import numpy as np
from  csmc import csmc
from seq_mc import seq_mc

def iPMCMC(M, P, R, init_conditional_traj, obs, N):
    c_P = np.random.choice(range(M), P, replace=False)
    conditional_traj = np.zeros((R, P))
    conditional_traj[0] = init_conditional_traj
    for r in range(1, R+1):
        M_no_c_P = [i for i in range(M) if i not in c_P]
        Z = {}
        for m in M_no_c_P:
            _, weights = seq_mc(obs)
            Z[m] = np.prod(np.mean(weights, axis=1))
        conditional_weights = {}
        conditional_particles = {}
        for c in c_P:
            cond_particles, cond_weights = csmc(obs)
            conditional_weights[c] = cond_weights
            conditional_particles[c] = cond_particles
        
        new_c_P = []
        for j in range(P):
            c_P_no_j = [c for c in c_P in c != j]
            zeta = [z if index not in c_P_no_j else 0 for z, index in enumerate(Z)]
            zeta = np.array(zeta)/sum(zeta)
            new_c_P.append(np.random.choice(range(M), p=zeta))

            cond_weight = conditional_weights[c_P[j]]
            b_j = np.random.choice(range(N), p=cond_weight/sum(cond_weight))
            conditional_traj[r, j] = conditional_particles[c_P[j]][-1,b_j]
        c_P = np.array(new_c_P)
