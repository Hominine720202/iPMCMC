import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_ancestors(ancestors: np.ndarray, img: str):
    raise DeprecationWarning("TODO: Refactor according to the new shape (time, n_particles)")
    # TODO: Refactor according to the new shape (time, n_particles)
    G = nx.Graph()

    for p_i in range(ancestors.shape[0]):
        G.add_node("p={};t={}".format(p_i, 0))
    
    for t_i in range(ancestors.shape[1]):
        for p_i, a_t_i in enumerate(ancestors[:,t_i]):
            G.add_node("p={};t={}".format(int(p_i), int(t_i+1)))
            G.add_edge("p={};t={}".format(int(a_t_i), int(t_i)) ,"p={};t={}".format(int(p_i), int(t_i+1)))
    
    coeff = 10000
    fixed_positions = {node: (int(node.split(';')[0][2:])*coeff,int(node.split(';')[1][2:])*coeff)
        for node in G.nodes
    }

    plt.figure(figsize=(120,120))
    pos = nx.spring_layout(G,pos=fixed_positions, fixed=G.nodes)
    nx.draw_networkx(G,pos)
    plt.savefig(img)
    plt.close()
   
    return G