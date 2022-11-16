import networkx as nx
import numpy as np
from tqdm import tqdm


def one_step_simulation(graph, beta, delta, state, rg):
    """
    Simulate one step of the SIS model.
    """
    new_state = state.copy()
    for i in range(graph.number_of_nodes()):
        if state[i] == 1:
            j = list(graph.neighbors(i))
            change = (rg.rand(len(j)) < beta).astype(int)
            new_state[j] = state[j] | change
            new_state[i] = 0 if rg.rand() < delta else 1
    return new_state


def simulate_SIS(graph, beta=0.2, delta=0.2, infected=[0], max_time=500, seed=10):
    """
    Simulate the SIS model with the given parameters.
    """
    N = max(graph.nodes()) + 1
    rg = np.random.RandomState(seed)
    # Initialize the state vector
    state = np.zeros(N, dtype=int)
    for i in infected:
        state[i] = 1

    # simulate max_time steps
    states = [state]
    for i in tqdm(range(max_time)):
        state = one_step_simulation(graph, beta, delta, state, rg)
        states.append(state)
    return states


def read_edge_list(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    graph = nx.Graph()
    for line in lines:
        u, v = line.split()
        graph.add_edge(int(u), int(v))
    return graph
