import networkx as nx
import numpy as np
from typing import List


def sample_nodes(G, num_nodes, rng: np.random.RandomState):
    nodes = list(G.nodes)
    return rng.choice(nodes, num_nodes, replace=False)


def sample_neighbors(
    G: nx.Graph, num_nodes: int, rng: np.random.RandomState
) -> np.ndarray | List[int]:
    nodes = list(G.nodes)
    rand_nodes = rng.choice(nodes, num_nodes, replace=False)
    return [rng.choice(list(G.neighbors(node))) for node in rand_nodes]


def remove_nodes_from_graph(G: nx.Graph, nodes):
    G.remove_nodes_from(nodes)
    return G
