import pytest
import numpy as np
from graph_construction.graph import Graph
from graph_construction.vamana import vamana


def test_vamana_graph_construction():
    # Create a small synthetic dataset of 2D points
    data = np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 1.0],  # Node 1
        [2.0, 2.0],  # Node 2
        [3.0, 3.0],  # Node 3
        [4.0, 4.0],  # Node 4
        [5.0, 5.0],  # Node 5
    ], dtype=np.float32)

    # Parameters for the Vamana algorithm
    max_degree = 3  # Maximum number of neighbors per node
    alpha = 1.2  # Distance threshold multiplier for RobustPrune
    metric = "euclidean"
    seed = 42

    # Initialize an empty graph
    graph = Graph()

    # Run the Vamana algorithm
    vamana(graph, data, max_degree=max_degree, alpha=alpha, metric=metric, seed=seed)

    # Assertions
    assert len(graph.adjacency_list) == len(data), "Graph should have the same number of nodes as the dataset."
    for node_id, neighbors in graph.adjacency_list.items():
        assert len(neighbors) <= max_degree, f"Node {node_id} has more than {max_degree} neighbors."

    print("Graph adjacency list:")
    for node_id, neighbors in graph.adjacency_list.items():
        print(f"Node {node_id}: {neighbors}")