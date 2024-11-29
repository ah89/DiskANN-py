import numpy as np
from graph_construction.graph import Graph
from graph_construction.greedy_search import greedy_search
from graph_construction.robust_prune import robust_prune

def vamana(graph, data, max_degree, alpha, metric="euclidean", seed=42):
    """
    Construct a nearest neighbor graph using the Vamana algorithm.

    :param graph: An empty `Graph` object to build the graph.
    :param data: A numpy array of shape (n, d), where n is the number of points and
                 d is the dimensionality of each point.
    :param max_degree: The maximum degree (number of neighbors) for each node.
    :param alpha: Distance threshold multiplier for RobustPrune.
    :param metric: The distance metric to use ("euclidean" or "cosine").
    :param seed: Random seed for reproducibility.
    :return: The constructed `Graph` object.
    """
    np.random.seed(seed)  # Ensure reproducibility

    # Step 1: Initialize the graph with all nodes and random edges
    num_nodes = data.shape[0]
    for node_id in range(num_nodes):
        graph.add_node(node_id, vector=data[node_id])

    # Step 2: Perform the Vamana index construction
    for node_id in range(num_nodes):
        # Step 2.1: Perform greedy search to find candidate neighbors
        # Start from a random node (excluding the current one)
        start_node = np.random.choice([i for i in range(num_nodes) if i != node_id])
        candidate_neighbors = greedy_search(
            graph, query_vector=data[node_id], start_node=start_node, k=max_degree * 2, metric=metric
        )

        # Step 2.2: Prune candidate neighbors using RobustPrune
        candidate_neighbors_vectors = [(neighbor_id, graph.node_vectors[neighbor_id]) for neighbor_id, _ in candidate_neighbors]
        pruned_neighbors = robust_prune(
            candidate_neighbors_vectors,
            node_vector=data[node_id],
            alpha=alpha,
            max_degree=max_degree,
            metric=metric,
        )

        # Step 2.3: Add pruned neighbors as edges in the graph
        for neighbor_id in pruned_neighbors:
            graph.add_edge(node_id, neighbor_id)

    # Step 3: Improve connectivity using backward edges
    for node_id in range(num_nodes):
        for neighbor_id in graph.get_neighbors(node_id):
            graph.add_edge(neighbor_id, node_id)

    return graph


### Test the Vamana algorithm ###
if __name__ == "__main__":
    # Import dependencies
    from graph_construction.graph import Graph

    # Create a small dataset of 2D points
    data = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
    ])

    # Parameters for the Vamana algorithm
    max_degree = 2
    alpha = 1.2
    metric = "euclidean"
    seed = 42

    # Initialize an empty graph
    graph = Graph()

    # Run the Vamana algorithm
    vamana(graph, data, max_degree, alpha, metric, seed)

    # Print the resulting graph
    print("Vamana Graph:")
    graph.print_graph()