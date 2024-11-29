import numpy as np
from graph_construction.graph import Graph


def greedy_search(graph, query_vector, start_node, k, metric="euclidean"):
    """
    Perform GreedySearch on the graph to find the k-nearest neighbors to the query vector.

    :param graph: A `Graph` object (from graph_construction.graph).
    :param query_vector: The query vector as a numpy array.
    :param start_node: The starting node ID for the search.
    :param k: The number of nearest neighbors to find.
    :param metric: The distance metric to use ("euclidean" or "cosine").
    :return: A list of (neighbor_node_id, distance) tuples sorted by distance.
    """

    def compute_distance(vec1, vec2, metric):
        """Helper function to compute the distance between two vectors."""
        if metric == "euclidean":
            return np.linalg.norm(vec1 - vec2)
        elif metric == "cosine":
            # Cosine similarity distance = 1 - cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return 1 - similarity
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    # Priority queue to track visited nodes and their distances
    visited_nodes = set()
    current_best = [(start_node, compute_distance(query_vector, graph.node_vectors[start_node], metric))]

    # Greedy traversal
    while True:
        # Sort current candidates by distance and pick the closest one
        current_best.sort(key=lambda x: x[1])
        closest_node, closest_distance = current_best[0]

        # If the closest node has already been visited, stop the search
        if closest_node in visited_nodes:
            break

        # Mark the node as visited
        visited_nodes.add(closest_node)

        # Explore neighbors of the current node
        for neighbor in graph.get_neighbors(closest_node):
            if neighbor not in visited_nodes:
                neighbor_distance = compute_distance(query_vector, graph.node_vectors[neighbor], metric)
                current_best.append((neighbor, neighbor_distance))

        # Stop if we've visited enough nodes to find k neighbors
        if len(visited_nodes) >= k:
            break

    # Return the k-nearest neighbors sorted by distance
    return sorted(current_best, key=lambda x: x[1])[:k]


### Test the GreedySearch algorithm ###
if __name__ == "__main__":
    # Import necessary dependencies
    from graph_construction.graph import Graph

    # Create a graph
    graph = Graph()

    # Add nodes with vectors
    graph.add_node(1, vector=np.array([1.0, 2.0]))
    graph.add_node(2, vector=np.array([3.0, 4.0]))
    graph.add_node(3, vector=np.array([5.0, 6.0]))
    graph.add_node(4, vector=np.array([7.0, 8.0]))

    # Add edges
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    # Define a query vector
    query_vector = np.array([4.0, 5.0])

    # Perform GreedySearch
    k = 2
    start_node = 1
    results = greedy_search(graph, query_vector, start_node, k, metric="euclidean")

    # Print results
    print(f"The {k}-nearest neighbors to the query vector {query_vector} starting at node {start_node} are:")
    for neighbor_id, distance in results:
        print(f"Node {neighbor_id} with distance {distance:.4f}")