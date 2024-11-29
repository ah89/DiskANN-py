import numpy as np


def robust_prune(candidate_neighbors, node_vector, alpha, max_degree, metric="euclidean"):
    """
    Perform RobustPrune to select a sparse subset of neighbors for a node.

    :param candidate_neighbors: A list of tuples [(neighbor_id, vector), ...].
                                Each tuple contains the ID and vector of a neighbor.
    :param node_vector: The vector of the node for which neighbors are being pruned.
    :param alpha: Distance threshold multiplier (controls pruning aggressiveness).
    :param max_degree: The maximum allowed degree (number of neighbors) for the node.
    :param metric: The distance metric to use ("euclidean" or "cosine").
    :return: A list of selected neighbor IDs after pruning.
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

    # Sort candidate neighbors by distance to the node
    candidate_neighbors = sorted(
        candidate_neighbors,
        key=lambda x: compute_distance(node_vector, x[1], metric)
    )

    selected_neighbors = []  # List of selected neighbors after pruning

    for neighbor_id, neighbor_vector in candidate_neighbors:
        # Check if adding this neighbor violates the distance threshold with existing neighbors
        is_valid = True
        for selected_id, selected_vector in selected_neighbors:
            dist_to_selected = compute_distance(neighbor_vector, selected_vector, metric)
            dist_to_node = compute_distance(node_vector, selected_vector, metric)
            if dist_to_selected <= alpha * dist_to_node:
                is_valid = False
                break

        if is_valid:
            selected_neighbors.append((neighbor_id, neighbor_vector))

        # Stop if the maximum degree is reached
        if len(selected_neighbors) >= max_degree:
            break

    # Return only the IDs of the selected neighbors
    return [neighbor_id for neighbor_id, _ in selected_neighbors]


### Test the RobustPrune algorithm ###
if __name__ == "__main__":
    # Define some sample nodes with 2D vectors
    node_vector = np.array([1.0, 1.0])
    candidate_neighbors = [
        (1, np.array([2.0, 2.0])),
        (2, np.array([3.0, 3.0])),
        (3, np.array([4.0, 4.0])),
        (4, np.array([5.0, 5.0])),
    ]

    # Parameters for RobustPrune
    alpha = 1.2
    max_degree = 2
    metric = "euclidean"

    # Run RobustPrune
    selected_neighbors = robust_prune(candidate_neighbors, node_vector, alpha, max_degree, metric)

    # Print the results
    print(f"Selected neighbor IDs after pruning: {selected_neighbors}")