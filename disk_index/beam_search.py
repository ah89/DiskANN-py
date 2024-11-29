import numpy as np
from heapq import heappush, heappop


def compute_distance(point_a, point_b, metric="euclidean"):
    """
    Compute the distance between two points.

    :param point_a: First point (numpy array).
    :param point_b: Second point (numpy array).
    :param metric: Distance metric, supports "euclidean" or "inner_product".
    :return: Distance as a float.
    """
    if metric == "euclidean":
        return np.linalg.norm(point_a - point_b)
    elif metric == "inner_product":
        return -np.dot(point_a, point_b)  # Negate for max inner product
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def beam_search(graph, query_vector, start_node, k=1, beam_size=5, metric="euclidean"):
    """
    Perform a beam search on the graph to find approximate nearest neighbors.

    :param graph: The Graph object containing adjacency_list and node_vectors.
    :param query_vector: The query vector (numpy array).
    :param start_node: The starting node for the search.
    :param k: Number of nearest neighbors to find.
    :param beam_size: Beam size for the search.
    :param metric: Distance metric, supports "euclidean" or "inner_product".
    :return: A list of tuples (neighbor_id, neighbor_distance) for the top-k neighbors.
    """
    visited = set()  # Track visited nodes to avoid revisiting
    # Initialize beam with the starting node (distance, node_id)
    beam = [(compute_distance(query_vector, graph.node_vectors[start_node], metric), start_node)]
    final_results = []  # Track final results as (node_id, distance)

    while beam:
        next_beam = []
        for _, current_node in beam:
            if current_node in visited:
                continue

            visited.add(current_node)
            neighbors = graph.adjacency_list.get(current_node, [])  # Get neighbors of the current node

            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                # Compute distance from the query to the neighbor
                distance = compute_distance(query_vector, graph.node_vectors[neighbor], metric)
                heappush(next_beam, (distance, neighbor))

            # Keep only the top `beam_size` candidates from this iteration
            next_beam = sorted(next_beam, key=lambda x: x[0])[:beam_size]

        # Update the beam for the next iteration
        beam = next_beam

        # Add nodes from the current beam to the final results
        for distance, neighbor_id in beam:
            final_results.append((neighbor_id, distance))

    # Remove duplicates and return the top-k results
    final_results = list(set(final_results))  # Deduplicate pairs
    return sorted(final_results, key=lambda x: x[1])[:k]  # Sort by distance and return top-k