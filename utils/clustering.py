import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


def perform_clustering(data, num_clusters, overlap=0.0, seed=42, metric="euclidean"):
    """
    Partition the dataset into clusters using k-means clustering.

    :param data: A numpy array of shape (n, d), where n is the number of points
                 and d is the dimensionality of each point.
    :param num_clusters: The number of clusters to create.
    :param overlap: Overlap factor (a value between 0 and 1). Determines how many
                    points are assigned to multiple clusters. Default is 0.0 (no overlap).
    :param seed: Random seed for reproducibility.
    :param metric: Distance metric to use ("euclidean" or "cosine").
    :return: A dictionary where keys are cluster IDs and values are lists of point indices.
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

    # Step 1: Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    kmeans.fit(data)

    # Step 2: Assign points to clusters
    cluster_assignments = defaultdict(list)
    for point_idx, point in enumerate(data):
        # Compute distances to all cluster centroids
        distances = [compute_distance(point, centroid, metric) for centroid in kmeans.cluster_centers_]
        # Sort by distance (ascending) and assign to the closest cluster(s)
        sorted_clusters = np.argsort(distances)

        # Assign point to the closest cluster
        cluster_assignments[sorted_clusters[0]].append(point_idx)

        # If overlap > 0, assign point to additional clusters
        if overlap > 0:
            num_extra_clusters = int(overlap * num_clusters)
            for extra_cluster in sorted_clusters[1:1 + num_extra_clusters]:
                cluster_assignments[extra_cluster].append(point_idx)

    # Convert cluster_assignments to a standard dictionary (not defaultdict)
    return {cluster_id: list(set(indices)) for cluster_id, indices in cluster_assignments.items()}


### Test the Clustering Algorithm ###
if __name__ == "__main__":
    # Create a small synthetic dataset
    data = np.array([
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [8.0, 8.0],
        [9.0, 8.0],
        [8.0, 9.0],
    ])

    # Parameters for clustering
    num_clusters = 2
    overlap = 0.1  # Allow a small overlap
    seed = 42
    metric = "euclidean"

    # Perform clustering
    clusters = perform_clustering(data, num_clusters, overlap, seed, metric)

    # Print the resulting clusters
    print("Cluster assignments:")
    for cluster_id, indices in clusters.items():
        print(f"Cluster {cluster_id}: Points {indices}")

    # Show the cluster centroids
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    kmeans.fit(data)
    print("\nCluster centroids:")
    print(kmeans.cluster_centers_)