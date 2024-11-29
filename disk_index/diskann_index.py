import os
import numpy as np
from graph_construction.graph import Graph
from graph_construction.vamana import vamana
from graph_construction.greedy_search import greedy_search
from utils.clustering import perform_clustering


class DiskANNIndex:
    """
    DiskANN Index class for managing the construction, saving, loading, and querying of approximate nearest neighbor indices.

    Attributes:
        data (numpy.ndarray): The dataset to be indexed (shape: n x d).
        num_clusters (int): Number of clusters to partition the data into.
        max_degree (int): Maximum degree for each node in the Vamana graph.
        alpha (float): Distance threshold multiplier for RobustPrune.
        metric (str): Distance metric to use ("euclidean" or "cosine").
        overlap (float): Overlap factor for clustering (controls cluster redundancy).
        cluster_graphs (dict): A dictionary containing the graph for each cluster.
        cluster_assignments (dict): A dictionary mapping cluster IDs to point indices.
    """

    def __init__(self, data, num_clusters, max_degree, alpha, overlap=0.0, metric="euclidean", seed=42):
        """
        Initialize the DiskANN Index.

        :param data: A numpy array of shape (n, d), where n is the number of points and d is the dimensionality.
        :param num_clusters: Number of clusters to partition the data into.
        :param max_degree: Maximum degree for each node in the Vamana graph.
        :param alpha: Distance threshold multiplier for RobustPrune.
        :param overlap: Overlap factor for clustering (default: 0.0).
        :param metric: Distance metric to use ("euclidean" or "cosine").
        :param seed: Random seed for reproducibility.
        """
        self.data = data
        self.num_clusters = num_clusters
        self.max_degree = max_degree
        self.alpha = alpha
        self.overlap = overlap
        self.metric = metric
        self.seed = seed

        self.cluster_graphs = {}  # Each cluster's graph
        self.cluster_assignments = {}  # Cluster assignments for each point

    def build_index(self):
        """
        Build the DiskANN index by performing clustering and graph construction.

        :return: None
        """
        print("[INFO] Performing clustering...")
        self.cluster_assignments = perform_clustering(
            self.data, self.num_clusters, overlap=self.overlap, seed=self.seed, metric=self.metric
        )

        print("[INFO] Clustering complete. Constructing graphs for each cluster...")

        for cluster_id, point_indices in self.cluster_assignments.items():
            print(f"[INFO] Constructing graph for cluster {cluster_id} with {len(point_indices)} points...")
            cluster_data = self.data[point_indices]  # Extract data for the current cluster

            # Create an empty graph for the cluster
            cluster_graph = Graph()

            # Build the Vamana graph for the cluster
            vamana(
                graph=cluster_graph,
                data=cluster_data,
                max_degree=self.max_degree,
                alpha=self.alpha,
                metric=self.metric,
                seed=self.seed,
            )

            # Store the graph and cluster data
            self.cluster_graphs[cluster_id] = {"graph": cluster_graph, "data": cluster_data, "indices": point_indices}

        print("[INFO] Graph construction complete.")

    def save_index(self, output_dir):
        """
        Save the index to disk, including cluster assignments and graphs.

        :param output_dir: Directory where the index will be saved.
        :return: None
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save cluster assignments
        cluster_assignments_path = os.path.join(output_dir, "cluster_assignments.npy")
        np.save(cluster_assignments_path, self.cluster_assignments)
        print(f"[INFO] Cluster assignments saved to {cluster_assignments_path}")

        # Save cluster graphs
        for cluster_id, cluster_data in self.cluster_graphs.items():
            graph_path = os.path.join(output_dir, f"cluster_{cluster_id}_graph.npy")
            cluster_data["graph"].save(graph_path)
            print(f"[INFO] Graph for cluster {cluster_id} saved to {graph_path}")

        print(f"[INFO] Index saved successfully to {output_dir}")

    def load_index(self, input_dir):
        """
        Load the index from disk, including cluster assignments and graphs.

        :param input_dir: Directory where the index is stored.
        :return: None
        """
        # Load cluster assignments
        cluster_assignments_path = os.path.join(input_dir, "cluster_assignments.npy")
        self.cluster_assignments = np.load(cluster_assignments_path, allow_pickle=True).item()
        print(f"[INFO] Cluster assignments loaded from {cluster_assignments_path}")

        # Load cluster graphs
        self.cluster_graphs = {}
        for cluster_id, point_indices in self.cluster_assignments.items():
            graph_path = os.path.join(input_dir, f"cluster_{cluster_id}_graph.npy")
            graph = Graph().load(graph_path)

            # Retrieve the cluster data and indices
            cluster_data = self.data[point_indices]
            self.cluster_graphs[cluster_id] = {
                "graph": graph,
                "data": cluster_data,
                "indices": point_indices,  # Add the indices of the points in this cluster
            }
            print(f"[INFO] Graph for cluster {cluster_id} loaded from {graph_path}")

        print("[INFO] Index loaded successfully.")

    def query(self, query_point, top_k=10):
        """
        Perform a nearest neighbor query on the index.

        :param query_point: A single query vector of shape (d,).
        :param top_k: Number of nearest neighbors to return.
        :return: A list of (point_index, distance) tuples representing the top-k nearest neighbors.
        """
        # Identify the closest cluster(s) to the query point
        cluster_distances = []
        for cluster_id, cluster_data in self.cluster_graphs.items():
            # Compute centroid of the cluster
            cluster_centroid = np.mean(cluster_data["data"], axis=0)

            # Compute distance to the cluster centroid
            dist = np.linalg.norm(query_point - cluster_centroid)
            cluster_distances.append((cluster_id, dist))

        # Sort clusters by proximity to the query point
        cluster_distances = sorted(cluster_distances, key=lambda x: x[1])

        # Search the top cluster(s) for the nearest neighbors
        visited_points = set()
        nearest_neighbors = []

        for cluster_id, _ in cluster_distances[:2]:  # Search top 2 clusters
            cluster_graph = self.cluster_graphs[cluster_id]["graph"]
            cluster_data = self.cluster_graphs[cluster_id]["data"]
            cluster_indices = self.cluster_graphs[cluster_id]["indices"]

            # Perform GreedySearch within the cluster graph
            # Start the search at the first node in the cluster
            result = greedy_search(
                graph=cluster_graph,
                query_vector=query_point,
                start_node=0,  # Start at an arbitrary node (e.g., the first node in the cluster)
                k=top_k,
                metric=self.metric,
            )

            # Map results back to the original dataset indices
            for local_index, distance in result:
                global_index = cluster_indices[local_index]
                if global_index not in visited_points:
                    visited_points.add(global_index)
                    nearest_neighbors.append((global_index, distance))

        # Sort and return the top-k neighbors
        nearest_neighbors = sorted(nearest_neighbors, key=lambda x: x[1])[:top_k]
        return nearest_neighbors
    
if __name__ == "__main__":
    # Generate random data for demonstration
    data = np.random.rand(100, 128)  # 100 points in 128-dimensional space
    query_point = np.random.rand(128)

    # Create and build the index
    diskann = DiskANNIndex(data, num_clusters=5, max_degree=10, alpha=1.2, metric="euclidean")
    diskann.build_index()

    # Save the index to disk
    diskann.save_index("./diskann_index")

    # Load the index from disk
    diskann.load_index("./diskann_index")

    # Query the index
    results = diskann.query(query_point, top_k=5)
    print("Query results:", results)