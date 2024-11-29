import os
import numpy as np
from graph_construction.graph import Graph
from graph_construction.vamana import vamana
from graph_construction.greedy_search import greedy_search
from utils.clustering import perform_clustering
from disk_index.beam_search import beam_search
from utils.caching import RedisCache


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

    def __init__(self, data, num_clusters, max_degree, alpha, overlap=0.0, metric="euclidean", seed=42, use_cache=False):
        """
        Initialize the DiskANN Index.

        :param data: A numpy array of shape (n, d), where n is the number of points and d is the dimensionality.
        :param num_clusters: Number of clusters to partition the data into.
        :param max_degree: Maximum degree for each node in the Vamana graph.
        :param alpha: Distance threshold multiplier for RobustPrune.
        :param overlap: Overlap factor for clustering (default: 0.0).
        :param metric: Distance metric to use ("euclidean" or "cosine").
        :param seed: Random seed for reproducibility.
        :param use_cache: Boolean flag to enable/disable caching (default: False).
        """
        self.data = data
        self.num_clusters = num_clusters
        self.max_degree = max_degree
        self.alpha = alpha
        self.overlap = overlap
        self.metric = metric
        self.seed = seed
        self.use_cache = use_cache  # Switch for enabling/disabling caching

        self.cluster_graphs = {}  # Each cluster's graph
        self.cluster_assignments = {}  # Cluster assignments for each point

        # Initialize Redis cache if caching is enabled
        if self.use_cache:
            self.cache = RedisCache(host="localhost", port=6379, db=0, ttl=3600)  # 1-hour TTL
        else:
            self.cache = None

    def build_index(self):
        """
        Build the DiskANN index by performing clustering and graph construction.

        :return: None
        """
        print("[INFO] Performing clustering...")

        # Check if cluster assignments are cached (only if caching is enabled)
        if self.use_cache:
            cached_assignments = self.cache.get("cluster_assignments")
            if cached_assignments is not None:
                print("[INFO] Loaded cluster assignments from cache.")
                self.cluster_assignments = cached_assignments
                return

        # Perform clustering and optionally cache the result
        self.cluster_assignments = perform_clustering(
            self.data, self.num_clusters, overlap=self.overlap, seed=self.seed, metric=self.metric
        )

        if self.use_cache:
            self.cache.set("cluster_assignments", self.cluster_assignments)

        print("[INFO] Clustering complete. Constructing graphs for each cluster...")

        for cluster_id, point_indices in self.cluster_assignments.items():
            print(f"[INFO] Constructing graph for cluster {cluster_id} with {len(point_indices)} points...")
            cluster_data = self.data[point_indices]  # Extract data for the current cluster

            # Check if the graph for this cluster is cached (only if caching is enabled)
            if self.use_cache:
                cached_graph = self.cache.get(f"cluster_graph:{cluster_id}")
                if cached_graph is not None:
                    print(f"[INFO] Loaded graph for cluster {cluster_id} from cache.")
                    self.cluster_graphs[cluster_id] = cached_graph
                    continue

            # Create a new graph
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

            # Cache the graph (only if caching is enabled)
            if self.use_cache:
                self.cache.set(f"cluster_graph:{cluster_id}", self.cluster_graphs[cluster_id])

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

    def query(self, query_point, top_k=10, method="beam", beam_size=5):
        """
        Query the index to find the top-k nearest neighbors of the given query point.

        :param query_point: A 1D numpy array representing the query vector.
        :param top_k: Number of nearest neighbors to retrieve.
        :param method: Search method to use ("beam" or "greedy").
        :param beam_size: Beam size for the beam search (ignored for greedy search).
        :return: A list of tuples (global_index, distance).
        """
        # Generate a unique key for the query
        query_key = f"query:{hash(query_point.tobytes())}:{top_k}:{method}:{beam_size}"

        # Check if the query result is cached (only if caching is enabled)
        if self.use_cache:
            cached_result = self.cache.get(query_key)
            if cached_result is not None:
                print("[INFO] Returning cached query result.")
                return cached_result

        print("[INFO] Querying index...")

        # Compute distances between the query point and cluster centroids
        cluster_distances = []
        for cluster_id, cluster_data in self.cluster_graphs.items():
            # Generate a unique cache key for the query-cluster pair
            cache_key = f"distance:{hash(query_point.tobytes())}:{cluster_id}"

            # Check if the distance is cached (only if caching is enabled)
            if self.use_cache:
                cached_distance = self.cache.get(cache_key)
                if cached_distance is not None:
                    dist = cached_distance
                else:
                    # Compute the distance if not cached
                    cluster_centroid = np.mean(cluster_data["data"], axis=0)
                    dist = np.linalg.norm(query_point - cluster_centroid)

                    # Cache the computed distance (only if caching is enabled)
                    self.cache.set(cache_key, dist)
            else:
                # Compute the distance without caching
                cluster_centroid = np.mean(cluster_data["data"], axis=0)
                dist = np.linalg.norm(query_point - cluster_centroid)

            # Append the result
            cluster_distances.append((cluster_id, dist))

        # Sort clusters by distance
        cluster_distances = sorted(cluster_distances, key=lambda x: x[1])

        visited_points = set()
        nearest_neighbors = []

        # Process the top 2 closest clusters (or more if needed)
        for cluster_id, _ in cluster_distances[:2]:
            cluster_graph = self.cluster_graphs[cluster_id]["graph"]
            cluster_indices = self.cluster_graphs[cluster_id]["indices"]

            # Choose the search method
            if method == "beam":
                result = beam_search(
                    graph=cluster_graph,
                    query_vector=query_point,
                    start_node=0,
                    k=top_k,
                    beam_size=beam_size,
                    metric=self.metric,
                )
            elif method == "greedy":
                result = greedy_search(
                    graph=cluster_graph,
                    query_vector=query_point,
                    start_node=0,
                    k=top_k,
                    metric=self.metric,
                )
            else:
                raise ValueError(f"Unsupported query method: {method}")

            # Map local indices to global indices
            for local_index, distance in result:
                global_index = cluster_indices[local_index]
                if global_index not in visited_points:
                    visited_points.add(global_index)
                    nearest_neighbors.append((global_index, distance))

        # Sort and return the top-k global neighbors
        nearest_neighbors = sorted(nearest_neighbors, key=lambda x: x[1])[:top_k]

        # Cache the query result (only if caching is enabled)
        if self.use_cache:
            self.cache.set(query_key, nearest_neighbors)

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