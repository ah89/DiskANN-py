import numpy as np
from sklearn.cluster import KMeans


class ProductQuantizer:
    """
    Product Quantizer for compressing high-dimensional data into a compact representation.

    Attributes:
        num_subvectors (int): Number of sub-vectors to split the data into.
        num_clusters (int): Number of clusters (codebook size) for each sub-vector.
        subvector_dim (int): Dimensionality of each sub-vector.
        codebooks (list): List of trained KMeans models (one for each sub-vector).
    """

    def __init__(self, num_subvectors, num_clusters):
        """
        Initialize the Product Quantizer.

        :param num_subvectors: Number of sub-vectors to split the data into.
        :param num_clusters: Number of clusters (codebook size) for each sub-vector.
        """
        self.num_subvectors = num_subvectors
        self.num_clusters = num_clusters
        self.subvector_dim = None  # Computed during training
        self.codebooks = []  # Stores k-means models for each sub-vector

    def fit(self, data):
        """
        Train the Product Quantizer on the given data.

        :param data: A numpy array of shape (n, d), where n is the number of points and d is the dimensionality.
        :return: None
        """
        n, d = data.shape
        self.subvector_dim = d // self.num_subvectors

        if d % self.num_subvectors != 0:
            raise ValueError(
                f"Dimensionality {d} cannot be evenly divided into {self.num_subvectors} sub-vectors."
            )

        self.codebooks = []
        for i in range(self.num_subvectors):
            start = i * self.subvector_dim
            end = (i + 1) * self.subvector_dim
            subvector_data = data[:, start:end]

            # Train k-means for this sub-vector
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            kmeans.fit(subvector_data)
            self.codebooks.append(kmeans)

        print(f"[INFO] Product Quantizer trained with {self.num_subvectors} sub-vectors and {self.num_clusters} clusters per sub-vector.")

    def encode(self, data):
        """
        Encode the data into PQ compressed representation.

        :param data: A numpy array of shape (n, d), where n is the number of points and d is the dimensionality.
        :return: A numpy array of shape (n, num_subvectors), where each value is the cluster index.
        """
        if not self.codebooks:
            raise ValueError("Product Quantizer has not been trained. Call fit() before encode().")

        n, d = data.shape
        if d != self.subvector_dim * self.num_subvectors:
            raise ValueError(f"Data dimensionality {d} does not match the trained sub-vector dimensionality.")

        compressed_data = np.zeros((n, self.num_subvectors), dtype=np.int32)

        for i in range(self.num_subvectors):
            start = i * self.subvector_dim
            end = (i + 1) * self.subvector_dim
            subvector_data = data[:, start:end]

            # Assign each sub-vector to the nearest cluster
            cluster_indices = self.codebooks[i].predict(subvector_data)
            compressed_data[:, i] = cluster_indices

        return compressed_data

    def decode(self, codes):
        """
        Decode the PQ compressed representation back to approximate original data.

        :param codes: A numpy array of shape (n, num_subvectors), where each value is the cluster index.
        :return: A numpy array of shape (n, d), where d = subvector_dim * num_subvectors.
        """
        if not self.codebooks:
            raise ValueError("Product Quantizer has not been trained. Call fit() before decode().")

        n, num_subvectors = codes.shape
        if num_subvectors != self.num_subvectors:
            raise ValueError(f"Number of sub-vectors {num_subvectors} does not match the trained configuration.")

        reconstructed_data = np.zeros((n, self.subvector_dim * self.num_subvectors))

        for i in range(self.num_subvectors):
            start = i * self.subvector_dim
            end = (i + 1) * self.subvector_dim
            cluster_centers = self.codebooks[i].cluster_centers_

            # Map the cluster indices back to the cluster centers
            reconstructed_data[:, start:end] = cluster_centers[codes[:, i]]

        return reconstructed_data

if __name__ == "__main__":
    # Generate random data
    data = np.random.rand(1000, 64)  # 1000 points in 64-dimensional space

    # Initialize the Product Quantizer
    pq = ProductQuantizer(num_subvectors=4, num_clusters=256)

    # Train the quantizer
    pq.fit(data)

    # Encode the data
    compressed_data = pq.encode(data)
    print("Compressed data shape:", compressed_data.shape)

    # Decode the compressed data
    reconstructed_data = pq.decode(compressed_data)
    print("Reconstructed data shape:", reconstructed_data.shape)

    # Compare original and reconstructed data
    print("Original data sample:", data[0])
    print("Reconstructed data sample:", reconstructed_data[0])