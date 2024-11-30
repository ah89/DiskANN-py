import pytest
import numpy as np
from disk_index.diskann_index import DiskANNIndex

@pytest.fixture
def sample_data():
    # Create a small dataset of 10 points in 3D space
    return np.random.rand(10, 3).astype(np.float32)

def test_diskann_index_construction(sample_data):
    # Initialize the DiskANN index
    index = DiskANNIndex(
        data=sample_data,
        num_clusters=2,
        max_degree=5,
        alpha=1.2,
        overlap=0.1,
        metric="euclidean",
    )

    # Build the index
    index.build_index()

    # Assertions to ensure the index is constructed correctly
    assert hasattr(index, "cluster_graphs"), "DiskANNIndex should have a cluster_graphs attribute."
    assert len(index.cluster_graphs) == index.num_clusters, (
        "Number of cluster graphs should match the number of clusters."
    )
    assert index.data.shape == sample_data.shape, "The indexed data shape should match the input data shape."

    print("DiskANN index construction test passed.")