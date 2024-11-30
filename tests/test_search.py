import numpy as np
import pytest
from graph_construction.graph import Graph  # Assuming `Graph` is the custom graph class
from disk_index.beam_search import beam_search

@pytest.fixture
def mock_graph():
    """
    Create a mock graph for testing.
    """
    graph = Graph()
    graph.add_node(0, vector=np.array([0.0, 0.0]))
    graph.add_node(1, vector=np.array([1.0, 1.0]))
    graph.add_node(2, vector=np.array([2.0, 2.0]))
    graph.add_node(3, vector=np.array([4.0, 4.0]))

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    return graph


def test_beam_search(mock_graph):
    """
    Test the beam search algorithm on the mock graph.
    """
    query_vector = np.array([3.5, 3.5])  # Query point close to node 3
    start_node = 1
    k = 2
    beam_size = 3

    # Perform beam search
    results = beam_search(mock_graph, query_vector, start_node, k, beam_size, metric="euclidean")

    # Assert the results
    assert len(results) == k, f"Expected {k} neighbors, got {len(results)}."
    assert results[0][0] == 3, "The closest node should be Node 3."
    assert results[1][0] == 2, "The second closest node should be Node 2."


def test_beam_search_no_neighbors(mock_graph):
    """
    Test beam search when starting at a node with no neighbors.
    """
    # Add an isolated node to the graph
    mock_graph.add_node(4, vector=np.array([10.0, 10.0]))
    mock_graph.add_edge(3, 4)

    query_vector = np.array([11.0, 11.0])  # Query point far from all nodes
    start_node = 3
    k = 2
    beam_size = 3

    # Perform beam search
    results = beam_search(mock_graph, query_vector, start_node, k, beam_size, metric="euclidean")

    # Assert results contain only the isolated node
    assert len(results) == 1, "Only the isolated node should be returned."
    assert results[0][0] == 4, "The isolated node should be Node 5."


def test_beam_search_inner_product(mock_graph):
    """
    Test beam search with inner product metric.
    """
    query_vector = np.array([3.0, 3.0])  # Query point close to Node 2
    start_node = 0
    k = 2
    beam_size = 3

    # Perform beam search using inner product
    results = beam_search(mock_graph, query_vector, start_node, k, beam_size, metric="inner_product")

    # Assert the results
    assert len(results) == k, f"Expected {k} neighbors, got {len(results)}."
    assert results[0][0] == 3, "The closest node (inner product) should be Node 3."
    assert results[1][0] == 2, "The second closest node (inner product) should be Node 2."