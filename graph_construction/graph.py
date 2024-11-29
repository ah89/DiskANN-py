import numpy as np
import pickle


class Graph:
    """
    A directed graph structure to store and manage nearest neighbor connections.
    Each node corresponds to a point in the dataset, and edges represent directed
    connections between nodes.
    """

    def __init__(self):
        # Adjacency list: maps node IDs to their neighbors
        self.adjacency_list = {}
        # Stores the vectors associated with each node (optional, for testing or visualization)
        self.node_vectors = {}

    ### Core Graph Operations ###

    def add_node(self, node_id, vector=None):
        """
        Add a node to the graph.
        :param node_id: Unique identifier for the node.
        :param vector: Optional, the vector associated with the node.
        """
        if node_id not in self.adjacency_list:
            self.adjacency_list[node_id] = set()  # Initialize with an empty set of neighbors
        if vector is not None:
            self.node_vectors[node_id] = vector

    def add_edge(self, from_node, to_node):
        """
        Add a directed edge from one node to another.
        :param from_node: Source node ID.
        :param to_node: Target node ID.
        """
        if from_node not in self.adjacency_list:
            raise ValueError(f"Node {from_node} does not exist in the graph.")
        if to_node not in self.adjacency_list:
            raise ValueError(f"Node {to_node} does not exist in the graph.")
        self.adjacency_list[from_node].add(to_node)

    def remove_edge(self, from_node, to_node):
        """
        Remove a directed edge from one node to another.
        :param from_node: Source node ID.
        :param to_node: Target node ID.
        """
        if from_node in self.adjacency_list and to_node in self.adjacency_list[from_node]:
            self.adjacency_list[from_node].remove(to_node)

    def get_neighbors(self, node_id):
        """
        Get the neighbors (outgoing edges) of a given node.
        :param node_id: Node ID.
        :return: A set of neighboring node IDs.
        """
        if node_id not in self.adjacency_list:
            raise ValueError(f"Node {node_id} does not exist in the graph.")
        return self.adjacency_list[node_id]

    ### Utility Functions ###

    def get_all_nodes(self):
        """
        Get the list of all node IDs in the graph.
        :return: A list of all node IDs.
        """
        return list(self.adjacency_list.keys())

    def degree(self, node_id):
        """
        Get the out-degree of a given node.
        :param node_id: Node ID.
        :return: The number of outgoing edges from the node.
        """
        if node_id not in self.adjacency_list:
            raise ValueError(f"Node {node_id} does not exist in the graph.")
        return len(self.adjacency_list[node_id])

    ### Graph Serialization and Deserialization ###

    def save(self, filepath):
        """
        Save the graph to disk using pickle.
        :param filepath: Path to the file where the graph will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Load a graph from a pickle file.
        :param filepath: Path to the file from which the graph will be loaded.
        :return: A Graph object.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    ### Visualization (Optional for Debugging) ###

    def print_graph(self):
        """
        Print the adjacency list representation of the graph for debugging.
        """
        for node, neighbors in self.adjacency_list.items():
            print(f"Node {node}: {sorted(neighbors)}")

if __name__ == "__main__":
    graph = Graph()

    # Add nodes
    graph.add_node(1, vector=np.array([1.0, 2.0]))
    graph.add_node(2, vector=np.array([3.0, 4.0]))
    graph.add_node(3, vector=np.array([5.0, 6.0]))

    # Add edges
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(1, 3)

    # Get neighbors
    print("Neighbors of node 1:", graph.get_neighbors(1))

    # Print the graph
    graph.print_graph()

    # Save the graph to disk
    graph.save("graph.pkl")

    # Load the graph back
    loaded_graph = Graph.load("graph.pkl")
    loaded_graph.print_graph()