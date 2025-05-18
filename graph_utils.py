import rustworkx as rx
from typing import List, Tuple, Sequence # For type hinting

# Note: SparsePauliOp was previously imported but not used in this file.
# If future functions here need it, it should be re-imported.

def create_graph(num_nodes: int, edges_with_weights: List[Tuple[int, int, float]] = None) -> rx.PyGraph:
    """
    Creates a graph object for representing a Max-Cut problem instance.

    The 'check_cycle' parameter, previously available in some versions of
    rustworkx.PyGraph, is not used as it's deprecated or handled internally
    in newer versions.

    Args:
        num_nodes: The total number of nodes (vertices) in the graph.
        edges_with_weights: A list of tuples, where each tuple (u, v, weight)
                            defines an edge between node u and node v with a given weight.
                            If None, a default graph structure (a cycle graph for num_nodes > 1)
                            will be created with unit weights.

    Returns:
        A rustworkx.PyGraph object representing the constructed graph.
    """
    # Initialize an empty undirected graph.
    # Rustworkx handles node indexing (0 to num_nodes-1) automatically when nodes are added.
    graph = rx.PyGraph()

    # Add the specified number of nodes to the graph.
    graph.add_nodes_from(range(num_nodes))

    if edges_with_weights:
        # If a list of edges with weights is provided, add them to the graph.
        # Each item in edge_list_with_weights should be (node_index_1, node_index_2, edge_weight).
        graph.add_edges_from(edges_with_weights)
    else:
        # Default behavior: If no specific edges are provided, create a simple cycle graph.
        # This connects node i to node (i+1), and the last node back to the first node (0).
        # This is useful for quick tests or as a basic graph structure.
        if num_nodes > 1: # A cycle graph requires at least 2 nodes.
            for i in range(num_nodes):
                # Add an edge between node i and node (i+1) modulo num_nodes (for wraparound).
                # Assign a default weight of 1.0 to each edge in the cycle.
                graph.add_edge(i, (i + 1) % num_nodes, 1.0)
    return graph

def build_max_cut_paulis(graph: rx.PyGraph) -> List[Tuple[str, float]]:
    """
    Constructs the Pauli list representation of the Max-Cut cost Hamiltonian
    from a given graph.

    The Max-Cut problem aims to partition the graph's nodes into two sets such that
    the sum of weights of edges connecting nodes in different sets is maximized.
    The cost Hamiltonian for Max-Cut is often formulated as:
    H_cost = sum_{(i,j) in Edges} w_ij * (1 - Z_i Z_j) / 2
    This can be simplified (up to a constant offset and overall scaling, which
    don't affect the optimal solution) to:
    H_cost' = sum_{(i,j) in Edges} w_ij * Z_i Z_j
    This function implements the terms for H_cost'. The Z_i Z_j term contributes
    -1 if nodes i and j are in different partitions and +1 if they are in the same.
    The weights w_ij scale these contributions.

    Args:
        graph: The input graph (rustworkx.PyGraph object), where edges are assumed
               to have weights. Node indices are used for qubit mapping.

    Returns:
        A list of tuples. Each tuple contains:
            - A Pauli string (e.g., "IIZIJZ") representing a term in the Hamiltonian.
              Qubit 0 is the rightmost character.
            - The coefficient (float) for that Pauli string, typically the edge weight.
    """
    pauli_list = [] # Initialize an empty list to store (Pauli string, coefficient) tuples.
    num_graph_nodes = len(graph) # The number of qubits will correspond to the number of nodes in the graph.

    # Iterate over each edge in the graph.
    # graph.weighted_edge_list() provides tuples of (node1_idx, node2_idx, weight).
    for u, v, weight in graph.weighted_edge_list():
        # Create a list of 'I' (identity) operators, one for each qubit.
        paulis_for_this_term = ["I"] * num_graph_nodes

        # For the current edge (u,v), place 'Z' operators on the corresponding qubits.
        paulis_for_this_term[u] = "Z"
        paulis_for_this_term[v] = "Z"

        # Qiskit's convention for Pauli strings is that the rightmost character corresponds to qubit 0.
        # Therefore, reverse the list of characters before joining them into a string.
        pauli_string = "".join(paulis_for_this_term)[::-1]

        # Add the constructed Pauli string and its coefficient (the edge weight) to the list.
        # Ensure the weight is a float.
        pauli_list.append((pauli_string, float(weight)))

    return pauli_list

def evaluate_max_cut_solution(bitstring: Sequence[int], graph: rx.PyGraph) -> float:
    """
    Calculates the value of a cut for a given graph and a specific node partition
    represented by a bitstring.

    The cut value is the sum of weights of all edges that connect nodes assigned
    to different partitions. For a bitstring [b_0, b_1, ..., b_{N-1}], an edge (i,j)
    is in the cut if b_i != b_j.

    Args:
        bitstring: A sequence (list or array) of 0s and 1s. The length of the
                   bitstring must match the number of nodes in the graph.
                   `bitstring[k]` represents the partition of node k.
        graph: The input graph (rustworkx.PyGraph object) for which to evaluate the cut.

    Returns:
        The total weight of the edges crossing the cut, as a float.

    Raises:
        ValueError: If the length of the `bitstring` does not match the number of
                    nodes in the `graph`.
    """
    # Handle the edge case of an empty graph: if there are no nodes and the bitstring is empty,
    # the cut value is 0.
    if graph.num_nodes() == 0 and not bitstring:
        return 0.0

    # Validate that the bitstring length matches the number of nodes in the graph.
    # This is crucial for correct evaluation.
    if len(bitstring) != graph.num_nodes():
        raise ValueError(
            f"Length of bitstring ({len(bitstring)}) must match the number of nodes "
            f"({graph.num_nodes()}) in the graph."
        )

    cut_value = 0.0 # Initialize the sum of weights of edges in the cut.

    # Iterate through all edges in the graph along with their weights.
    for u, v, weight in graph.weighted_edge_list():
        # Check if the two nodes connected by the current edge are in different partitions.
        # bitstring[u] gives the partition of node u, and bitstring[v] for node v.
        if bitstring[u] != bitstring[v]:
            # If they are in different partitions, the edge crosses the cut.
            # Add its weight to the total cut value.
            cut_value += float(weight) # Ensure the weight is treated as a float.

    return cut_value
