import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from typing import Sequence, List, Tuple, Any, Callable

# This list will store the objective function values during optimization.
# It's passed as an argument to the cost function to avoid global variables.
objective_func_vals_list = []

def build_max_cut_paulis(graph: rx.PyGraph) -> List[Tuple[str, float]]:
    """
    Converts a graph representation of a Max-Cut problem into a list of Pauli strings
    and their coefficients, which form the cost Hamiltonian.

    Args:
        graph: The input graph (rustworkx.PyGraph).

    Returns:
        A list of tuples, where each tuple contains a Pauli string and its coefficient.
    """
    pauli_list = []
    num_nodes = len(graph) # Use graph.num_nodes() or len(graph)
    for u, v, weight in graph.weighted_edge_list():
        paulis = ["I"] * num_nodes
        paulis[u] = "Z"
        paulis[v] = "Z"
        # Qiskit Pauli strings are read from right to left (qubit 0 is the rightmost)
        pauli_list.append(("".join(paulis)[::-1], float(weight))) # Ensure weight is float
    return pauli_list

def cost_func_estimator(
    params: np.ndarray,
    ansatz: Any, # Qiskit circuit (QAOAAnsatz)
    cost_hamiltonian: SparsePauliOp,
    estimator: Estimator,
    callback_list: List[float] # List to store objective values
) -> float:
    """
    Cost function for the QAOA optimization.
    Assigns parameters to the ansatz, runs the circuit using the Estimator,
    and returns the expectation value of the cost Hamiltonian.

    Args:
        params: The parameters (angles) for the QAOA ansatz.
        ansatz: The QAOA ansatz circuit.
        cost_hamiltonian: The cost Hamiltonian (SparsePauliOp).
        estimator: Qiskit Runtime Estimator primitive.
        callback_list: A list to append the current cost value to.

    Returns:
        The estimated cost (expectation value).
    """
    pub = (ansatz, [cost_hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    cost = result[0].data.evs[0]

    callback_list.append(cost)
    print(f"Parameters: {params}, Cost: {cost}") # Optional: for logging progress
    return cost

def to_bitstring(integer_representation: int, num_bits: int) -> List[int]:
    """
    Converts an integer to its binary representation as a list of bits.

    Args:
        integer_representation: The integer to convert.
        num_bits: The desired number of bits in the output bitstring.

    Returns:
        A list of integers (0 or 1) representing the bitstring.
    """
    # np.binary_repr returns a string, e.g., '101'
    # We need to convert it to a list of ints [1, 0, 1]
    # The string is typically ordered from most significant to least significant.
    # Depending on convention, you might need to reverse it.
    # Qiskit's convention is usually q_n-1 ... q_1 q_0 (rightmost is qubit 0).
    # If the integer comes from measurement outcomes, it usually matches this.
    bitstring_str = np.binary_repr(integer_representation, width=num_bits)
    return [int(digit) for digit in bitstring_str] # MSB first by default from binary_repr

def evaluate_max_cut_solution(bitstring: Sequence[int], graph: rx.PyGraph) -> float:
    """
    Evaluates the quality of a Max-Cut solution (bitstring) for a given graph.
    The cut value is the sum of weights of edges connecting nodes in different partitions.

    Args:
        bitstring: A list or array of 0s and 1s, where each element corresponds
                   to a node's partition.
        graph: The input graph (rustworkx.PyGraph).

    Returns:
        The value of the cut defined by the bitstring.
    """
    if len(bitstring) != graph.num_nodes():
        raise ValueError("Length of bitstring must match the number of nodes in the graph.")

    cut_value = 0.0
    for u, v, weight in graph.weighted_edge_list():
        # If nodes u and v are in different partitions, add the edge weight to the cut
        if bitstring[u] != bitstring[v]:
            cut_value += float(weight) # Ensure weight is float
    return cut_value
