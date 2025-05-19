import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from typing import Sequence, List, Tuple, Any, Callable

# This list will store the objective function values during optimization.
# It's passed as an argument to the cost function to avoid global variables.
objective_func_vals_list_functions_py = []

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
    num_nodes = len(graph.nodes()) # Use graph.num_nodes() or len(graph.nodes())
    for u, v, weight in graph.weighted_edge_list():
        paulis = ["I"] * num_nodes
        # Ensure u and v are valid indices for the paulis list
        if u < num_nodes and v < num_nodes:
            paulis[u] = "Z"
            paulis[v] = "Z"
            # Qiskit Pauli strings are read from right to left (qubit 0 is the rightmost)
            pauli_list.append(("".join(paulis)[::-1], float(weight))) # Ensure weight is float
        else:
            print(f"Warning: Edge ({u}, {v}) contains node index out of bounds for {num_nodes} nodes. Skipping this edge for Pauli construction.")
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
    # Ensure the ansatz has parameters to bind, otherwise pub might be invalid
    # This is more of a sanity check; QAOAAnsatz should have parameters.
    if not ansatz.parameters:
        print("Warning: Ansatz has no parameters to bind.")
        # Depending on EstimatorV2 behavior with parameterless circuits and value lists,
        # this might need specific handling. For now, assume it proceeds or errors if invalid.

    pub = (ansatz, [cost_hamiltonian], [params]) # params should be a list of values

    try:
        result = estimator.run(pubs=[pub]).result()
        cost = result[0].data.evs[0]
    except Exception as e:
        print(f"Error in cost_func_estimator during estimator.run() or result processing: {e}")
        print(f"  Parameters: {params}")
        print(f"  Ansatz num_qubits: {ansatz.num_qubits if hasattr(ansatz, 'num_qubits') else 'N/A'}")
        print(f"  Hamiltonian num_qubits: {cost_hamiltonian.num_qubits if hasattr(cost_hamiltonian, 'num_qubits') else 'N/A'}")
        # Potentially re-raise or return a high cost to penalize
        raise  # Re-raise the exception to make the optimizer aware of the failure

    callback_list.append(cost)
    # The following line was the optional print, now removed:
    # print(f"Parameters: {params}, Cost: {cost}")
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
    bitstring_str = np.binary_repr(integer_representation, width=num_bits)
    return [int(digit) for digit in bitstring_str]

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
    if graph.num_nodes() == 0: # Handle empty graph
        return 0.0
    if len(bitstring) != graph.num_nodes():
        # This can happen if the bitstring comes from a circuit with M qubits
        # but the logical graph has N qubits (N < M).
        # The main script should handle slicing the bitstring to N bits before calling this.
        print(f"Warning in evaluate_max_cut_solution: Bitstring length ({len(bitstring)}) "
              f"does not match graph nodes ({graph.num_nodes()}). Ensure bitstring is for logical qubits.")
        # Adjust bitstring if it's longer, assuming it's for the physical qubits
        # and the logical ones are the first N. This depends on bitstring ordering.
        # For now, we'll raise an error if not matching, expecting main to handle.
        raise ValueError(f"Length of bitstring ({len(bitstring)}) must match the number of nodes ({graph.num_nodes()}) in the graph for evaluation.")


    cut_value = 0.0
    for u, v, weight in graph.weighted_edge_list():
        # Ensure u and v are within the bounds of the bitstring
        if u < len(bitstring) and v < len(bitstring):
            if bitstring[u] != bitstring[v]:
                cut_value += float(weight)
        else:
            print(f"Warning: Edge ({u},{v}) in graph is out of bounds for bitstring of length {len(bitstring)}. Skipping this edge in evaluation.")
    return cut_value
