import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit import QuantumCircuit, ParameterExpression # For type hinting and potential debug checks
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit.compiler import transpile as qiskit_transpile # For converting to basis gates
from qiskit.transpiler import Layout # For creating initial_layout during transpilation
from scipy.optimize import minimize # Classical optimizer
from typing import List, Tuple, Any, Callable # Standard type hints

# Import utility from graph_utils to build the Hamiltonian components
from graph_utils import build_max_cut_paulis

# Global list to store objective function values during the optimization process.
# This list is populated by the `cost_func_estimator` callback.
# While convenient, for more complex scenarios, consider passing state via objects or closures.
objective_func_vals_list: List[float] = []

def setup_qaoa_ansatz_and_hamiltonian(graph: Any, num_layers: int) -> Tuple[SparsePauliOp, QuantumCircuit | None]:
    """
    Sets up the Max-Cut problem by building the cost Hamiltonian from the graph
    and constructing the parameterized QAOA ansatz circuit. The ansatz is then
    decomposed into a standard QuantumCircuit composed of basic gates.

    Args:
        graph: The problem graph, expected to be a `rustworkx.PyGraph` instance.
               Node indices are mapped to qubit indices.
        num_layers: The number of layers (p) for the QAOA ansatz. Each layer
                    consists of applying the cost Hamiltonian and then the mixer Hamiltonian.

    Returns:
        A tuple: (cost_hamiltonian, decomposed_qaoa_circuit).
        - `cost_hamiltonian`: A `SparsePauliOp` representing the Max-Cut cost function.
        - `decomposed_qaoa_circuit`: The QAOA ansatz as a `QuantumCircuit`, decomposed
                                     into elementary gates. Returns `None` if the Hamiltonian
                                     is for 0 qubits (e.g., an empty graph was provided).
    """
    # Generate the Pauli terms and coefficients for the Max-Cut Hamiltonian based on the graph structure.
    max_cut_paulis = build_max_cut_paulis(graph)

    # Handle cases where the graph might be empty or have no edges.
    if not max_cut_paulis and len(graph) > 0: # Graph has nodes but no edges
        print("Warning: The graph has nodes but no edges. The cost Hamiltonian will be effectively zero.")
        # Create a zero Hamiltonian (all identities) for the number of qubits in the graph.
        cost_hamiltonian = SparsePauliOp(["I"*len(graph)], coeffs=[0])
    elif not max_cut_paulis and len(graph) == 0: # Graph is completely empty (0 nodes, 0 edges)
        print("Warning: The graph is empty.")
        # Represent an operator on 0 qubits. QAOAAnsatz cannot be built for this.
        cost_hamiltonian = SparsePauliOp([], coeffs=[])
    else: # Graph has edges, construct Hamiltonian normally.
        cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    print(f"Cost Function Hamiltonian (num_qubits: {cost_hamiltonian.num_qubits}):\n{cost_hamiltonian.paulis}")

    # If the Hamiltonian is for 0 qubits (e.g., from an empty graph), a QAOA ansatz cannot be created.
    if cost_hamiltonian.num_qubits == 0 :
        print("Error: Cost Hamiltonian is for 0 qubits. Cannot create QAOAAnsatz.")
        return cost_hamiltonian, None

    # Create the QAOA ansatz circuit using the cost Hamiltonian and the specified number of layers.
    # The QAOAAnsatz class from Qiskit's circuit library handles the construction.
    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=num_layers)
    # Add measurements to all qubits in the ansatz. This is necessary for the Sampler primitive.
    ansatz.measure_all()

    # Decompose the high-level QAOAAnsatz object into a standard QuantumCircuit
    # composed of more fundamental gates. This is crucial for compatibility with
    # Qiskit Runtime primitives and backends, which expect circuits in this form.
    decomposed_qaoa_circuit = ansatz.decompose()
    print("QAOAAnsatz decomposed into a standard QuantumCircuit.")

    return cost_hamiltonian, decomposed_qaoa_circuit

def transpile_for_backend(circuit: QuantumCircuit, backend: Any, optimization_level: int = 1) -> QuantumCircuit:
    """
    Transpiles the given quantum circuit for a specific backend.
    This process adapts the circuit to the backend's native gate set and qubit connectivity (coupling map),
    and applies various optimization passes. If the circuit is smaller than the backend,
    an `initial_layout` is used to attempt to preserve the logical qubit count of the output circuit object,
    mapping the circuit's qubits to the first available physical qubits on the backend.

    Args:
        circuit: The `QuantumCircuit` to transpile. It should already be decomposed from any
                 higher-level circuit objects (like QAOAAnsatz).
        backend: The Qiskit backend object (e.g., `AerSimulator`, a `FakeBackendV2` instance,
                 or a real device object from `QiskitRuntimeService`).
        optimization_level: The Qiskit optimization level for transpilation (0, 1, 2, or 3).
                            Level 0: Minimal transpilation, mostly gate unrolling.
                            Level 1: Light optimization, includes routing. Good default.
                            Level 2: Heavier optimization.
                            Level 3: Most intensive optimization (can be slow).

    Returns:
        The transpiled `QuantumCircuit`, which is ISA-compatible and layout-aware for the target backend.
        Returns the original circuit if transpilation fails or is not possible (e.g., circuit too large).
    """
    backend_name_str = backend.name if hasattr(backend, 'name') else str(type(backend))

    # Determine the number of qubits on the backend safely.
    backend_num_qubits = 0
    if hasattr(backend, 'num_qubits') and backend.num_qubits is not None:
         backend_num_qubits = backend.num_qubits
    elif hasattr(backend, 'configuration'): # Fallback for BackendV1 style
         config = backend.configuration()
         if hasattr(config, 'n_qubits'):
             backend_num_qubits = config.n_qubits

    print(f"Transpiling circuit with {circuit.num_qubits} logical qubits for backend {backend_name_str} "
          f"({backend_num_qubits if backend_num_qubits > 0 else 'N/A'} physical qubits) "
          f"with optimization level {optimization_level}...")

    initial_layout_qiskit = None
    # If the circuit is smaller than the backend, create an initial layout
    # to map logical qubits 0..N-1 to physical qubits 0..N-1.
    # This helps ensure the transpiled circuit object is still defined over N logical qubits.
    if backend_num_qubits > 0 and circuit.num_qubits < backend_num_qubits:
        # Ensure we don't try to map to more physical qubits than available if the circuit is somehow larger
        # than the backend (though this case should be caught by the next check).
        num_physical_qubits_for_layout = min(circuit.num_qubits, backend_num_qubits)
        layout_dict = {i: i for i in range(num_physical_qubits_for_layout)}
        initial_layout_qiskit = Layout(layout_dict)
        print(f"Applying initial layout for {circuit.num_qubits} logical qubits onto the first "
              f"{num_physical_qubits_for_layout} physical qubits of the backend.")
    elif backend_num_qubits > 0 and circuit.num_qubits > backend_num_qubits:
        # This is a critical error: the circuit requires more qubits than the backend has.
        print(f"ERROR: Circuit has {circuit.num_qubits} qubits, but backend {backend_name_str} only has {backend_num_qubits}. "
              "Cannot transpile. Returning original (untranspiled) circuit.")
        return circuit # Return original circuit as transpilation is not feasible.

    try:
        # Use qiskit.compiler.transpile. It handles basis gates, coupling map, and optimizations
        # when a backend object is provided. `initial_layout` guides the mapping for smaller circuits.
        isa_circuit = qiskit_transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
            initial_layout=initial_layout_qiskit,
            # seed_transpiler=42 # Uncomment for reproducible transpilation during debugging
        )
    except Exception as e:
        print(f"Error during qiskit.compiler.transpile: {e}")
        print("Failed to transpile the circuit. Returning original (untranspiled) circuit.")
        return circuit # Return original on failure

    print(f"Circuit transpiled. Original logical qubits: {circuit.num_qubits}, Original operations: {circuit.count_ops()}")
    print(f"Transpiled circuit qubits: {isa_circuit.num_qubits}, Transpiled operations: {isa_circuit.count_ops()}")

    # Critical check: The transpilation should ideally preserve the logical qubit count of the circuit
    # when an initial_layout is used for smaller circuits on larger backends.
    # If this changes, it can lead to mismatches with observables defined for the original number of logical qubits.
    if isa_circuit.num_qubits != circuit.num_qubits:
        print(f"CRITICAL WARNING: Transpilation changed the number of qubits in the circuit object "
              f"from {circuit.num_qubits} (logical) to {isa_circuit.num_qubits}. "
              "This can cause a mismatch with observables if the observable is defined for the "
              "original logical qubit count. This might happen if 'initial_layout' was not fully effective "
              "or if the transpiler still expanded the circuit's qubit definition (e.g., by adding ancillas "
              "that are not uncomputed and removed, or if layout constraints forced a wider circuit). "
              "The observable is typically defined for {circuit.num_qubits} logical qubits.")

    return isa_circuit

def cost_func_estimator(
    params: np.ndarray,
    # This circuit is expected to be fully transpiled (ISA and layout compatible for its logical qubits)
    # but still parameterized. The Estimator will bind the numerical `params`.
    transpiled_parameterized_circuit: QuantumCircuit,
    cost_hamiltonian: SparsePauliOp,
    estimator: Estimator,
    callback_list: List[float] # List to store objective values for plotting
) -> float:
    """
    Cost function for the QAOA optimization loop, evaluated using the Estimator primitive.
    It takes numerical parameters, applies them to a pre-transpiled (but still parameterized)
    circuit, and computes the expectation value of the cost Hamiltonian.

    Args:
        params: A NumPy array of numerical parameter values (angles gamma and beta)
                for the current iteration of the classical optimizer.
        transpiled_parameterized_circuit: The QAOA circuit that has been decomposed and then
                                          transpiled for the target backend, but still contains
                                          its symbolic `Parameter` objects.
        cost_hamiltonian: The `SparsePauliOp` representing the cost function to be minimized.
                          Its number of qubits must match that of the `transpiled_parameterized_circuit`.
        estimator: The Qiskit Runtime `EstimatorV2` primitive.
        callback_list: A list to which the current cost value is appended, allowing
                       tracking of the optimization progress.

    Returns:
        The estimated cost (expectation value of the `cost_hamiltonian` for the
        given `params` and `circuit`).

    Raises:
        ValueError: If the qubit count of the circuit and the observable do not match.
    """
    # Critical check: Ensure the circuit (which should be N-qubit logical after transpilation)
    # matches the N-qubit observable.
    if transpiled_parameterized_circuit.num_qubits != cost_hamiltonian.num_qubits:
        error_msg = (f"FATAL ERROR in cost_func_estimator: Circuit qubit count ({transpiled_parameterized_circuit.num_qubits}) "
                     f"does not match observable qubit count ({cost_hamiltonian.num_qubits}). "
                     "This indicates a problem in the transpilation or problem setup, where the "
                     "transpiled circuit's logical width doesn't match the observable's definition.")
        print(error_msg)
        # For debugging this specific error:
        # print("Circuit details:", transpiled_parameterized_circuit.draw(output='text', fold=-1))
        # print("Observable details:", cost_hamiltonian)
        raise ValueError(error_msg)

    # Prepare the "Primitive Unified Bloc" (PUB) for the Estimator.
    # Format for EstimatorV2: (circuit, list_of_observables, list_of_parameter_value_sets)
    # Here, `params.tolist()` provides one set of parameter values.
    pub = (transpiled_parameterized_circuit, [cost_hamiltonian], [params.tolist()])

    try:
        # Run the job on the Estimator primitive.
        job = estimator.run(pubs=[pub])
        result = job.result() # This is a blocking call that waits for the job to complete.
    except Exception as e:
        print(f"Error during estimator.run(): {e}")
        print(f"Numerical parameters passed to estimator: {params.tolist()}")
        # Optionally, print circuit details if an error occurs here, for debugging.
        # print("Circuit QASM passed to estimator:\n", transpiled_parameterized_circuit.qasm())
        raise # Re-raise the exception to allow higher-level error handling or script termination.

    # Extract the expectation value from the result.
    # result[0] corresponds to the first (and only, in this case) PUB.
    # .data.evs contains the list of expectation values for the observables in that PUB.
    cost = result[0].data.evs[0]

    # Append the current cost to the callback list for tracking optimization progress.
    callback_list.append(cost)
    # Optional: print progress for each iteration of the optimizer.
    # print(f"Optimizer Iteration Parameters: {params}, Current Cost: {cost:.5f}")
    return cost

def optimize_qaoa_parameters(
    # This circuit should be fully transpiled for the target backend but still parameterized.
    transpiled_parameterized_circuit: QuantumCircuit,
    cost_hamiltonian: SparsePauliOp,
    initial_parameters: np.ndarray, # Initial guess for parameters (gammas and betas)
    estimator: Estimator # The configured Estimator primitive
) -> Any: # Returns a scipy.optimize.OptimizeResult object
    """
    Optimizes the parameters (gammas and betas) of the QAOA circuit using a classical
    optimizer (SciPy's COBYLA method).

    The `cost_func_estimator` is used as the objective function for the minimization.
    This function relies on the global `objective_func_vals_list` (within this module)
    to be populated by `cost_func_estimator` for tracking the cost evolution.

    Args:
        transpiled_parameterized_circuit: The QAOA circuit, decomposed and then fully transpiled
                                          for the target backend, but still containing its
                                          symbolic `Parameter` objects.
        cost_hamiltonian: The `SparsePauliOp` representing the cost function to be minimized.
        initial_parameters: A NumPy array providing the initial guess for the QAOA parameters.
                            The length should be 2 * num_layers (p gammas and p betas).
        estimator: The configured Qiskit Runtime `EstimatorV2` primitive.

    Returns:
        The result object from `scipy.optimize.minimize`. This object contains
        information about the optimization, including:
        - `x`: The optimal parameters found.
        - `fun`: The final value of the cost function.
        - `nfev`: The number of function evaluations.
        - `success`: A boolean indicating if the optimization terminated successfully.
    """
    print(f"Starting QAOA parameter optimization with initial parameters: {initial_parameters}")
    # Clear the global list that tracks cost values, ensuring a fresh start for this optimization run.
    objective_func_vals_list.clear()

    # Prepare additional arguments to be passed to the `cost_func_estimator`.
    # The first argument (`params`) is handled by `minimize`.
    args_for_cost_func = (transpiled_parameterized_circuit, cost_hamiltonian, estimator, objective_func_vals_list)

    # Use SciPy's minimize function. COBYLA (Constrained Optimization BY Linear Approximations)
    # is a common choice for QAOA as it's a gradient-free optimization algorithm.
    result = minimize(
        fun=cost_func_estimator,    # The objective function to minimize.
        x0=initial_parameters,      # Initial guess for the parameters.
        args=args_for_cost_func,    # Extra arguments passed to `fun` (after `x0`).
        method="COBYLA",            # Specifies the optimization algorithm.
        tol=1e-3,                   # Tolerance for termination.
        options={
            'maxiter': 100,         # Maximum number of iterations.
            'disp': False           # Set to True for more verbose output from the optimizer.
        }
    )
    print("\nQAOA Parameter Optimization Finished.")
    print(f"  Optimal parameters found: {result.x}")
    print(f"  Final cost (expectation value): {result.fun:.5f}")
    print(f"  Number of cost function evaluations (iterations): {result.nfev}")
    print(f"  Optimization success: {result.success}")
    print(f"  Optimizer message: {result.message}")
    return result

def sample_optimized_circuit(
    # This circuit should be fully transpiled for the backend
    # AND have the optimal numerical parameters bound to it.
    transpiled_bound_circuit: QuantumCircuit,
    sampler: Sampler, # The configured Sampler primitive
    num_shots: int   # Number of times to run the circuit for sampling
) -> dict:
    """
    Samples an optimized QAOA circuit (which has been transpiled and had its
    optimal numerical parameters bound) to obtain a distribution of measurement outcomes.

    Args:
        transpiled_bound_circuit: The QAOA circuit that is:
                                  1. Decomposed from `QAOAAnsatz`.
                                  2. Transpiled for the target backend (ISA and layout compatible).
                                  3. Has its optimal numerical parameters assigned (it's no longer parametric).
                                  4. Includes measurements on all relevant qubits.
        sampler: The configured Qiskit Runtime `SamplerV2` primitive.
        num_shots: The number of times the circuit is executed to collect measurement samples.
                   Higher shots lead to more accurate probability distributions.

    Returns:
        A dictionary where keys are bitstrings (in binary string format, e.g., "01101")
        representing measurement outcomes, and values are the corresponding counts (how many
        times each bitstring was observed).
    """
    print(f"Sampling optimized (transpiled and parameter-bound) circuit with {num_shots} shots...")

    # For SamplerV2, if the circuit passed already has all parameters bound to numerical values
    # (i.e., it's not parametric), the second element of the PUB (parameter_values) is not needed
    # or should be an empty list/tuple.
    pub = (transpiled_bound_circuit, ) # Circuit is already bound, no separate parameter values.

    # Run the sampler. The 'shots' argument is a run-time option for SamplerV2.
    job = sampler.run(pubs=[pub], shots=num_shots)
    result = job.result() # Wait for the job to complete and get the result.

    # Extract the measurement counts from the result of the first (and only) PUB.
    # .data.meas refers to the measurement outcome data.
    # .get_counts() returns the dictionary of bitstring: count.
    counts_bin = result[0].data.meas.get_counts()
    print("Sampling complete.")
    return counts_bin

def to_bitstring(integer_representation: int, num_bits: int) -> List[int]:
    """
    Converts a non-negative integer to its binary string representation, then to a list of bits (integers 0 or 1).
    The output list is MSB (Most Significant Bit) first.

    Args:
        integer_representation: The non-negative integer to convert.
        num_bits: The desired number of bits in the output bitstring. The output will be
                  padded with leading zeros if the binary representation is shorter.

    Returns:
        A list of integers (0 or 1) representing the bitstring.
        Example: `to_bitstring(5, 4)` returns `[0, 1, 0, 1]`.
                 `to_bitstring(1, 3)` returns `[0, 0, 1]`.
    """
    # `np.binary_repr` converts an integer to its binary string representation.
    # The `width` argument ensures padding with leading zeros to achieve the desired `num_bits`.
    bitstring_str = np.binary_repr(integer_representation, width=num_bits)
    # Convert each character (digit) in the binary string to an integer.
    return [int(digit) for digit in bitstring_str]
