import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit import QuantumCircuit, Parameter, ParameterExpression
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit.compiler import transpile as qiskit_transpile
from qiskit.transpiler import Layout, Target, CouplingMap
from qiskit.circuit.library import CXGate, RZGate, SXGate, XGate, IGate, Measure
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from typing import List, Tuple, Any, Callable

from graph_utils import build_max_cut_paulis

objective_func_vals_list: List[float] = []

def setup_qaoa_ansatz_and_hamiltonian(graph: Any, num_layers: int) -> Tuple[SparsePauliOp, QuantumCircuit | None]:
    """
    Sets up the Max-Cut problem: N-qubit cost Hamiltonian and N-qubit parameterized QAOA ansatz,
    where N is the number of nodes in the graph.
    Ensures PauliEvolutionGate is decomposed for compatibility.
    """
    num_logical_qubits = len(graph.nodes())
    max_cut_paulis = build_max_cut_paulis(graph)

    if not max_cut_paulis and num_logical_qubits > 0:
        print(f"Warning: The graph has {num_logical_qubits} nodes but no edges. Cost Hamiltonian will be effectively zero.")
        cost_hamiltonian = SparsePauliOp(["I"*num_logical_qubits], coeffs=[0])
    elif num_logical_qubits == 0:
        print("Warning: The graph is empty (0 nodes).")
        cost_hamiltonian = SparsePauliOp([], coeffs=[])
    else:
        cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

    print(f"Cost Function Hamiltonian (initially {cost_hamiltonian.num_qubits} qubits, for {num_logical_qubits}-node graph):\n{cost_hamiltonian.paulis}")

    if num_logical_qubits == 0 :
        print("Error: Graph has 0 nodes. Cannot create QAOAAnsatz.")
        return cost_hamiltonian, None

    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=num_layers)

    decomposed_qaoa_circuit = ansatz.decompose()
    print(f"QAOAAnsatz (for {num_logical_qubits} logical qubits) initially decomposed.")

    aer_basis_gates = ['p', 'u', 'rz', 'ry', 'rx', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'x', 'y', 'z', 'id', 'cx', 'cy', 'cz', 'swap', 'measure', 'reset']

    print(f"Attempting to further decompose {num_logical_qubits}-qubit circuit into basis for Aer compatibility...")
    try:
        fully_decomposed_circuit = qiskit_transpile(decomposed_qaoa_circuit,
                                                    basis_gates=aer_basis_gates,
                                                    optimization_level=0)
        print("Circuit further decomposed for Aer compatibility.")
        logical_qaoa_circuit = fully_decomposed_circuit
    except Exception as e:
        print(f"Warning: Could not further decompose circuit for Aer: {e}. Using circuit after initial decomposition.")
        logical_qaoa_circuit = decomposed_qaoa_circuit

    has_measurements = any(instruction.operation.name == 'measure' for instruction in logical_qaoa_circuit.data)
    if not has_measurements and logical_qaoa_circuit.num_qubits > 0 :
        print("Adding measurements to the final logical QAOA circuit.")
        logical_qaoa_circuit.measure_all(inplace=True)

    return cost_hamiltonian, logical_qaoa_circuit # Returns N-qubit Hamiltonian and N-qubit Circuit

def transpile_for_backend(circuit: QuantumCircuit, backend: Any, optimization_level: int = 3) -> Tuple[QuantumCircuit, Layout | None]:
    """
    Transpiles the N-qubit logical circuit for an M-qubit backend.
    If M >= N, an initial_layout maps logical qubits to the first N physical qubits.
    The output circuit will have M qubits.
    Returns the M-qubit transpiled circuit and the N-to-M initial_layout used.
    """
    backend_name_str = backend.name if hasattr(backend, 'name') else str(type(backend))

    logical_num_qubits = circuit.num_qubits # N

    backend_reported_qubits = 0 # M
    if hasattr(backend, 'num_qubits') and backend.num_qubits is not None:
         backend_reported_qubits = backend.num_qubits
    elif hasattr(backend, 'configuration'):
         config = backend.configuration()
         if hasattr(config, 'n_qubits'):
             backend_reported_qubits = config.n_qubits

    print(f"Transpiling {logical_num_qubits}-qubit logical circuit for backend {backend_name_str} "
          f"({backend_reported_qubits if backend_reported_qubits > 0 else 'N/A'} reported physical qubits) "
          f"with optimization level {optimization_level}...")

    initial_layout_qiskit = None

    if logical_num_qubits == 0:
        print("Warning: Transpiling an empty (0-qubit) circuit. Returning as is.")
        return circuit, None

    if backend_reported_qubits == 0:
        # This case should ideally be prevented by select_backend for AerSimulator
        print(f"Warning: Backend {backend_name_str} reports 0 qubits. Cannot effectively transpile. "
              f"Using logical circuit as is, which might cause issues later if it's not empty.")
        return circuit, None

    if logical_num_qubits > backend_reported_qubits:
        print(f"ERROR: Logical circuit has {logical_num_qubits} qubits, but backend {backend_name_str} only reports {backend_reported_qubits}. "
              "Cannot transpile. Returning original circuit.")
        return circuit, None

    # Create initial layout mapping N logical Qubit objects to physical integer indices 0 to N-1.
    layout_dict = {circuit.qubits[i]: i for i in range(logical_num_qubits)}
    try:
        initial_layout_qiskit = Layout(layout_dict)
        print(f"Applying initial layout: logical qubits {list(range(logical_num_qubits))} mapped to physical qubits {list(range(logical_num_qubits))}.")
    except Exception as e:
        print(f"Warning: Could not create initial Layout: {e}. Proceeding without explicit initial_layout.")
        initial_layout_qiskit = None

    try:
        target_for_transpilation = None
        backend_for_transpilation = backend
        if hasattr(backend, 'target') and backend.target is not None:
            target_for_transpilation = backend.target
            backend_for_transpilation = None

        isa_circuit = qiskit_transpile(
            circuit,
            backend=backend_for_transpilation,
            target=target_for_transpilation,
            optimization_level=optimization_level,
            initial_layout=initial_layout_qiskit,
        )
    except Exception as e:
        print(f"Error during qiskit.compiler.transpile: {e}")
        print("Failed to transpile the circuit. Returning original circuit and no layout.")
        return circuit, None

    print(f"Circuit transpiled. Original logical qubits: {logical_num_qubits}, Original operations: {circuit.count_ops()}")
    print(f"Transpiled circuit qubits: {isa_circuit.num_qubits} (should match backend's reported qubits if M>=N). Transpiled operations: {isa_circuit.count_ops()}")

    if isa_circuit.num_qubits != backend_reported_qubits and backend_reported_qubits >= logical_num_qubits :
         print(f"Warning: Transpiled circuit has {isa_circuit.num_qubits} qubits, but backend reports {backend_reported_qubits}. This is unexpected when M>=N.")

    return isa_circuit, initial_layout_qiskit # Returns M-qubit circuit and N-to-M Layout

def cost_func_estimator(
    params: np.ndarray,
    transpiled_parameterized_circuit: QuantumCircuit, # M-qubit circuit
    expanded_hamiltonian: SparsePauliOp,             # M-qubit Hamiltonian
    estimator: Estimator,
    callback_list: List[float]
) -> float:
    """
    Cost function. Expects circuit and Hamiltonian to have matching M qubit counts.
    """
    if transpiled_parameterized_circuit.num_qubits != expanded_hamiltonian.num_qubits:
        error_msg = (f"FATAL ERROR in cost_func_estimator: Transpiled circuit qubit count ({transpiled_parameterized_circuit.num_qubits}) "
                     f"does not match expanded Hamiltonian qubit count ({expanded_hamiltonian.num_qubits}).")
        print(error_msg)
        raise ValueError(error_msg)

    pub = (transpiled_parameterized_circuit, [expanded_hamiltonian], [params.tolist()])

    try:
        job = estimator.run(pubs=[pub])
        result = job.result()
    except Exception as e:
        print(f"Error during estimator.run(): {e}")
        print(f"Numerical parameters passed to estimator: {params.tolist()}")
        raise

    cost = result[0].data.evs[0]
    callback_list.append(cost)
    return cost

def optimize_qaoa_parameters(
    transpiled_parameterized_circuit: QuantumCircuit, # M-qubit
    expanded_hamiltonian: SparsePauliOp,              # M-qubit
    initial_parameters: np.ndarray,
    estimator: Estimator
) -> Any:
    """
    Optimizes QAOA parameters. Expects M-qubit circuit and M-qubit Hamiltonian.
    """
    print(f"Starting QAOA parameter optimization with initial parameters: {initial_parameters}")
    objective_func_vals_list.clear()

    args_for_cost_func = (transpiled_parameterized_circuit, expanded_hamiltonian, estimator, objective_func_vals_list)

    result = minimize(
        fun=cost_func_estimator,
        x0=initial_parameters,
        args=args_for_cost_func,
        method="COBYLA",
        tol=1e-3,
        options={'maxiter': 100, 'disp': False}
    )
    print("\nQAOA Parameter Optimization Finished.")
    print(f"  Optimal parameters found: {result.x}")
    print(f"  Final cost (expectation value): {result.fun:.5f}")
    print(f"  Number of cost function evaluations (iterations): {result.nfev}")
    print(f"  Optimization success: {result.success}")
    print(f"  Optimizer message: {result.message}")
    return result

def sample_optimized_circuit(
    transpiled_bound_circuit: QuantumCircuit, # M-qubit
    sampler: Sampler,
    num_shots: int
) -> dict:
    """
    Samples an optimized QAOA circuit. Expects M-qubit circuit.
    """
    print(f"Sampling optimized (transpiled and parameter-bound) M-qubit circuit ({transpiled_bound_circuit.num_qubits} qubits) with {num_shots} shots...")
    pub = (transpiled_bound_circuit, )
    job = sampler.run(pubs=[pub], shots=num_shots)
    result = job.result()
    counts_bin = result[0].data.meas.get_counts()
    print("Sampling complete.")
    return counts_bin

def to_bitstring(integer_representation: int, num_bits: int) -> List[int]:
    bitstring_str = np.binary_repr(integer_representation, width=num_bits)
    return [int(digit) for digit in bitstring_str]

