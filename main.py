import matplotlib # Import matplotlib first
matplotlib.use('Agg') # Set the Agg backend before importing pyplot. Must be done before pyplot import.
import matplotlib.pyplot as plt # For creating and saving plots
import numpy as np
import rustworkx as rx # For graph creation and manipulation
from rustworkx.visualization import mpl_draw as draw_graph # For plotting graphs using matplotlib
from itertools import combinations # For generating complete graphs if needed
import os # For path operations (creating plot directories)

# Qiskit imports for runtime, primitives, and simulators
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
from qiskit_aer import AerSimulator
from qiskit.transpiler import Layout
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli

# Utility function imports from other project files
from graph_utils import (
    create_graph,
    evaluate_max_cut_solution
)
from qaoa_utils import (
    setup_qaoa_ansatz_and_hamiltonian,
    transpile_for_backend,
    optimize_qaoa_parameters,
    sample_optimized_circuit,
    objective_func_vals_list
)
from plotting_utils import (
    plot_cost_function_evolution,
    plot_results_distribution,
    plot_max_cut_solution
)
from execution_setup import (
    get_available_fake_backends_info,
    select_backend,
    determine_graph_parameters
)

PLOTS_BASE_DIR = "plots"


def main_qaoa_runner(num_nodes_config: int,
                     edge_list_config: list,
                     qaoa_layers: int,
                     num_shots_sampling: int,
                     estimator_shots: int,
                     use_real_backend_flag: bool = False,
                     chosen_backend_name: str = None,
                     use_backend_topology_as_graph: bool = False,
                     ibm_quantum_token_str: str = None,
                     qiskit_runtime_channel_config: str = None, # New parameter
                     min_qubits_for_least_busy_config: int = None # New parameter
                     ):
    """
    Main orchestrator function for QAOA Max-Cut.
    """
    active_backend = select_backend(
        use_real_backend_flag,
        chosen_backend_name,
        num_nodes_config, # This is num_nodes_config_for_auto_fake in select_backend
        ibm_quantum_token_str,
        qiskit_runtime_channel=qiskit_runtime_channel_config, # Pass channel
        min_qubits_for_least_busy=min_qubits_for_least_busy_config # Pass min qubits
    )
    if active_backend is None:
        print("Failed to select or initialize a backend. Exiting QAOA runner.")
        return

    # ... (rest of main_qaoa_runner remains the same as your latest working version) ...
    num_logical_nodes, actual_edge_list = determine_graph_parameters(
        active_backend, use_backend_topology_as_graph, num_nodes_config, edge_list_config
    )
    if num_logical_nodes == 0:
        print("Problem graph parameters could not be determined (0 nodes). Exiting QAOA runner.")
        return

    print(f"\n--- Problem Definition Finalized ---")
    print(f"Number of logical nodes for QAOA problem: {num_logical_nodes}")

    backend_total_qubits = active_backend.num_qubits if hasattr(active_backend, 'num_qubits') else num_logical_nodes
    if hasattr(active_backend, 'configuration'):
        conf = active_backend.configuration()
        if hasattr(conf, 'n_qubits'):
            backend_total_qubits = conf.n_qubits
    print(f"Target backend '{active_backend.name}' reports {backend_total_qubits} total qubits.")

    graph = create_graph(num_logical_nodes, actual_edge_list)

    backend_name_for_dir = (active_backend.name if hasattr(active_backend, 'name') else str(type(active_backend).__name__)).replace(" ", "_").replace(":", "_")
    model_plot_subdir = f"{backend_name_for_dir}_{num_logical_nodes}q_problem_{qaoa_layers}l"
    specific_plot_path = os.path.join(PLOTS_BASE_DIR, model_plot_subdir)
    os.makedirs(specific_plot_path, exist_ok=True)
    initial_graph_plot_filename = "initial_problem_graph.png"
    full_initial_graph_plot_path = os.path.join(specific_plot_path, initial_graph_plot_filename)
    plt.figure(figsize=(8,8));
    draw_graph(graph, node_size=600, with_labels=True, edge_labels=lambda edge: f"{edge:.0f}");
    plt.title(f"Logical Problem Graph ({num_logical_nodes} nodes, {graph.num_edges()} edges) for {backend_name_for_dir}")
    try:
        plt.savefig(full_initial_graph_plot_path);
        print(f"Initial graph plot saved to: {full_initial_graph_plot_path}")
    except Exception as e:
        print(f"Error saving initial graph plot: {e}")
    plt.close()

    print("\n--- Setting up N-qubit QAOA Ansatz and N-qubit Cost Hamiltonian ---")
    cost_hamiltonian_N_qubit, logical_qaoa_circuit_N_qubit = setup_qaoa_ansatz_and_hamiltonian(graph, qaoa_layers)
    if logical_qaoa_circuit_N_qubit is None:
        print("Error: QAOA circuit could not be created. Exiting QAOA runner.")
        return
    print(f"N-qubit logical QAOA circuit created for {logical_qaoa_circuit_N_qubit.num_qubits} qubits.")
    print(f"N-qubit cost Hamiltonian created for {cost_hamiltonian_N_qubit.num_qubits} qubits.")

    print("\n--- Configuring Primitives for Execution ---")
    estimator = None; sampler = None
    is_aer_simulator = isinstance(active_backend, AerSimulator)
    is_runtime_fake_backend = not is_aer_simulator and hasattr(active_backend, '__class__') and hasattr(active_backend.__class__, '__module__') and active_backend.__class__.__module__.startswith('qiskit_ibm_runtime.fake_provider')
    is_real_backend_for_session = use_real_backend_flag and active_backend is not None and not is_aer_simulator and not is_runtime_fake_backend and hasattr(active_backend, 'provider') and callable(getattr(active_backend, 'provider', None)) and active_backend.provider() is not None

    if is_real_backend_for_session:
        try:
            # Service is already initialized in select_backend if needed.
            # Here, we just use the active_backend with a Session.
            with Session(backend=active_backend) as session: # Pass service if not using default account
                print(f"Session opened with real backend: {active_backend.name}")
                estimator = RuntimeEstimatorV2(session=session)
                estimator.options.default_shots = estimator_shots
                estimator.options.dynamical_decoupling.enable = True
                estimator.options.dynamical_decoupling.sequence_type = "XY4"
                estimator.options.twirling.enable_gates = True
                estimator.options.twirling.num_randomizations = "auto"
                print("Estimator options (real backend):", estimator.options)

                sampler = RuntimeSamplerV2(session=session)
                print("Sampler options (real backend):", sampler.options)

                run_optimization_and_sampling(logical_qaoa_circuit_N_qubit, cost_hamiltonian_N_qubit, active_backend,
                                              estimator, sampler, qaoa_layers, num_logical_nodes, graph,
                                              num_shots_sampling, PLOTS_BASE_DIR, model_plot_subdir)
            print("Session closed."); return
        except Exception as e:
            print(f"Error during real backend session execution: {e}"); return
    else:
        backend_display_name = active_backend.name if hasattr(active_backend, 'name') else 'AerSimulator'
        print(f"Using backend for execution (mode-based): {backend_display_name}")
        try:
            estimator = RuntimeEstimatorV2(mode=active_backend)
            estimator.options.default_shots = estimator_shots
            if not is_aer_simulator:
                estimator.options.dynamical_decoupling.enable = True
                estimator.options.dynamical_decoupling.sequence_type = "XY4"
                estimator.options.twirling.enable_gates = True
                estimator.options.twirling.num_randomizations = "auto"
            print("Estimator options (sim/fake backend):", estimator.options)

            sampler = RuntimeSamplerV2(mode=active_backend)
            print("Sampler options (sim/fake backend):", sampler.options)

        except Exception as e:
             print(f"Error initializing primitives with backend {backend_display_name}: {e}"); return

    if estimator is None or sampler is None:
        print("Error: Estimator or Sampler could not be initialized. Exiting QAOA runner."); return

    run_optimization_and_sampling(
        logical_qaoa_circuit_N_qubit, cost_hamiltonian_N_qubit, active_backend, estimator, sampler,
        qaoa_layers, num_logical_nodes, graph,
        num_shots_sampling,
        PLOTS_BASE_DIR, model_plot_subdir
    )

# run_optimization_and_sampling function remains the same as your latest working version
def run_optimization_and_sampling(
    logical_qaoa_circuit_N_qubit,
    cost_hamiltonian_N_qubit,
    active_backend,
    estimator, sampler, qaoa_layers,
    num_logical_nodes,
    graph_N_node,
    num_shots_for_sampler,
    plots_base_dir: str, model_plot_subdir_name: str
    ):
    backend_name = active_backend.name if hasattr(active_backend, 'name') else type(active_backend).__name__
    print(f"\n--- Starting QAOA Execution on Backend: {backend_name} ---")

    print(f"Transpiling N-qubit logical QAOA circuit ({logical_qaoa_circuit_N_qubit.num_qubits} qubits) for M-qubit backend: {backend_name}...")
    transpiled_circuit_M_qubit, initial_layout = transpile_for_backend(
        logical_qaoa_circuit_N_qubit.copy(), active_backend, optimization_level=3 # Using opt_level 3 for QPU
    )

    if not isinstance(transpiled_circuit_M_qubit, QuantumCircuit):
        print(f"ERROR: transpile_for_backend did not return a QuantumCircuit as the first element. Got: {type(transpiled_circuit_M_qubit)}")
        return

    hamiltonian_for_estimator = cost_hamiltonian_N_qubit

    if transpiled_circuit_M_qubit.num_qubits != cost_hamiltonian_N_qubit.num_qubits:
        target_M_qubits = transpiled_circuit_M_qubit.num_qubits
        original_N_qubits = cost_hamiltonian_N_qubit.num_qubits

        if target_M_qubits > original_N_qubits:
            print(f"Expanding {original_N_qubits}-qubit Hamiltonian to {target_M_qubits} qubits.")
            try:
                identity_padding_str = "I" * (target_M_qubits - original_N_qubits)
                expanded_pauli_list = []
                for pauli_str_N, coeff in cost_hamiltonian_N_qubit.to_list(array=False):
                    pauli_str_M = identity_padding_str + pauli_str_N
                    expanded_pauli_list.append((pauli_str_M, coeff))

                if not expanded_pauli_list and original_N_qubits == 0 :
                     hamiltonian_for_estimator = SparsePauliOp(["I"*target_M_qubits], coeffs=[0]) if target_M_qubits > 0 else SparsePauliOp([], coeffs=[])
                elif expanded_pauli_list:
                     hamiltonian_for_estimator = SparsePauliOp.from_list(expanded_pauli_list)
                else:
                     print(f"Warning: Original {original_N_qubits}-qubit Hamiltonian led to empty expanded list for {target_M_qubits} qubits. Creating zero M-qubit Hamiltonian.")
                     hamiltonian_for_estimator = SparsePauliOp(["I"*target_M_qubits], coeffs=[0]) if target_M_qubits > 0 else SparsePauliOp([], coeffs=[])

                print(f"Hamiltonian expanded to {hamiltonian_for_estimator.num_qubits} qubits.")
                if hamiltonian_for_estimator.num_qubits != target_M_qubits:
                    print(f"ERROR: Expanded Hamiltonian has {hamiltonian_for_estimator.num_qubits} qubits, expected {target_M_qubits}.")
                    return
            except Exception as e:
                print(f"ERROR: Failed to manually expand Hamiltonian: {e}")
                return
        elif target_M_qubits < original_N_qubits:
            print(f"ERROR: Transpiled circuit ({target_M_qubits}q) has fewer qubits than original Hamiltonian ({original_N_qubits}q). This is unexpected.")
            return
    else:
        print(f"Hamiltonian and transpiled circuit have matching qubit counts ({transpiled_circuit_M_qubit.num_qubits}). No Hamiltonian expansion needed.")

    initial_gamma = np.array([np.pi / 4] * qaoa_layers)
    initial_beta = np.array([np.pi / 2] * qaoa_layers)
    initial_params = np.concatenate((initial_gamma, initial_beta))

    print("Optimizing QAOA parameters using COBYLA optimizer...")
    optimization_result = optimize_qaoa_parameters(
        transpiled_circuit_M_qubit,
        hamiltonian_for_estimator,
        initial_params,
        estimator
    )
    if optimization_result is None:
        print("Optimization failed or returned None. Exiting.")
        return
    optimal_params = optimization_result.x

    transpiled_bound_circuit_M_qubit = transpiled_circuit_M_qubit.assign_parameters(optimal_params)

    print(f"Sampling the optimized M-qubit circuit ({transpiled_bound_circuit_M_qubit.num_qubits} qubits) with {num_shots_for_sampler} shots...")
    counts_bin = sample_optimized_circuit(transpiled_bound_circuit_M_qubit, sampler, num_shots_for_sampler)

    print("\n--- Processing and Plotting Results ---")
    # ... (result processing and plotting remains the same) ...
    cost_filename = f"qaoa_cost_evolution.png"
    dist_filename = f"qaoa_results_distribution.png"
    plot_cost_function_evolution(objective_func_vals_list, base_save_dir=plots_base_dir, model_subdir=model_plot_subdir_name, filename=cost_filename)

    if not counts_bin:
        print("Error: No measurement counts obtained. Cannot determine or plot solution.")
        return

    plot_results_distribution(counts_bin, top_n=10, base_save_dir=plots_base_dir, model_subdir=model_plot_subdir_name, filename=dist_filename)

    most_likely_bitstring_str = max(counts_bin, key=counts_bin.get)

    if len(most_likely_bitstring_str) >= num_logical_nodes :
        solution_bitstring_N_logical_str = most_likely_bitstring_str[-num_logical_nodes:] if num_logical_nodes > 0 else ""
    elif len(most_likely_bitstring_str) >= num_logical_nodes:
         print(f"Warning: Sampled bitstring length ({len(most_likely_bitstring_str)}) does not match M-qubit circuit ({transpiled_bound_circuit_M_qubit.num_qubits}). "
               f"Attempting to extract last {num_logical_nodes} bits for N-logical part.")
         solution_bitstring_N_logical_str = most_likely_bitstring_str[-num_logical_nodes:] if num_logical_nodes > 0 else ""
    else:
        print(f"Warning: Sampled bitstring length ({len(most_likely_bitstring_str)}) is less than num_logical_nodes ({num_logical_nodes}). Cannot reliably extract N-logical part. Using full string.")
        solution_bitstring_N_logical_str = most_likely_bitstring_str

    solution_bitstring_list_N_logical = [int(b) for b in solution_bitstring_N_logical_str]

    print(f"Most likely M-qubit solution bitstring: {most_likely_bitstring_str}")
    print(f"Extracted N-logical solution bitstring: {solution_bitstring_N_logical_str} (list: {solution_bitstring_list_N_logical})")

    if len(solution_bitstring_list_N_logical) != num_logical_nodes and num_logical_nodes > 0 :
        print(f"Warning: Extracted N-logical bitstring length ({len(solution_bitstring_list_N_logical)}) != logical graph nodes ({num_logical_nodes}). This is unexpected after extraction logic.")
        final_solution_for_evaluation = solution_bitstring_list_N_logical[:num_logical_nodes]
        while len(final_solution_for_evaluation) < num_logical_nodes: final_solution_for_evaluation.append(0)
    elif num_logical_nodes == 0:
        final_solution_for_evaluation = []
    else:
        final_solution_for_evaluation = solution_bitstring_list_N_logical

    if num_logical_nodes == 0:
        empty_graph_plot_filename = "qaoa_max_cut_empty_graph.png"
        print("Info: Graph has 0 nodes. No cut value to evaluate.");
        plot_max_cut_solution(graph_N_node, [], title="Max-Cut Solution (Empty Graph)", base_save_dir=plots_base_dir, model_subdir=model_plot_subdir_name, filename=empty_graph_plot_filename);
        return

    cut_value = evaluate_max_cut_solution(final_solution_for_evaluation, graph_N_node)
    print(f"Max-Cut value for the N-logical solution: {cut_value:.3f}")

    solution_plot_filename = f"qaoa_max_cut_solution.png"
    plot_max_cut_solution(graph_N_node, final_solution_for_evaluation,
                          title=f"Max-Cut Solution on {backend_name} (N-logical: {num_logical_nodes}q, Cut: {cut_value:.2f})",
                          base_save_dir=plots_base_dir, model_subdir=model_plot_subdir_name,
                          filename=solution_plot_filename)


if __name__ == "__main__":
    get_available_fake_backends_info()

    # --- USER CONFIGURATION ---
    param_num_nodes_config = 5
    param_edge_list_config = [
        (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
        (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)
    ]
    param_use_real_backend = True # Set to True to use a real QPU
    # To use least_busy: set this to "least_busy"
    # To use a specific QPU: set this to its name, e.g., "ibm_kyoto"
    param_chosen_backend_name = "least_busy"

    param_use_backend_topology = False # Recommended to True for real QPU if problem can match device size
                                      # Set to False to run your custom graph on the (potentially larger) QPU

    param_qaoa_layers = 2
    param_num_shots_sampling = 4096
    param_estimator_shots = 1000

    # 7. IBM Quantum Token (only needed if param_use_real_backend is True and token not saved)
    param_ibm_token = None
    # 8. Qiskit Runtime Service channel ('ibm_quantum' or 'ibm_cloud', or None for default)
    param_qiskit_runtime_channel = "ibm_quantum" # change if your default is ibm_cloud
    # 9. Min qubits for 'least_busy' real backend selection (if chosen_backend_name is "least_busy")
    param_min_qubits_for_least_busy = 127 # Notebook used 127; adjust as needed, or None for any size
    # --- End of User Configuration ---

    if param_use_real_backend and not param_chosen_backend_name:
        print("Error: 'param_use_real_backend' is True, but 'param_chosen_backend_name' is not set. "
              "Please specify a real backend name or 'least_busy'.")
    else:
        print(f"\nStarting QAOA for Max-Cut...")
        main_qaoa_runner(
            num_nodes_config=param_num_nodes_config,
            edge_list_config=param_edge_list_config,
            qaoa_layers=param_qaoa_layers,
            num_shots_sampling=param_num_shots_sampling,
            estimator_shots=param_estimator_shots,
            use_real_backend_flag=param_use_real_backend,
            chosen_backend_name=param_chosen_backend_name,
            use_backend_topology_as_graph=param_use_backend_topology,
            ibm_quantum_token_str=param_ibm_token,
            qiskit_runtime_channel_config=param_qiskit_runtime_channel, # Pass new param
            min_qubits_for_least_busy_config=param_min_qubits_for_least_busy # Pass new param
        )
        print("\nQAOA execution finished.")

