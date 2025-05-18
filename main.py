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
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2 # Alias for clarity
from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2   # Alias for clarity
from qiskit_aer import AerSimulator # For local, noise-free simulation

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
    objective_func_vals_list # Global list from qaoa_utils for tracking cost
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

# Define a base directory for saving all generated plots
PLOTS_BASE_DIR = "plots" # All plot subdirectories will be created under this folder.


def main_qaoa_runner(num_nodes_config: int,
                     edge_list_config: list,
                     qaoa_layers: int,
                     num_shots_sampling: int,
                     estimator_shots: int,
                     use_real_backend_flag: bool = False,
                     chosen_backend_name: str = None,
                     use_backend_topology_as_graph: bool = False,
                     ibm_quantum_token_str: str = None
                     ):
    """
    Main orchestrator function to run the Quantum Approximate Optimization Algorithm (QAOA)
    for a Max-Cut problem.
    """

    # --- Step 0: Select Backend ---
    active_backend = select_backend(
        use_real_backend_flag, chosen_backend_name, num_nodes_config, ibm_quantum_token_str
    )
    if active_backend is None:
        print("Failed to select or initialize a backend. Exiting QAOA runner.")
        return

    # --- Step 1: Determine Graph Parameters ---
    actual_num_nodes, actual_edge_list = determine_graph_parameters(
        active_backend, use_backend_topology_as_graph, num_nodes_config, edge_list_config
    )
    if actual_num_nodes == 0:
        print("Problem graph parameters could not be determined (0 nodes). Exiting QAOA runner.")
        return

    print(f"\n--- Problem Definition Finalized ---")
    print(f"Actual number of nodes for QAOA: {actual_num_nodes}")

    # Create the graph instance
    graph = create_graph(actual_num_nodes, actual_edge_list)

    # Prepare directory and filename for saving the initial graph plot
    backend_name_for_dir = (active_backend.name if hasattr(active_backend, 'name') else str(type(active_backend).__name__)).replace(" ", "_").replace(":", "_")
    model_plot_subdir = f"{backend_name_for_dir}_{actual_num_nodes}q_{qaoa_layers}l"
    specific_plot_path = os.path.join(PLOTS_BASE_DIR, model_plot_subdir)
    os.makedirs(specific_plot_path, exist_ok=True)

    initial_graph_plot_filename = "initial_problem_graph.png"
    full_initial_graph_plot_path = os.path.join(specific_plot_path, initial_graph_plot_filename)

    # Plot and save the initial problem graph
    plt.figure(figsize=(8,8))
    draw_graph(graph, node_size=600, with_labels=True, edge_labels=lambda edge: f"{edge:.0f}")
    plt.title(f"Problem Graph ({actual_num_nodes} nodes, {graph.num_edges()} edges) for {backend_name_for_dir}")
    try:
        plt.savefig(full_initial_graph_plot_path)
        print(f"Initial graph plot saved to: {full_initial_graph_plot_path}")
    except Exception as e:
        print(f"Error saving initial graph plot to {full_initial_graph_plot_path}: {e}")
    plt.close()

    # --- Step 2: Setup QAOA Ansatz and Cost Hamiltonian ---
    print("\n--- Setting up QAOA Ansatz (decomposed) and Cost Hamiltonian ---")
    cost_hamiltonian, logical_qaoa_circuit = setup_qaoa_ansatz_and_hamiltonian(graph, qaoa_layers)
    if logical_qaoa_circuit is None:
        print("Error: QAOA circuit could not be created. Exiting QAOA runner.")
        return
    print(f"Logical QAOA circuit created for {logical_qaoa_circuit.num_qubits} qubits (matching graph).")

    # --- Step 3: Initialize Primitives (Estimator & Sampler) for Execution ---
    print("\n--- Configuring Primitives for Execution ---")
    estimator = None
    sampler = None

    is_aer_simulator = isinstance(active_backend, AerSimulator)
    is_runtime_fake_backend = not is_aer_simulator and \
                              hasattr(active_backend, '__class__') and \
                              hasattr(active_backend.__class__, '__module__') and \
                              active_backend.__class__.__module__.startswith('qiskit_ibm_runtime.fake_provider')

    is_real_backend_for_session = use_real_backend_flag and \
                                  active_backend is not None and \
                                  not is_aer_simulator and \
                                  not is_runtime_fake_backend and \
                                  hasattr(active_backend, 'provider') and \
                                  callable(getattr(active_backend, 'provider', None)) and \
                                  active_backend.provider() is not None

    if is_real_backend_for_session:
        try:
            service = QiskitRuntimeService()
            with Session(service=service, backend=active_backend) as session:
                print(f"Session opened with real backend: {active_backend.name}")
                estimator = RuntimeEstimatorV2(session=session)
                estimator.options.default_shots = estimator_shots
                sampler = RuntimeSamplerV2(session=session)

                run_optimization_and_sampling(logical_qaoa_circuit, cost_hamiltonian, active_backend,
                                              estimator, sampler, qaoa_layers, actual_num_nodes, graph,
                                              num_shots_sampling, PLOTS_BASE_DIR, model_plot_subdir)
            print("Session closed.")
            return
        except Exception as e:
            print(f"Error during real backend session execution: {e}")
            return
    else:
        backend_display_name = active_backend.name if hasattr(active_backend, 'name') else 'AerSimulator'
        print(f"Using backend for execution (mode-based): {backend_display_name}")
        try:
            estimator = RuntimeEstimatorV2(mode=active_backend)
            estimator.options.default_shots = estimator_shots
            sampler = RuntimeSamplerV2(mode=active_backend)
        except Exception as e:
             print(f"Error initializing primitives with backend {backend_display_name}: {e}")
             return

    if estimator is None or sampler is None:
        print("Error: Estimator or Sampler could not be initialized. Exiting QAOA runner.")
        return

    run_optimization_and_sampling(
        logical_qaoa_circuit, cost_hamiltonian, active_backend, estimator, sampler,
        qaoa_layers, actual_num_nodes, graph, num_shots_sampling,
        PLOTS_BASE_DIR, model_plot_subdir
    )


def run_optimization_and_sampling(
    logical_qaoa_circuit, cost_hamiltonian, active_backend,
    estimator, sampler, qaoa_layers, num_nodes, graph, num_shots_for_sampler,
    plots_base_dir: str, model_plot_subdir_name: str
    ):
    """
    Helper function to execute the core QAOA steps: transpilation, optimization, and sampling.
    """
    backend_name = active_backend.name if hasattr(active_backend, 'name') else type(active_backend).__name__
    print(f"\n--- Starting QAOA Execution on Backend: {backend_name} ---")

    print(f"Transpiling logical QAOA circuit ({logical_qaoa_circuit.num_qubits} qubits) for backend: {backend_name}...")
    transpiled_parameterized_circuit = transpile_for_backend(logical_qaoa_circuit.copy(), active_backend, optimization_level=1)

    initial_gamma = np.array([np.pi / 4] * qaoa_layers)
    initial_beta = np.array([np.pi / 2] * qaoa_layers)
    initial_params = np.concatenate((initial_gamma, initial_beta))

    print("Optimizing QAOA parameters using COBYLA optimizer...")
    optimization_result = optimize_qaoa_parameters(
        transpiled_parameterized_circuit, cost_hamiltonian, initial_params, estimator
    )
    optimal_params = optimization_result.x

    transpiled_bound_circuit = transpiled_parameterized_circuit.assign_parameters(optimal_params)

    print(f"Sampling the optimized (transpiled and bound) circuit with {num_shots_for_sampler} shots...")
    counts_bin = sample_optimized_circuit(transpiled_bound_circuit, sampler, num_shots_for_sampler)

    # --- Process and Plot Results ---
    print("\n--- Processing and Plotting Results ---")
    cost_filename = f"qaoa_cost_evolution.png"
    dist_filename = f"qaoa_results_distribution.png"

    plot_cost_function_evolution(objective_func_vals_list,
                                 base_save_dir=plots_base_dir,
                                 model_subdir=model_plot_subdir_name,
                                 filename=cost_filename)

    if not counts_bin:
        print("Error: No measurement counts obtained from sampling. Cannot determine or plot solution.")
        return

    plot_results_distribution(counts_bin, top_n=10,
                              base_save_dir=plots_base_dir,
                              model_subdir=model_plot_subdir_name,
                              filename=dist_filename)

    most_likely_bitstring_str = max(counts_bin, key=counts_bin.get)
    solution_bitstring_list = [int(b) for b in most_likely_bitstring_str]

    print(f"Most likely solution bitstring: {most_likely_bitstring_str} (list: {solution_bitstring_list})")

    final_solution_for_evaluation = solution_bitstring_list
    if len(solution_bitstring_list) != num_nodes and num_nodes > 0 :
        print(f"Warning: Bitstring length ({len(solution_bitstring_list)}) != graph nodes ({num_nodes}). Truncating/padding might occur for evaluation.")
        final_solution_for_evaluation = solution_bitstring_list[:num_nodes]
        if len(final_solution_for_evaluation) < num_nodes:
            print(f"Error: Adjusted bitstring is too short for evaluation. Expected {num_nodes} bits.")
            return
    elif num_nodes == 0:
        empty_graph_plot_filename = "qaoa_max_cut_empty_graph.png"
        print("Info: Graph has 0 nodes. No cut value to evaluate.")
        plot_max_cut_solution(graph, [], title="Max-Cut Solution (Empty Graph)",
                              base_save_dir=plots_base_dir, model_subdir=model_plot_subdir_name,
                              filename=empty_graph_plot_filename)
        return

    cut_value = evaluate_max_cut_solution(final_solution_for_evaluation, graph)
    print(f"Max-Cut value for the most likely solution: {cut_value:.3f}")

    solution_plot_filename = f"qaoa_max_cut_solution.png"
    plot_max_cut_solution(graph, final_solution_for_evaluation,
                          title=f"Max-Cut Solution on {backend_name} (Cut: {cut_value:.2f})",
                          base_save_dir=plots_base_dir, model_subdir=model_plot_subdir_name,
                          filename=solution_plot_filename)


if __name__ == "__main__":
    # Display available fake backends to help user choose
    get_available_fake_backends_info()

    # --- USER CONFIGURATION ---
    # 1. User's preferred number of nodes.
    param_num_nodes_config = 5 # Example: 5 nodes

    # 2. User's preferred edge list.
    #    If a specific backend dictates num_nodes, and use_backend_topology is False,
    #    a complete graph for that backend's size will be generated by determine_graph_parameters.
    #    If param_use_backend_topology is True, this config is ignored if a backend topology is found.
    if param_num_nodes_config > 1:
        # Default to a complete graph if no specific topology is used or derived
        param_edge_list_config = [(u, v, 1.0) for u, v in combinations(range(param_num_nodes_config), 2)]
    else:
        param_edge_list_config = []

    # 3. Choose if using a real backend or a fake one
    param_use_real_backend = False

    # 4. Specify backend name (real or fake).
    param_chosen_backend_name = "FakeBurlingtonV2"
    # param_chosen_backend_name = "auto"
    # param_chosen_backend_name = None # To default to AerSimulator

    # 5. Choose if the graph topology should be derived from the backend
    param_use_backend_topology = True # Set to True to use backend's coupling map

    # 6. QAOA specific parameters
    param_qaoa_layers = 2
    param_num_shots_sampling = 4096
    param_estimator_shots = 1024

    # 7. IBM Quantum Token (only needed if param_use_real_backend is True and token not saved)
    param_ibm_token = None
    # --- End of User Configuration ---

    if param_use_real_backend and not param_chosen_backend_name:
        print("Error: 'param_use_real_backend' is True, but 'param_chosen_backend_name' is not set. "
              "Please specify a real backend name.")
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
            ibm_quantum_token_str=param_ibm_token
        )
        print("\nQAOA execution finished.")

