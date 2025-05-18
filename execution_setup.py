import inspect # For discovering fake backends
from itertools import combinations # For generating complete graphs if needed

# Qiskit imports
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
import qiskit_ibm_runtime.fake_provider as fp_module # Import the module for dynamic loading
from qiskit.transpiler import CouplingMap # Ensure CouplingMap is imported

# Try to import FakeBackendV2 for type checking, but handle if it's not found (e.g. very old qiskit-terra)
try:
    # This is for type checking, actual fake backends come from fp_module
    from qiskit.providers.fake_provider import FakeBackendV2
except ImportError:
    print("Warning: Could not import FakeBackendV2 from qiskit.providers.fake_provider. Type checks against it may be limited.")
    class FakeBackendV2: pass # Dummy class for isinstance check if actual import fails


def get_available_fake_backends_info():
    """
    Lists known fake backends from qiskit_ibm_runtime.fake_provider and their qubit counts
    to guide user selection. This function provides a curated list for clarity.

    Returns:
        dict: A dictionary where keys are qubit counts and values are lists of
              backend name strings with comments about their general topology.
    """
    available_backends_by_qubits = {}
    # This predefined dictionary is more reliable than dynamic inspection for qubit counts
    # without instantiating every backend. It should be kept updated with common V2 fakes.
    known_fake_backends_info = {
        "FakeArmonkV2": {"qubits": 1, "comment": "Single qubit"},
        "FakeManilaV2": {"qubits": 5, "comment": "Often linear or star-like topology"},
        "FakeVigoV2": {"qubits": 5, "comment": "Similar to Manila, 5Q"},
        "FakeAthensV2": {"qubits": 5, "comment": "Similar to Manila, 5Q"},
        "FakeLimaV2": {"qubits": 5, "comment": "Similar to Manila, 5Q"},
        "FakeQuitoV2": {"qubits": 5, "comment": "Similar to Manila, 5Q"},
        "FakeEssexV2": {"qubits": 5, "comment": "Similar to Manila, 5Q"},
        "FakeRomeV2": {"qubits": 5, "comment": "Similar to Manila, 5Q"},
        "FakeYorktownV2": {"qubits": 5, "comment": "Similar to Manila, 5Q"},
        "FakeBurlingtonV2": {"qubits": 5, "comment": "T-shape or line with branch"},
        "FakeGuadalupeV2": {"qubits": 16, "comment": "Heavy-hexagon lattice fragment"},
        "FakeMelbourneV2": {"qubits": 16, "comment": "Ladder-like or grid topology (may be V1 style)"},
        "FakeAlmadenV2": {"qubits": 20, "comment": "Grid-like topology"},
        "FakeSingaporeV2": {"qubits": 20, "comment": "Grid-like topology"},
        "FakeKolkataV2": {"qubits": 27, "comment": "Heavy-hexagon lattice fragment"},
        "FakeMontrealV2": {"qubits": 27, "comment": "Heavy-hexagon lattice fragment"},
        "FakeCairoV2": {"qubits": 27, "comment": "Heavy-hexagon lattice fragment"},
        "FakeSydneyV2": {"qubits": 27, "comment": "Heavy-hexagon lattice fragment"},
        "FakeTorontoV2": {"qubits": 27, "comment": "Heavy-hexagon lattice fragment"},
        "FakeParisV2": {"qubits": 27, "comment": "Heavy-hexagon lattice fragment"},
        "FakePragueV2": {"qubits": 33, "comment": "Heavy-hexagon based"},
        "FakeBrooklynV2": {"qubits": 65, "comment": "Heavy-hexagon lattice"},
        "FakeWashingtonV2": {"qubits": 127, "comment": "Large heavy-hexagon lattice"},
        "FakeBelemV2": {"qubits": 5, "comment": "Linear topology"},
    }

    print("Scanning for available fake backends in qiskit_ibm_runtime.fake_provider (based on a known list)...")
    found_count = 0
    for name, data in known_fake_backends_info.items():
        if hasattr(fp_module, name):
            q_count = data["qubits"]
            if q_count not in available_backends_by_qubits:
                available_backends_by_qubits[q_count] = []
            available_backends_by_qubits[q_count].append(f"{name} ({data.get('comment', 'Generic topology')})")
            found_count +=1

    if found_count == 0:
         print("Warning: No fake backends from the known list were found in your qiskit_ibm_runtime.fake_provider.")
         print("The list below might be empty or incomplete. Ensure qiskit_ibm_runtime is up to date.")

    print("----------------------------------------------------------------------")
    print("INFO: How to choose a backend for your problem:")
    print("1. From the list below, note the exact name of a fake backend, or a real backend name if desired.")
    print("2. Set 'param_chosen_backend_name' in the main script's USER CONFIGURATION section.")
    print("   - If a fake backend is chosen, the number of qubits for the problem will be its size.")
    print("   - If 'auto' is chosen for fake backend, selection is based on 'param_num_nodes_config'.")
    print("   - If AerSimulator is used (param_chosen_backend_name=None and not real), problem size comes from 'param_num_nodes_config'.")
    print("----------------------------------------------------------------------\n")

    if not available_backends_by_qubits:
        print("No fake backends found or listed from the known set. ")
    for q_count, names in sorted(available_backends_by_qubits.items()):
        print(f"--- {q_count} Qubits ---")
        for n_idx, full_name_comment in enumerate(sorted(names)):
            print(f"  {n_idx + 1}. {full_name_comment}")
    print("----------------------------------------------------------------------")
    return available_backends_by_qubits


def select_backend(use_real_backend_flag: bool, chosen_backend_name: str = None,
                   num_nodes_config_for_auto: int = 5, ibm_quantum_token_str: str = None):
    """
    Selects and instantiates the backend for the QAOA run based on user configuration.
    """
    print("\n--- Selecting Backend ---")
    active_backend = None

    if use_real_backend_flag:
        if not chosen_backend_name:
            print("Error: Real backend requested, but no backend name provided in 'chosen_backend_name'.")
            return None
        print(f"Attempting to use real IBM Quantum backend: {chosen_backend_name}")
        try:
            if ibm_quantum_token_str:
                QiskitRuntimeService.save_account(channel="ibm_quantum", token=ibm_quantum_token_str, overwrite=True, set_as_default=True)
            service = QiskitRuntimeService()
            active_backend = service.backend(chosen_backend_name)
            if active_backend:
                print(f"Successfully accessed real backend: {active_backend.name}")
            else:
                print(f"Could not find real backend: {chosen_backend_name}")
        except Exception as e:
            print(f"Error: Failed to initialize real backend '{chosen_backend_name}': {e}.")
            active_backend = None

    elif chosen_backend_name:
        temp_selected_fake_backend = None
        backend_to_instantiate_name = None

        if chosen_backend_name.lower() == "auto":
            print(f"Attempting to automatically select a fake backend based on configured num_nodes: {num_nodes_config_for_auto}...")
            if num_nodes_config_for_auto <= 1: backend_to_instantiate_name = "FakeArmonkV2"
            elif num_nodes_config_for_auto <= 5: backend_to_instantiate_name = "FakeManilaV2"
            elif num_nodes_config_for_auto <= 16: backend_to_instantiate_name = "FakeGuadalupeV2"
            elif num_nodes_config_for_auto <= 27: backend_to_instantiate_name = "FakeKolkataV2"
            elif num_nodes_config_for_auto <= 65: backend_to_instantiate_name = "FakeBrooklynV2"
            elif num_nodes_config_for_auto <= 127: backend_to_instantiate_name = "FakeWashingtonV2"
            else:
                print(f"Warning: num_nodes_config ({num_nodes_config_for_auto}) is too large for 'auto' selection.")
        else:
            backend_to_instantiate_name = chosen_backend_name

        if backend_to_instantiate_name:
            print(f"Attempting to instantiate fake backend: {backend_to_instantiate_name}")
            try:
                fake_backend_class = getattr(fp_module, backend_to_instantiate_name)
                temp_selected_fake_backend = fake_backend_class()
                print(f"Successfully instantiated: {temp_selected_fake_backend.name}")
            except (AttributeError, TypeError) as e:
                print(f"Error: Could not find or instantiate fake backend '{backend_to_instantiate_name}': {e}.")

        if temp_selected_fake_backend:
            active_backend = temp_selected_fake_backend

    if not active_backend:
        print("No real or specific fake backend selected/loaded. Defaulting to AerSimulator.")
        try:
            active_backend = AerSimulator()
            print(f"Using AerSimulator.")
        except Exception as e:
            print(f"FATAL Error: Could not initialize AerSimulator: {e}")
            return None

    return active_backend


def determine_graph_parameters(active_backend, use_topology_flag: bool,
                               num_nodes_config: int, edge_list_config: list):
    """
    Determines the actual number of nodes and edge list for the problem graph.
    Prioritizes V2 backend target attributes, then V1 backend attributes for topology.
    """
    print("\n--- Determining Graph Parameters ---")
    actual_num_nodes = num_nodes_config
    actual_edge_list = edge_list_config
    backend_name_for_graph = active_backend.name if hasattr(active_backend, 'name') else str(type(active_backend))

    is_aer_simulator = isinstance(active_backend, AerSimulator)
    is_runtime_fake_backend = not is_aer_simulator and \
                              hasattr(active_backend, '__class__') and \
                              hasattr(active_backend.__class__, '__module__') and \
                              active_backend.__class__.__module__.startswith('qiskit_ibm_runtime.fake_provider')
    is_real_backend = not is_aer_simulator and not is_runtime_fake_backend

    backend_qubit_count = 0
    if hasattr(active_backend, 'num_qubits') and active_backend.num_qubits is not None:
        backend_qubit_count = active_backend.num_qubits
    elif hasattr(active_backend, 'configuration'):
        config = active_backend.configuration()
        if hasattr(config, 'n_qubits'):
            backend_qubit_count = config.n_qubits

    if (is_runtime_fake_backend or is_real_backend) and backend_qubit_count > 0:
        actual_num_nodes = backend_qubit_count
        print(f"Problem size (num_nodes) will be set by backend '{backend_name_for_graph}': {actual_num_nodes} qubits.")

        if use_topology_flag:
            print(f"Deriving graph edges from {backend_name_for_graph} topology.")
            current_edges = []
            coupling_map_obj_found = None
            source_of_cmap_info = "Unknown"

            # --- Attempt 1: Standard V2 Target Access (backend.target.coupling_map) ---
            if hasattr(active_backend, 'target') and active_backend.target is not None:
                target = active_backend.target
                if hasattr(target, 'coupling_map') and target.coupling_map is not None:
                    if isinstance(target.coupling_map, CouplingMap):
                        coupling_map_obj_found = target.coupling_map
                        source_of_cmap_info = "backend.target.coupling_map (CouplingMap instance)"
                    elif hasattr(target.coupling_map, 'edge_list'):
                        coupling_map_obj_found = target.coupling_map
                        source_of_cmap_info = "backend.target.coupling_map (rustworkx graph)"

                # --- Attempt 2: V2 Target Internal Graph Access (backend.target._coupling_graph) ---
                if not coupling_map_obj_found:
                    if hasattr(target, '_coupling_graph') and target._coupling_graph is not None:
                        if hasattr(target._coupling_graph, 'edge_list'):
                            coupling_map_obj_found = target._coupling_graph
                            source_of_cmap_info = "backend.target._coupling_graph (rustworkx graph)"

            # --- Attempt 3: V1 Style Access (backend.coupling_map) ---
            if not coupling_map_obj_found:
                if hasattr(active_backend, 'coupling_map') and active_backend.coupling_map is not None:
                    v1_cmap_obj = active_backend.coupling_map
                    if isinstance(v1_cmap_obj, CouplingMap):
                        coupling_map_obj_found = v1_cmap_obj
                        source_of_cmap_info = "backend.coupling_map (V1-style, CouplingMap instance)"
                    elif isinstance(v1_cmap_obj, list):
                        try:
                            coupling_map_obj_found = CouplingMap(v1_cmap_obj)
                            source_of_cmap_info = "backend.coupling_map (V1-style, list converted to CouplingMap)"
                        except Exception as e_conv:
                            # Silently fail conversion for cleanup, or print minimal warning
                            print(f"    Warning: Failed to convert list from V1 backend.coupling_map to CouplingMap: {e_conv}")

            # --- Extract edges from the found coupling map object ---
            if coupling_map_obj_found:
                try:
                    if isinstance(coupling_map_obj_found, CouplingMap):
                        current_edges = coupling_map_obj_found.get_edges()
                    elif hasattr(coupling_map_obj_found, 'edge_list'):
                        current_edges = coupling_map_obj_found.edge_list()
                except Exception as e_extract:
                    print(f"    Warning: Error during edge extraction from coupling_map_obj_found: {e_extract}")
                    current_edges = []

            if current_edges:
                actual_edge_list = [(u, v, 1.0) for u, v in current_edges]
                print(f"Graph based on {backend_name_for_graph} topology (source: {source_of_cmap_info}): "
                      f"{actual_num_nodes} nodes, {len(actual_edge_list)} edges.")
            else:
                print(f"Warning: Could not get coupling map for {backend_name_for_graph} after trying all methods. "
                      f"Generating a complete graph for {actual_num_nodes} nodes instead.")
                if actual_num_nodes > 1:
                    actual_edge_list = [(u, v, 1.0) for u, v in combinations(range(actual_num_nodes), 2)]
                else:
                    actual_edge_list = []
        else:
            print(f"Generating a complete graph for {actual_num_nodes} nodes (matching backend '{backend_name_for_graph}' size because use_topology_flag is False).")
            if actual_num_nodes > 1:
                actual_edge_list = [(u, v, 1.0) for u, v in combinations(range(actual_num_nodes), 2)]
            else:
                actual_edge_list = []

    elif is_aer_simulator:
        actual_num_nodes = num_nodes_config if num_nodes_config > 0 else 5
        print(f"Using AerSimulator. Problem size set to {actual_num_nodes} nodes (from config or default).")
        if use_topology_flag:
            print("Info: 'use_backend_topology_as_graph' is True, but AerSimulator has no inherent topology. "
                  "Generating a complete graph.")
        if actual_num_nodes > 1:
            actual_edge_list = [(u,v, 1.0) for u,v in combinations(range(actual_num_nodes),2)]
        else:
            actual_edge_list = []
    else:
        print(f"Backend type unclear or qubit count not determined. Using user-configured graph parameters: {actual_num_nodes} nodes.")
        if actual_num_nodes > 1 and not actual_edge_list:
             print(f"Generating complete graph for {actual_num_nodes} nodes as a fallback.")
             actual_edge_list = [(u,v, 1.0) for u,v in combinations(range(actual_num_nodes),2)]

    if actual_num_nodes == 0:
        print("Error: Final number of nodes for the problem is 0. Cannot proceed.")
        return 0, []

    return actual_num_nodes, actual_edge_list
