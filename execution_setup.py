import inspect # For discovering fake backends
from itertools import combinations # For generating complete graphs if needed

# Qiskit imports
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator, Aer
import qiskit_ibm_runtime.fake_provider as fp_module
from qiskit.transpiler import CouplingMap

try:
    from qiskit.providers.fake_provider import FakeBackendV2
except ImportError:
    print("Warning: Could not import FakeBackendV2 from qiskit.providers.fake_provider. Type checks against it may be limited.")
    class FakeBackendV2: pass


def get_available_fake_backends_info():
    """
    Lists known fake backends from qiskit_ibm_runtime.fake_provider and their qubit counts.
    """
    available_backends_by_qubits = {}
    # (known_fake_backends_info dictionary remains the same as your current version)
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
    print("Scanning for available fake backends...")
    # ... (rest of get_available_fake_backends_info remains the same) ...
    for name, data in known_fake_backends_info.items():
        if hasattr(fp_module, name):
            q_count = data["qubits"]
            if q_count not in available_backends_by_qubits:
                available_backends_by_qubits[q_count] = []
            available_backends_by_qubits[q_count].append(f"{name} ({data.get('comment', 'Generic topology')})")
    print("----------------------------------------------------------------------")
    print("INFO: How to choose a backend for your problem:")
    print("1. From the list below, note the exact name of a fake backend, or a real backend name if desired.")
    print("2. Set 'param_chosen_backend_name' in the main script's USER CONFIGURATION section.")
    print("   - To use a specific real backend, provide its name (e.g., 'ibm_kyoto').")
    print("   - To use the least busy real backend, set to 'least_busy' and optionally set 'param_min_qubits_for_least_busy'.")
    print("   - If a fake backend is chosen, the number of qubits for the problem will be its size.")
    print("   - If 'auto' is chosen for fake backend, selection is based on 'param_num_nodes_config'.")
    print("   - If AerSimulator is used (param_chosen_backend_name=None and not real), problem size comes from 'param_num_nodes_config'.")
    print("----------------------------------------------------------------------\n")
    for q_count, names in sorted(available_backends_by_qubits.items()):
        print(f"--- {q_count} Qubits ---")
        for n_idx, full_name_comment in enumerate(sorted(names)):
            print(f"  {n_idx + 1}. {full_name_comment}")
    print("----------------------------------------------------------------------")
    return available_backends_by_qubits


def select_backend(use_real_backend_flag: bool,
                   chosen_backend_name: str = None,
                   num_nodes_config_for_auto_fake: int = 5, # Renamed for clarity
                   ibm_quantum_token_str: str = None,
                   qiskit_runtime_channel: str = None, # New parameter for channel
                   min_qubits_for_least_busy: int = None # New parameter for least_busy
                   ):
    """
    Selects and instantiates the backend.
    Handles real QPUs (by name or least_busy), fake backends, and AerSimulator.
    """
    print("\n--- Selecting Backend ---")
    active_backend = None

    if use_real_backend_flag:
        print(f"Attempting to use real IBM Quantum backend.")
        try:
            # Initialize QiskitRuntimeService with channel and token if provided
            if ibm_quantum_token_str:
                # If token is provided, channel defaults to 'ibm_cloud' if not specified,
                # but user's notebook used 'ibm_quantum'. We'll prioritize qiskit_runtime_channel.
                effective_channel = qiskit_runtime_channel if qiskit_runtime_channel else "ibm_cloud"
                print(f"Initializing QiskitRuntimeService with channel='{effective_channel}' and provided token.")
                service = QiskitRuntimeService(channel=effective_channel, token=ibm_quantum_token_str)
                # Optionally save if this is the primary way to set credentials
                # QiskitRuntimeService.save_account(channel=effective_channel, token=ibm_quantum_token_str, overwrite=True, set_as_default=True)
            else:
                # If no token, use specified channel or rely on saved default account.
                print(f"Initializing QiskitRuntimeService (channel='{qiskit_runtime_channel if qiskit_runtime_channel else 'default'}').")
                service = QiskitRuntimeService(channel=qiskit_runtime_channel if qiskit_runtime_channel else None)

            if chosen_backend_name == "least_busy":
                print(f"Searching for the least busy backend with min_num_qubits={min_qubits_for_least_busy if min_qubits_for_least_busy is not None else 'any'}...")
                active_backend = service.least_busy(min_num_qubits=min_qubits_for_least_busy)
            elif chosen_backend_name:
                print(f"Attempting to access specific real backend: {chosen_backend_name}")
                active_backend = service.backend(chosen_backend_name)
            else:
                print("Error: Real backend requested, but no specific name or 'least_busy' provided in 'chosen_backend_name'.")
                return None

            if active_backend:
                print(f"Successfully accessed real backend: {active_backend.name} (Qubits: {active_backend.num_qubits if hasattr(active_backend, 'num_qubits') else 'N/A'})")
            else:
                # This case might not be reached if service.backend() or least_busy() raises an error
                print(f"Could not find or access the specified real backend configuration ('{chosen_backend_name}').")
        except Exception as e:
            print(f"Error: Failed to initialize or access real backend '{chosen_backend_name}': {e}.")
            active_backend = None

    elif chosen_backend_name and chosen_backend_name.lower() != "least_busy":
        # Handle fake backends (chosen_backend_name is not None and not "least_busy")
        temp_selected_fake_backend = None
        backend_to_instantiate_name = None

        if chosen_backend_name.lower() == "auto":
            print(f"Attempting to automatically select a fake backend based on configured num_nodes: {num_nodes_config_for_auto_fake}...")
            # (Auto-selection logic for fake backends remains the same)
            if num_nodes_config_for_auto_fake <= 1: backend_to_instantiate_name = "FakeArmonkV2"
            elif num_nodes_config_for_auto_fake <= 5: backend_to_instantiate_name = "FakeManilaV2"
            # ... (other auto selections)
            else:
                print(f"Warning: num_nodes_config ({num_nodes_config_for_auto_fake}) is too large for 'auto' fake backend selection.")
        else:
            backend_to_instantiate_name = chosen_backend_name

        if backend_to_instantiate_name:
            print(f"Attempting to instantiate fake backend: {backend_to_instantiate_name}")
            try:
                fake_backend_class = getattr(fp_module, backend_to_instantiate_name)
                temp_selected_fake_backend = fake_backend_class()
                print(f"Successfully instantiated fake backend: {temp_selected_fake_backend.name}")
            except (AttributeError, TypeError) as e:
                print(f"Error: Could not find or instantiate fake backend '{backend_to_instantiate_name}': {e}.")

        if temp_selected_fake_backend:
            active_backend = temp_selected_fake_backend

    # Default to AerSimulator if no other backend was successfully selected
    if not active_backend:
        print(f"No real or specific fake backend selected/loaded. Defaulting to AerSimulator.")
        # num_nodes_config_for_auto_fake is the problem size defined by the user (param_num_nodes_config in main.py)
        # This is used to size AerSimulator correctly if it's the default for a custom graph.
        num_qubits_for_aer = num_nodes_config_for_auto_fake if num_nodes_config_for_auto_fake > 0 else 1

        print(f"INFO: AerSimulator will be configured for {num_qubits_for_aer} qubits.")
        try:
            active_backend = Aer.get_backend('aer_simulator')
            active_backend.set_options(num_qubits=num_qubits_for_aer)

            final_aer_qubits = active_backend.num_qubits if hasattr(active_backend, 'num_qubits') else -1
            print(f"Using AerSimulator, configured with num_qubits={final_aer_qubits}.")
            if final_aer_qubits != num_qubits_for_aer:
                 print(f"Warning: AerSimulator final qubit count ({final_aer_qubits}) does not match requested ({num_qubits_for_aer}).")

        except Exception as e:
            print(f"FATAL Error: Could not initialize or configure AerSimulator for {num_qubits_for_aer} qubits: {e}")
            return None
    return active_backend

# determine_graph_parameters function remains the same as your latest working version
def determine_graph_parameters(active_backend, use_topology_flag: bool,
                               num_nodes_config: int, edge_list_config: list):
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

    backend_reported_qubits = 0
    if hasattr(active_backend, 'num_qubits') and active_backend.num_qubits is not None:
        backend_reported_qubits = active_backend.num_qubits
    elif hasattr(active_backend, 'configuration'):
        conf = active_backend.configuration()
        if hasattr(conf, 'n_qubits'):
            backend_reported_qubits = conf.n_qubits

    if (is_runtime_fake_backend or is_real_backend) and backend_reported_qubits > 0:
        if use_topology_flag:
            actual_num_nodes = backend_reported_qubits
            print(f"Problem size (num_nodes) will be set by backend '{backend_name_for_graph}': {actual_num_nodes} qubits (using backend topology).")

            print(f"Deriving graph edges from {backend_name_for_graph} topology.")
            current_edges = []
            coupling_map_obj_found = None
            source_of_cmap_info = "Unknown"

            if hasattr(active_backend, 'target') and active_backend.target is not None:
                target = active_backend.target
                if hasattr(target, 'coupling_map') and target.coupling_map is not None:
                    if isinstance(target.coupling_map, CouplingMap):
                        coupling_map_obj_found = target.coupling_map
                        source_of_cmap_info = "backend.target.coupling_map (CouplingMap instance)"
                    elif hasattr(target.coupling_map, 'edge_list'):
                        coupling_map_obj_found = target.coupling_map
                        source_of_cmap_info = "backend.target.coupling_map (rustworkx graph)"

                if not coupling_map_obj_found and hasattr(target, '_coupling_graph') and target._coupling_graph is not None:
                    if hasattr(target._coupling_graph, 'edge_list'):
                        coupling_map_obj_found = target._coupling_graph
                        source_of_cmap_info = "backend.target._coupling_graph (rustworkx graph)"

            if not coupling_map_obj_found and hasattr(active_backend, 'coupling_map') and active_backend.coupling_map is not None:
                v1_cmap_obj = active_backend.coupling_map
                if isinstance(v1_cmap_obj, CouplingMap):
                    coupling_map_obj_found = v1_cmap_obj
                    source_of_cmap_info = "backend.coupling_map (V1-style, CouplingMap instance)"
                elif isinstance(v1_cmap_obj, list):
                    try:
                        coupling_map_obj_found = CouplingMap(v1_cmap_obj)
                        source_of_cmap_info = "backend.coupling_map (V1-style, list converted to CouplingMap)"
                    except Exception as e_conv:
                        print(f"    Warning: Failed to convert list from V1 backend.coupling_map to CouplingMap: {e_conv}")

            if coupling_map_obj_found:
                try:
                    if isinstance(coupling_map_obj_found, CouplingMap):
                        current_edges = coupling_map_obj_found.get_edges()
                    elif hasattr(coupling_map_obj_found, 'edge_list'):
                        current_edges = coupling_map_obj_found.edge_list()
                except Exception as e_extract:
                    print(f"    Warning: Error during edge extraction: {e_extract}")
                    current_edges = []

            if current_edges:
                actual_edge_list = [(u, v, 1.0) for u, v in current_edges]
                print(f"Graph based on {backend_name_for_graph} topology (source: {source_of_cmap_info}): "
                      f"{actual_num_nodes} nodes, {len(actual_edge_list)} edges.")
            else:
                print(f"Warning: Could not get coupling map for {backend_name_for_graph}. Generating complete graph for {actual_num_nodes} nodes.")
                if actual_num_nodes > 1: actual_edge_list = [(u, v, 1.0) for u, v in combinations(range(actual_num_nodes), 2)]
                else: actual_edge_list = []
        else:
            print(f"Using backend '{backend_name_for_graph}' ({backend_reported_qubits} qubits), but graph defined by user configuration: {actual_num_nodes} nodes (use_backend_topology_as_graph=False).")
            if actual_num_nodes > backend_reported_qubits:
                 print(f"Warning: Configured graph num_nodes ({actual_num_nodes}) > backend's reported qubits ({backend_reported_qubits}). This might lead to issues during transpilation if the backend cannot support this size.")
            if actual_num_nodes > 1 and not actual_edge_list:
                print(f"User-defined edge list is empty. Generating a complete graph for {actual_num_nodes} nodes.")
                actual_edge_list = [(u, v, 1.0) for u, v in combinations(range(actual_num_nodes), 2)]


    elif is_aer_simulator:
        print(f"Using AerSimulator. Problem size set to {actual_num_nodes} nodes (from config).")
        print(f"   (AerSimulator instance reports {backend_reported_qubits} qubits).")

        if use_topology_flag:
            print("Info: 'use_backend_topology_as_graph' is True, but AerSimulator has no inherent topology beyond all-to-all for its configured size. "
                  "Using user-defined graph or generating a complete graph based on its configured size.")

        if actual_num_nodes > 1 and not actual_edge_list:
            print(f"User-defined edge list is empty for AerSimulator. Generating a complete graph for {actual_num_nodes} nodes.")
            actual_edge_list = [(u,v, 1.0) for u,v in combinations(range(actual_num_nodes),2)]
        else:
            print(f"Using user-defined graph for AerSimulator with {actual_num_nodes} nodes.")


    else:
        print(f"Backend type unclear. Using user-configured graph: {actual_num_nodes} nodes.")
        if actual_num_nodes > 1 and not actual_edge_list:
             actual_edge_list = [(u,v, 1.0) for u,v in combinations(range(actual_num_nodes),2)]

    if actual_num_nodes == 0:
        print("Error: Final number of nodes for the problem is 0. Cannot proceed.")
        return 0, []

    return actual_num_nodes, actual_edge_list
