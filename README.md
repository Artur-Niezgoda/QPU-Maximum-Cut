# QAOA for Max-Cut on Qiskit

This project implements the Quantum Approximate Optimization Algorithm (QAOA) to solve the Max-Cut problem using Qiskit. It provides a flexible framework for defining graphs, selecting various quantum backends (simulators, fake backends, and real QPUs), configuring QAOA parameters, and visualizing results.

## Overview

The Max-Cut problem is a classic NP-hard problem in graph theory, which involves partitioning the vertices of a graph into two sets such that the number of edges connecting vertices in different sets is maximized. QAOA is a hybrid quantum-classical algorithm well-suited for tackling such combinatorial optimization problems on near-term quantum devices.

This implementation allows users to:
- Define custom graphs or use topologies derived from quantum backends.
- Execute QAOA on local simulators, fake backends mimicking real hardware, or actual IBM Quantum hardware.
- Visualize the problem, optimization process, and solution.

## Features

- **Modular Design:** Code is organized into separate modules for graph utilities (`graph_utils.py`), QAOA logic (`qaoa_utils.py`), execution setup (`execution_setup.py`), and plotting (`plotting_utils.py`).
- **Flexible Graph Definition:**
    - Create graphs with a specified number of nodes and custom edge lists (with weights).
    - Option to derive the graph topology directly from the chosen backend's coupling map.
    - Defaults to a complete graph if no specific topology is provided when not using backend topology.
- **Versatile Backend Selection (`execution_setup.py`):**
    - **Local Simulation:** `AerSimulator` from `qiskit-aer` for noise-free simulations.
    - **Fake Backends:** Utilizes `qiskit_ibm_runtime.fake_provider` to simulate various IBM Quantum hardware characteristics.
    - **Real Quantum Hardware:**
        - Connects to IBM Quantum QPUs via `QiskitRuntimeService`.
        - Supports selection of a specific QPU by its name.
        - Supports automatic selection of the "least busy" available QPU, with an option to specify minimum qubit requirements.
- **QAOA Implementation (`qaoa_utils.py`):**
    - Constructs the Max-Cut cost Hamiltonian (`SparsePauliOp`) from the problem graph.
    - Builds the parameterized `QAOAAnsatz` circuit.
    - Decomposes the ansatz and `PauliEvolutionGate`s for backend compatibility.
    - Optimizes QAOA parameters (gammas and betas) using `scipy.optimize.minimize` (COBYLA method) with `EstimatorV2`.
- **Transpilation and Execution:**
    - Transpiles the logical QAOA circuit for the target backend, considering an initial layout to map logical qubits to physical qubits.
    - Handles scenarios where the logical problem size (N qubits) is smaller than the backend capacity (M qubits) by:
        - Transpiling the N-qubit circuit to an M-qubit circuit that acts non-trivially on the mapped N physical qubits.
        - Manually expanding the N-qubit Hamiltonian to an M-qubit Hamiltonian (acting with identities on ancillary qubits) for compatibility with `EstimatorV2`.
    - Uses `SamplerV2` to obtain measurement outcomes from the optimized circuit.
- **Error Mitigation Options:**
    - Allows enabling dynamical decoupling and Pauli twirling for `EstimatorV2` and `SamplerV2` when running on real hardware or compatible simulators.
- **Result Visualization (`plotting_utils.py`):**
    - Plots the initial problem graph.
    - Visualizes the evolution of the cost function during optimization.
    - Displays the probability distribution of measurement outcomes.
    - Plots the Max-Cut solution (graph partitioning) on the problem graph.
- **Configuration (`main.py`):**
    - Centralized user configuration section in `main.py` to easily set parameters like:
        - Number of nodes and edge list for custom graphs.
        - Choice of backend (None for AerSimulator, fake backend name, specific QPU name, or "least_busy").
        - Flag to use real backend.
        - Flag to use backend's topology as the problem graph.
        - Number of QAOA layers (p).
        - Number of shots for Estimator and Sampler.
        - IBM Quantum token (if not already saved).
        - Qiskit Runtime Service channel (e.g., `ibm_quantum`, `ibm_cloud`).
        - Minimum qubits for "least_busy" QPU selection.

## Prerequisites

- Python 3.9+
- Qiskit and related packages.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    The project relies on several Python packages listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include `qiskit`, `qiskit-ibm-runtime`, `qiskit-aer`, `rustworkx`, `numpy`, `scipy`, and `matplotlib`.

4.  **IBM Quantum Account (for real QPU access):**
    - If you plan to run on real IBM Quantum hardware, you need an IBM Quantum account.
    - Save your API token to your local Qiskit configuration, or provide it via `param_ibm_token` in `main.py`.
      ```python
      # Example for saving account (run once in a Python script/notebook)
      # from qiskit_ibm_runtime import QiskitRuntimeService
      # QiskitRuntimeService.save_account(channel="ibm_cloud", token="YOUR_IBM_QUANTUM_TOKEN", overwrite=True, set_as_default=True)
      ```

## Configuration

All primary run parameters are configurable within the `if __name__ == "__main__":` block in `main.py`.

Key parameters to adjust:

-   `param_num_nodes_config`: Number of nodes for your graph if not using backend topology.
-   `param_edge_list_config`: List of tuples `(u, v, weight)` defining graph edges. If empty and not using backend topology, a complete graph is generated for `param_num_nodes_config`.
-   `param_use_real_backend`: Set to `True` to run on a real QPU or `False` for simulators/fake backends.
-   `param_chosen_backend_name`:
    -   `None`: Defaults to `AerSimulator`.
    -   `"FakeBackendName"` (e.g., `"FakeBurlingtonV2"`): Uses the specified fake backend.
    -   `"least_busy"`: If `param_use_real_backend` is `True`, selects the least busy QPU (use with `param_min_qubits_for_least_busy`).
    -   `"qpu_name"` (e.g., `"ibm_kyoto"`): If `param_use_real_backend` is `True`, uses the specified QPU.
-   `param_use_backend_topology`:
    -   `True`: The problem graph (nodes and edges) will be derived from the selected backend's coupling map. `param_num_nodes_config` and `param_edge_list_config` are ignored.
    -   `False`: The graph defined by `param_num_nodes_config` and `param_edge_list_config` is used.
-   `param_qaoa_layers`: Number of QAOA layers (p).
-   `param_num_shots_sampling`: Shots for the final sampling with `SamplerV2`.
-   `param_estimator_shots`: Default shots for `EstimatorV2` during optimization.
-   `param_ibm_token`: Your IBM Quantum API token (if not already saved/default).
-   `param_qiskit_runtime_channel`: Channel for `QiskitRuntimeService` (e.g., `"ibm_quantum"`, `"ibm_cloud"`). If `None`, uses the saved default.
-   `param_min_qubits_for_least_busy`: Minimum number of qubits when `param_chosen_backend_name` is `"least_busy"`.

## Usage

1.  Configure the parameters in the `USER CONFIGURATION` section at the bottom of `main.py`.
2.  Run the main script from your terminal:
    ```bash
    python main.py
    ```
3.  Output, including logs and paths to saved plots, will be printed to the console. Plots are saved in the `plots/` directory, organized into subdirectories based on the backend and problem configuration.

## File Structure

-   `main.py`: The main executable script. Contains user configuration and orchestrates the QAOA workflow.
-   `execution_setup.py`: Handles backend selection and determines graph parameters based on configuration.
-   `qaoa_utils.py`: Contains core QAOA logic, including ansatz setup, Hamiltonian construction, parameter optimization, and circuit sampling.
-   `graph_utils.py`: Utilities for creating graph instances (`rustworkx.PyGraph`) and evaluating Max-Cut solutions.
-   `plotting_utils.py`: Functions for generating and saving various plots (cost evolution, result distribution, solution graph).
-   `functions.py`: Contains auxiliary functions, including an alternative `cost_func_estimator` (note: the primary one used by the optimizer is in `qaoa_utils.py`).
-   `requirements.txt`: Lists project dependencies.
-   `plots/`: Directory where output plots are saved.

## Examples

**Example 1: Run a custom 5-node graph on `AerSimulator`**
In `main.py`:
```python
param_num_nodes_config = 5
param_edge_list_config = [
    (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
    (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)
]
param_use_real_backend = False
param_chosen_backend_name = None # Defaults to AerSimulator
param_use_backend_topology = False

Example 2: Run using FakeBurlingtonV2's topology
In main.py:

param_use_real_backend = False
param_chosen_backend_name = "FakeBurlingtonV2"
param_use_backend_topology = True # Graph will be Burlington's 5-qubit T-shape

Example 3: Run a 4-node complete graph on the least busy 7-qubit (or more) QPU
In main.py:

param_num_nodes_config = 4
param_edge_list_config = [] # Will generate a 4-node complete graph
param_use_real_backend = True
param_chosen_backend_name = "least_busy"
param_use_backend_topology = False # Use our 4-node graph on the QPU
param_qiskit_runtime_channel = "ibm_cloud" # Or "ibm_quantum" or None
param_min_qubits_for_least_busy = 7

Contributing
Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
