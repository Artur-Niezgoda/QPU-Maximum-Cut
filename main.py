import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import matplotlib.pyplot as plt
from functions import build_max_cut_paulis
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz


# Graph initialization

n = 5 # number of nodes
weight = 1.0

graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, n, 1))
edge_list = [(0, 1, weight), (0, 2, weight), (0, 4, weight), (1, 2, weight),
              (2, 3, weight), (3, 4, weight)]
graph.add_edges_from(edge_list)
draw_graph(graph, node_size=600, with_labels=True)
#plt.clf()
#plt.show()

# STEP 1: Map classical inputs to a quantum problem

max_cut_paulis = build_max_cut_paulis(graph)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis) # create cost Hamiltonian
print("Cost Function Hamiltonian:", cost_hamiltonian)

circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
circuit.measure_all()

#circuit.draw('mpl')


