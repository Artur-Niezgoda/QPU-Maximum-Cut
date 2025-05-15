import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import matplotlib.pyplot as plt


# Graph initialization

n = 5 # number of nodes
line_len = 1.0

graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, n, 1))
edge_list = [(0, 1, line_len), (0, 2, line_len), (0, 4, line_len), (1, 2, line_len),
              (2, 3, line_len), (3, 4, line_len)]
graph.add_edges_from(edge_list)
draw_graph(graph, node_size=600, with_labels=True)
# plt.show()

# STEP 1: Map classical inputs to a quantum problem

