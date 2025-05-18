import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as rustworkx_mpl_draw # Alias for clarity
from typing import List, Dict, Any # Standard type hints
import os # For creating directories and joining paths

# Note: Matplotlib backend is set in the main script (e.g., to 'Agg').
# This utility file assumes matplotlib is imported and usable.

def plot_cost_function_evolution(cost_values: List[float],
                                 base_save_dir: str = "plots",
                                 model_subdir: str = "default_run",
                                 filename: str = "cost_evolution.png"):
    """
    Plots the evolution of the cost function (expectation value) during
    the QAOA optimization process and saves the plot to a file within a
    model-specific subdirectory.

    Args:
        cost_values: A list of cost values, where each value corresponds to
                     an iteration of the classical optimizer.
        base_save_dir: The base directory where plot folders will be created (e.g., "plots").
                       This directory is created if it doesn't exist.
        model_subdir: The name of the subdirectory for this specific model run or configuration
                      (e.g., derived from backend name, qubit count, layers).
                      This directory is created under `base_save_dir` if it doesn't exist.
        filename: The name of the file (e.g., "cost_evolution.png") to save the plot to,
                  within the `base_save_dir/model_subdir/` path.
    """
    if not cost_values:
        print("No cost values provided to plot cost function evolution.")
        return

    # Construct the full directory path for saving the plot
    save_path_dir = os.path.join(base_save_dir, model_subdir)
    # Create the directory (and any parent directories in base_save_dir) if it doesn't exist.
    # `exist_ok=True` prevents an error if the directory already exists.
    os.makedirs(save_path_dir, exist_ok=True)
    full_save_path = os.path.join(save_path_dir, filename) # Full path including filename

    plt.figure(figsize=(10, 6)) # Create a new figure with a specified size for better readability
    plt.plot(cost_values, marker='o', linestyle='-', color='royalblue', label="Cost Value")
    plt.xlabel("Optimization Iteration", fontsize=12)
    plt.ylabel("Cost (Expectation Value)", fontsize=12)
    plt.title("QAOA Cost Function Evolution", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for easier value reading
    plt.legend() # Display the legend (shows the "Cost Value" label)
    plt.tight_layout() # Adjust plot parameters for a tight layout

    try:
        plt.savefig(full_save_path) # Save the figure to the specified path
        print(f"Cost evolution plot saved to: {full_save_path}")
    except Exception as e:
        print(f"Error saving cost evolution plot to {full_save_path}: {e}")

    # Close the figure to free up memory, especially important when generating many plots in a loop
    # or when using non-interactive backends like 'Agg'.
    plt.close()

def plot_results_distribution(counts_bin: Dict[str, int], top_n: int = 10,
                              base_save_dir: str = "plots",
                              model_subdir: str = "default_run",
                              filename: str = "results_distribution.png"):
    """
    Plots the probability distribution of the measurement outcomes (bitstrings)
    obtained from sampling the final QAOA circuit. It highlights the `top_n`
    most frequent outcomes and saves the plot to a file within a model-specific subdirectory.

    Args:
        counts_bin: A dictionary where keys are bitstrings (binary strings like "01101")
                    and values are their corresponding measurement counts.
        top_n: The number of most frequent bitstring outcomes to display prominently in the plot.
        base_save_dir: The base directory for saving plots.
        model_subdir: Subdirectory name specific to this model run or configuration.
        filename: Filename for the saved plot image.
    """
    if not counts_bin:
        print("No measurement counts provided to plot results distribution.")
        return

    # Construct the full directory path for saving the plot
    save_path_dir = os.path.join(base_save_dir, model_subdir)
    os.makedirs(save_path_dir, exist_ok=True) # Ensure directory exists
    full_save_path = os.path.join(save_path_dir, filename) # Full path for the image file

    total_shots = sum(counts_bin.values()) # Calculate total number of shots from counts
    # Sort the measurement outcomes by their frequency (count) in descending order.
    sorted_counts = sorted(counts_bin.items(), key=lambda item: item[1], reverse=True)

    # Determine how many of the top results to plot (either top_n or all if fewer than top_n).
    num_to_plot = min(top_n, len(sorted_counts))
    # Extract labels (bitstrings) and values (probabilities) for the bars.
    labels = [item[0] for item in sorted_counts[:num_to_plot]]
    values = [item[1] / total_shots for item in sorted_counts[:num_to_plot]] # Convert counts to probabilities

    plt.figure(figsize=(12, 7)) # Create a new figure
    bars = plt.bar(labels, values, color="mediumpurple", alpha=0.85) # Create bar plot
    plt.xlabel("Bitstring Outcomes", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title(f"Top {num_to_plot} Measurement Outcomes (Total Shots: {total_shots})", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10) # Rotate x-axis labels for better readability if many
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines for easier probability reading
    plt.tight_layout() # Adjust layout

    # Add probability values on top of each bar for precise reading.
    for bar_plot_item in bars: # Renamed variable to avoid conflict with plt.bar
        yval = bar_plot_item.get_height()
        plt.text(
            bar_plot_item.get_x() + bar_plot_item.get_width()/2.0, # X position (center of bar)
            yval + 0.005, # Y position (slightly above the bar)
            f'{yval:.3f}',    # Text: probability formatted to 3 decimal places
            ha='center',      # Horizontal alignment
            va='bottom',      # Vertical alignment
            fontsize=9
        )

    try:
        plt.savefig(full_save_path) # Save the figure
        print(f"Results distribution plot saved to: {full_save_path}")
    except Exception as e:
        print(f"Error saving results distribution plot to {full_save_path}: {e}")
    plt.close() # Close figure

def plot_max_cut_solution(graph: Any, solution_bitstring: List[int],
                          title: str = "Max-Cut Solution Visualization",
                          base_save_dir: str = "plots",
                          model_subdir: str = "default_run",
                          filename: str = "max_cut_solution.png"):
    """
    Visualizes the Max-Cut solution on the provided graph by coloring nodes
    according to their partition. Saves the plot to a file within a
    model-specific subdirectory.

    Args:
        graph: The problem graph, expected to be a `rustworkx.PyGraph` instance.
        solution_bitstring: A list of 0s and 1s representing the partition of each node.
                            The length must match the number of nodes in the graph.
        title: The title for the plot.
        base_save_dir: The base directory for saving plots.
        model_subdir: Subdirectory name specific to this model run or configuration.
        filename: Filename for the saved plot image.
    """
    # Validate graph object
    if not hasattr(graph, 'nodes') or not hasattr(graph, 'num_nodes') or graph.num_nodes() == 0:
        print("Graph has no nodes to plot or is not a valid graph object.")
        return

    # Construct the full directory path for saving the plot
    save_path_dir = os.path.join(base_save_dir, model_subdir)
    os.makedirs(save_path_dir, exist_ok=True) # Ensure directory exists
    full_save_path = os.path.join(save_path_dir, filename) # Full path for the image file

    # Determine node colors based on the solution bitstring.
    # Nodes in partition 0 get one color, nodes in partition 1 get another.
    if len(solution_bitstring) != graph.num_nodes():
        print(
            f"Error in plot_max_cut_solution: Solution bitstring length ({len(solution_bitstring)}) "
            f"does not match number of nodes ({graph.num_nodes()}). Plotting all nodes with a default color."
        )
        # Use a default color if the bitstring is mismatched, to still render the graph.
        colors = ["lightgrey"] * graph.num_nodes()
    else:
        colors = ["skyblue" if bit == 0 else "salmon" for bit in solution_bitstring]

    plt.figure(figsize=(8, 8)) # Create a new figure
    # Use a spring layout for node positioning; a seed ensures reproducible layouts.
    pos = rx.spring_layout(graph, seed=42)

    # Draw the graph using rustworkx's Matplotlib visualization capabilities.
    rustworkx_mpl_draw( # Using aliased import for clarity
        graph,
        pos=pos,             # Node positions
        node_color=colors,   # List of colors for nodes
        with_labels=True,    # Display node labels (indices)
        node_size=600,       # Size of the nodes in the plot
        font_color='black',  # Color of the node labels
        alpha=0.9,           # Node transparency
        width=1.5            # Edge width
    )
    plt.title(title, fontsize=14) # Set the plot title

    try:
        plt.savefig(full_save_path) # Save the figure
        print(f"Max-Cut solution plot saved to: {full_save_path}")
    except Exception as e:
        print(f"Error saving Max-Cut solution plot to {full_save_path}: {e}")
    plt.close() # Close figure
