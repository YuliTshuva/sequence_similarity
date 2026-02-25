"""
Yuli Tshuva
Creating graph representation for the sequences.
"""

# Imports
from utils import *
from os.path import join
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams

# Constants
rcParams['font.family'] = 'Times New Roman'
DATA_DIR = "data"


def embed_sequence_as_graph(f):
    # Get feature points for the sequence
    f_fps = feature_points(f)

    # Set the segments
    segments = [f_fps[i:i + 1 + 1] for i in range(len(f_fps) - 1)]

    # Create a graph with |f_fps|-1 nodes
    n_nodes = len(f_fps) - 1
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    # Add directed edges which represent the order of the segments
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            G.add_edge(i, j, type='time')

    # Add attributes to the nodes
    for i in range(n_nodes):
        print(i)
        print(segments[i])
        segment_data = f[segments[i][0]:segments[i][1]]
        G.nodes[i]['mean'] = np.mean(segment_data)
        G.nodes[i]['std'] = np.std(segment_data)
        G.nodes[i]['min'] = np.min(segment_data)
        G.nodes[i]['max'] = np.max(segment_data)

    return G


def plot_graph_and_sequence(G, f):
    # Set a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Set a color for edge types
    edge_to_color = {'time': 'royalblue'}

    # Plot the graph
    pos = nx.spring_layout(G)
    # Plot the graph with node attributes and edge colors based on edge types
    edge_colors = [edge_to_color[G.edges[edge]['type']] for edge in G.edges()]
    # Draw nodes with attributes as a vector beside each node
    for node in G.nodes():
        node_attrs = G.nodes[node]
        node_label = f"Node {node}\nMean: {node_attrs['mean']:.2f}\nStd: {node_attrs['std']:.2f}\nMin: {node_attrs['min']:.2f}\nMax: {node_attrs['max']:.2f}"
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='hotpink', ax=axes[0])
        nx.draw_networkx_labels(G, pos, labels={node: node_label}, font_size=8, ax=axes[0])
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, ax=axes[0])

    axes[0].set_title("Graph Representation of the Sequence", fontsize=22)
    # Add a legend for edge types
    for edge_type, color in edge_to_color.items():
        axes[0].plot([], [], color=color, label=edge_type)
    axes[0].legend()

    # Get feature points for the sequence
    f_fps = feature_points(f)

    # Plot the sequence
    axes[1].plot(f, label='Signal', color='turquoise')
    axes[1].vlines(f_fps, ymin=min(f), ymax=max(f), colors='hotpink', linestyles='dashed', label='Feature Points')
    axes[1].set_title("Original Sequence", fontsize=22)
    axes[1].set_xlabel("Time", fontsize=17)
    axes[1].set_ylabel("Value", fontsize=17)
    axes[1].legend()

    # Set a suptitle
    plt.suptitle("Graph Representation and Original Sequence", fontsize=35)
    plt.tight_layout()
    plt.show()


def main():
    # Load a sample data
    file_names = [f"Atkinson_cycle_{i + 12}.csv" for i in range(9)]
    for i, file_name in enumerate(file_names):
        # Construct file path
        file_path = join(DATA_DIR, file_name)

        # Read data
        f = load_data(file_path)

        # Embed the sequence as a graph
        f_embedding_graph = embed_sequence_as_graph(f)

        # Plot the graph and the sequence
        plot_graph_and_sequence(G=f_embedding_graph, f=f)

        return


if __name__ == "__main__":
    main()
