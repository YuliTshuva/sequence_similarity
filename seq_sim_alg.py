"""
Yuli Tshuva
Implementing the graph distance algorithm for sequence similarity.
1) Extract points from the sequences to create nodes in the graph using change_points_detection.
2) Create edges between the nodes based on the order of the segments and heights.
3) Extract features for each node.
4) Estimate an initial mapping using the features.
5) Refine the mapping using the optimization expression.
6) Compute the similarity score based on the final mapping.
"""

# Imports
from utils import *
from model_and_training_loop import *
import numpy as np

# Constants
rcParams['font.family'] = 'Times New Roman'
PLOT_MODE = False

# Model's parameters
ALPHA, FEATURE_WEIGHTS = 1, np.array([1] * 9)

# Make sure the feature weights sum to 1
FEATURE_WEIGHTS = FEATURE_WEIGHTS / np.sum(FEATURE_WEIGHTS)


def plot_two_sequences(seq1, seq2, suptitle="", vlines1=None, vlines2=None, vlines_label=""):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(seq1, color='royalblue', label='Sequence 1')
    axes[1].plot(seq2, color='hotpink', label='Sequence 2')
    if vlines1 is not None:
        axes[0].vlines(vlines1, ymin=min(seq1), ymax=max(seq1),
                       colors='turquoise', linestyles='dashed', label=vlines_label)
    if vlines2 is not None:
        axes[1].vlines(vlines2, ymin=min(seq2), ymax=max(seq2),
                       colors='turquoise', linestyles='dashed', label=vlines_label)
    plt.suptitle(suptitle, fontsize=30)
    for ax_i in axes:
        ax_i.set_xlabel("Timestep", fontsize=21)
    axes[0].set_ylabel("Value", fontsize=21)
    axes[0].legend(fontsize=15)
    axes[1].legend(fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_loss(losses, scheduler_steps, title=""):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color='darksalmon')
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss Value", fontsize=15)
    plt.xticks([0] + scheduler_steps)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def seq_distance(seq1, seq2, alpha=ALPHA, feature_weights=FEATURE_WEIGHTS):
    """
    Given any two numeric sequences, compute the distance between them as:
    dist = min_sigma [ dist_features + alpha * dist_structure ]
     - dist_features: distance between the features of the nodes (segments) in the two sequences
     - dist_structure: distance between the structure of the two sequences (edges between nodes)
        - sigma: the soft mapping matrix between the nodes of the two sequences (rows sum to 1)

    :return: distance, sigma
    """
    # Make sure feature weights sum to 1
    feature_weights = feature_weights / np.sum(feature_weights)

    # Normalize sequences to be in [0, 1]
    if np.max(seq1) - np.min(seq1) > 0:
        seq1 = (seq1 - np.min(seq1)) / (np.max(seq1) - np.min(seq1))
    else:
        seq1 = np.zeros_like(seq1) + 0.5
    if np.max(seq2) - np.min(seq2) > 0:
        seq2 = (seq2 - np.min(seq2)) / (np.max(seq2) - np.min(seq2))
    else:
        seq2 = np.zeros_like(seq2) + 0.5

    # Plot the sequences
    if PLOT_MODE:
        plot_two_sequences(seq1, seq2, suptitle="Initial Sequences")

    seq_1_change_points = change_points_detection(seq1)
    seq_2_change_points = change_points_detection(seq2)

    if PLOT_MODE:
        plot_two_sequences(seq1, seq2, suptitle="Sequences with Change Points",
                           vlines1=seq_1_change_points, vlines2=seq_2_change_points, vlines_label="Change Points")
        annotate_change_points_selection(seq1)
        annotate_change_points_selection(seq2)

    # Create nodes based on change points
    nodes_1 = mark_nodes_limits(len(seq1), seq_1_change_points)
    nodes_2 = mark_nodes_limits(len(seq2), seq_2_change_points)

    if PLOT_MODE:
        plot_two_sequences(seq1, seq2, suptitle="Sequences with Nodes Limits",
                           vlines1=[n[0] for n in nodes_1] + [nodes_1[-1][1]],
                           vlines2=[n[0] for n in nodes_2] + [nodes_2[-1][1]],
                           vlines_label="Node Limits")

    # Extract features for each node
    len_seq1, len_seq2 = len(seq1), len(seq2)
    features_seq_1 = np.array([extract_node_features(seq1[node[0]:node[1] + 1], len_seq1) for node in nodes_1])
    features_seq_2 = np.array([extract_node_features(seq2[node[0]:node[1] + 1], len_seq2) for node in nodes_2])
    # Apply feature weights
    features_seq_1 *= feature_weights
    features_seq_2 *= feature_weights

    # Estimate initial mapping using features similarity (cosine similarity)
    normalized_features_seq_1 = features_seq_1 / np.linalg.norm(features_seq_1, axis=1, keepdims=True)
    normalized_features_seq_2 = features_seq_2 / np.linalg.norm(features_seq_2, axis=1, keepdims=True)
    initial_mapping = normalized_features_seq_1 @ normalized_features_seq_2.T

    # Make sure the initial mapping is non-negative (cosine similarity can be negative)
    initial_mapping = np.maximum(initial_mapping, 0)

    # Normalize rows
    row_sums = initial_mapping.sum(axis=1, keepdims=True)
    initial_mapping /= row_sums

    # Set a model instance
    model = SequenceSimilarity(initial_mapping, features_seq_1, features_seq_2, alpha=alpha)

    # Send to training loop
    if PLOT_MODE:
        model, best_match_loss, loss_history, scheduler_steps = train_model(model, save_loss=True)
        # Plot the loss history
        plot_loss(loss_history, scheduler_steps, title="Training Loss History")
    else:
        model, best_match_loss = train_model(model)

    sigma = model.get_constrained_sigma()

    # Return the best match loss as the distance between the two sequences and the mapping matrix
    return best_match_loss, sigma


def main():
    # Read two sequences
    seq1 = load_data("data/Atkinson_cycle_44.csv")
    seq2 = load_data("data/Atkinson_cycle_2.csv")

    # Compute distance
    distance, sigma = seq_distance(seq1, seq2)

    # Print the distance
    print("Distance between the two sequences:", distance)
    print("The best mapping matrix (sigma):\n", sigma)

    # Annotate the mapping on the sequences
    annotate_mapping(seq1, seq2, sigma)


if __name__ == "__main__":
    main()
