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
from mine_data import load_sequences

# Constants
rcParams['font.family'] = 'Times New Roman'
PLOT_MODE = True

# Model's parameters
ALPHA = 5

FEATURE_WEIGHTS = np.array([
    0.127774,  # curvature
    0.059324,  # mean_diff
    0.194151,  # mean_abs_diff
    0.153703,  # mean_value
    0.069695,  # amplitude
    0.174445,  # length
    0.141257,  # sharp_increasing
    0.001659,  # light_increasing
    0.001452,  # sharp_decreasing
    0.012653,  # light_decreasing
    0.063887,  # constant
])

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
    if vlines1 and vlines2:
        axes[0].set_xticks(vlines1)
        axes[1].set_xticks(vlines2)
    axes[0].legend(fontsize=15)
    axes[1].legend(fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_loss(loss1, loss2, loss3, scheduler_steps, title=""):
    plt.figure(figsize=(8, 5))
    x = range(len(loss1))
    plt.plot(x, loss1, color='turquoise', label="Total Distance")
    plt.plot(x, loss2, color='hotpink', label="Feature Distance")
    plt.plot(x, loss3, color='royalblue', label="Structural Distance")
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss Value", fontsize=15)
    plt.xticks([0] + scheduler_steps)
    # plt.yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3])
    # plt.yscale('log')
    plt.title(title, fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.show()


def directional_seq_distance(seq1, seq2, alpha=ALPHA, feature_weights=FEATURE_WEIGHTS):
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
    if PLOT_MODE and False:
        plot_two_sequences(seq1, seq2, suptitle="Initial Sequences")

    seq_1_change_points = change_points_detection(seq1)
    seq_2_change_points = change_points_detection(seq2)

    if PLOT_MODE and False:
        plot_two_sequences(seq1, seq2, suptitle="Sequences with Change Points",
                           vlines1=seq_1_change_points, vlines2=seq_2_change_points, vlines_label="Change Points")

    # Create nodes based on change points
    nodes_1 = [(seq_1_change_points[i], seq_1_change_points[i + 1] - 1) for i in range(len(seq_1_change_points) - 1)]
    nodes_2 = [(seq_2_change_points[i], seq_2_change_points[i + 1] - 1) for i in range(len(seq_2_change_points) - 1)]
    # Correct the last node to include the end of the sequence
    nodes_1[-1] = (nodes_1[-1][0], len(seq1) - 1)
    nodes_2[-1] = (nodes_2[-1][0], len(seq2) - 1)

    if PLOT_MODE:
        # plot_two_sequences(seq1, seq2, suptitle="Sequences with Nodes Limits",
        #                    vlines1=[n[0] for n in nodes_1] + [nodes_1[-1][1]],
        #                    vlines2=[n[0] for n in nodes_2] + [nodes_2[-1][1]],
        #                    vlines_label="Node Limits")
        annotate_change_points_selection(seq1)
        annotate_change_points_selection(seq2)

    return

    # ## Merge intervals of the sequence with the less nodes
    # if len(nodes_1) < len(nodes_2):
    #     nodes_2 = merge_intervals(nodes_2, len(nodes_1))
    # elif len(nodes_2) < len(nodes_1):
    #     nodes_1 = merge_intervals(nodes_1, len(nodes_2))
    #
    # if PLOT_MODE:
    #     plot_two_sequences(seq1, seq2, suptitle="Sequences with Merged segments",
    #                        vlines1=[n[0] for n in nodes_1] + [nodes_1[-1][1]],
    #                        vlines2=[n[0] for n in nodes_2] + [nodes_2[-1][1]],
    #                        vlines_label="Node Limits")

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
    model = SequenceSimilarity(initial_mapping, normalized_features_seq_1,
                               normalized_features_seq_2, alpha=alpha)

    # Send to training loop
    model, best_match_loss = train_model(model)

    sigma = model.get_constrained_sigma()
    distance = model.compute_features_distance(sigma).item()

    return distance, sigma

def seq_distance(seq1, seq2, alpha=ALPHA, feature_weights=FEATURE_WEIGHTS):
    """
    Wrapper for directional_seq_distance to compute a symmetric distance between two sequences.
    """
    dist_1_to_2, sigma_1_to_2 = directional_seq_distance(seq1, seq2, alpha=alpha, feature_weights=feature_weights)
    dist_2_to_1, sigma_2_to_1 = directional_seq_distance(seq2, seq1, alpha=alpha, feature_weights=feature_weights)

    # Average the distances to get a symmetric distance
    distance = (dist_1_to_2 + dist_2_to_1) / 2

    # Average the mapping matrices (after transposing the second one to match dimensions)
    sigma = (sigma_1_to_2 + sigma_2_to_1.T) / 2

    return distance, sigma


def main():
    seqs, _ = load_sequences()
    import random
    seq1, seq2 = seqs[random.randint(0, 1000)], seqs[random.randint(0, 1000)]

    # Make the sequences f length 1000 by interpolation
    new_length = 1000
    seq1 = np.interp(np.linspace(0, len(seq1) - 1, new_length), np.arange(len(seq1)), seq1)
    seq2 = np.interp(np.linspace(0, len(seq2) - 1, new_length), np.arange(len(seq2)), seq2)

    # Compute distance
    distance, sigma = directional_seq_distance(seq1, seq2, alpha=ALPHA)

    # Print the distance
    print("Distance between the two sequences:", distance)

    # Plot sigma as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(sigma.detach().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Mapping Strength')
    plt.xlabel('Nodes in Sequence 2', fontsize=15)
    plt.ylabel('Nodes in Sequence 1', fontsize=15)
    plt.title(f'Mapping Matrix (Sigma)', fontsize=20)
    plt.xticks(range(sigma.shape[1]))
    plt.yticks(range(sigma.shape[0]))
    plt.tight_layout()
    plt.show()

    annotate_mapping(seq1, seq2, sigma.detach().numpy())


if __name__ == "__main__":
    main()
