"""
Yuli Tshuva
Implementing the graph distance algorithm for sequence similarity.
I need to:
1) Extract points from the sequences to create nodes in the graph using change_points_detection.
2) Create edges between the nodes based on the order of the segments and heights.
3) Extract features for each node.
4) Estimate an initial mapping using the features.
5) Refine the mapping using the optimization expression.
6) Compute the similarity score based on the final mapping.
"""

# Imports
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utils import *
from model_and_training_loop import *

# Constants
rcParams['font.family'] = 'Times New Roman'


def plot_two_sequences(seq1, seq2):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].plot(seq1, color='royalblue')
    axes[1].plot(seq2, color='hotpink')
    plt.suptitle("Two Sequences for Similarity Comparison", fontsize=20)
    for ax_i in axes:
        ax_i.set_xlabel("Timestep", fontsize=15)
        ax_i.set_ylabel("Value", fontsize=15)
    plt.show()


def seq_distance(seq1, seq2):
    # Normalize sequences to be in [0, 1]
    seq1 = (seq1 - np.min(seq1)) / (np.max(seq1) - np.min(seq1))
    seq2 = (seq2 - np.min(seq2)) / (np.max(seq2) - np.min(seq2))

    # Find change points for both sequences
    seq_1_change_points = change_points_detection(seq1)
    seq_2_change_points = change_points_detection(seq2)

    # Create nodes based on change points
    nodes_1 = mark_nodes_limits(len(seq1), seq_1_change_points)
    nodes_2 = mark_nodes_limits(len(seq2), seq_2_change_points)

    # Extract features for each node
    len_seq1, len_seq2 = len(seq1), len(seq2)
    features_seq_1 = np.array([extract_node_features(seq1[node[0]:node[1] + 1], len_seq1) for node in nodes_1])
    features_seq_2 = np.array([extract_node_features(seq2[node[0]:node[1] + 1], len_seq2) for node in nodes_2])

    # Estimate initial mapping using features similarity (cosine similarity)
    features_seq_1 = features_seq_1 / np.linalg.norm(features_seq_1, axis=1, keepdims=True)
    features_seq_2 = features_seq_2 / np.linalg.norm(features_seq_2, axis=1, keepdims=True)
    initial_mapping = features_seq_1 @ features_seq_2.T

    # Make sure the initial mapping is non-negative (cosine similarity can be negative)
    initial_mapping = np.maximum(initial_mapping, 0)

    # Normalize the initial mapping such that each row sums to 1 and each column sums to a constant value.
    # Set n_iters and eps
    n_iters, eps = 10, 1e-9
    for _ in range(n_iters):
        # Normalize columns
        col_sums = initial_mapping.sum(axis=0, keepdims=True)
        initial_mapping /= (col_sums + eps)

        # Normalize rows
        row_sums = initial_mapping.sum(axis=1, keepdims=True)
        initial_mapping /= (row_sums + eps)

    # Check sum of rows and columns
    row_sum, col_sum = np.mean(initial_mapping.sum(axis=1)), np.mean(initial_mapping.sum(axis=0))
    print("Row sums (should be close to 1):", row_sum)
    print("Column sums (should be close to 1):", col_sum)

    # Set a model instance
    model = SequenceSimilarity(initial_mapping, features_seq_1, features_seq_2)

    # Send to training loop
    model, best_match_loss = train_model(model)

    # Return the best match loss as the distance between the two sequences and the mapping matrix
    return best_match_loss, model.sigma.detach().cpu().numpy()


def main():
    # Read two sequences
    seq1 = load_data("data/Atkinson_cycle_44.csv")
    seq2 = load_data("data/Atkinson_cycle_2.csv")

    # Plot the sequences
    plot_two_sequences(seq1, seq2)

    # Compute distance
    distance, sigma = seq_distance(seq1, seq2)

    # Print the distance
    print("Distance between the two sequences:", distance)
    print("The best mapping matrix (sigma):\n", sigma)


if __name__ == "__main__":
    main()
