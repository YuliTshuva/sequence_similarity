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
from mine_data import load_sequences
import os
from os.path import join
import numpy as np
from scipy.spatial.distance import cdist

# Constants
rcParams['font.family'] = 'Times New Roman'
PLOT_MODE = False

# Check if the OS is linux
if os.name == 'posix':
    PLOT_MODE = False

# Model's parameters
ALPHA = 5

FEATURE_WEIGHTS = np.array([
    0.089638,  # mean_curvature
    0.155970,  # mean_diff
    0.149875,  # mean_abs_diff
    0.081570,  # sum_abs_diff
    0.166547,  # mean_value
    0.060416,  # amplitude
    0.010039,  # length
    0.084600,  # sharp_increasing
    0.011294,  # light_increasing
    0.083740,  # sharp_decreasing
    0.022768,  # light_decreasing
    0.083542,  # constant
])

# Make sure the feature weights sum to 1
FEATURE_WEIGHTS = FEATURE_WEIGHTS / np.sum(FEATURE_WEIGHTS)


def seq_distance(seq1, seq2, feature_weights=FEATURE_WEIGHTS):
    # Make sure feature weights sum to 1
    feature_weights = feature_weights / np.sum(feature_weights)

    # Normalize sequences to be in [0, 1]
    seq1, seq2 = normalize_sequence(seq1), normalize_sequence(seq2)

    # Detect change points
    seq_1_change_points = change_points_detection(seq1)
    seq_2_change_points = change_points_detection(seq2)

    # Extract features for each node
    len_seq1, len_seq2 = len(seq1), len(seq2)
    der_f1, der_f2 = np.diff(seq1), np.diff(seq2)
    der_f1_amp, der_f2_amp = np.max(np.abs(der_f1)) - np.min(np.abs(der_f1)), np.max(np.abs(der_f2)) - np.min(
        np.abs(der_f2))
    features_seq_1 = np.array(
        [extract_segment_features(seq1[seq_1_change_points[i]:seq_1_change_points[i + 1]], len_seq1, der_f1_amp) for i
         in range(len(seq_1_change_points) - 1)])
    features_seq_2 = np.array(
        [extract_segment_features(seq2[seq_2_change_points[i]:seq_2_change_points[i + 1]], len_seq2, der_f2_amp) for i
         in range(len(seq_2_change_points) - 1)])

    # Apply feature weights
    features_seq_1 *= feature_weights
    features_seq_2 *= feature_weights

    # Compute pairwise L2 distances between all rows: shape (n_rows_a, n_rows_b)
    dist_matrix = cdist(features_seq_1, features_seq_2, metric='euclidean')
    # Find the mean and std of the distance matrix
    mean, std = dist_matrix.mean(), dist_matrix.std()

    # Find the length of each segment
    lens_seq_1 = [seq_1_change_points[i + 1] - seq_1_change_points[i] for i in range(len(seq_1_change_points) - 1)]
    lens_seq_2 = [seq_2_change_points[i + 1] - seq_2_change_points[i] for i in range(len(seq_2_change_points) - 1)]

    # Apply gdtw to the features
    distance, path = dtw_merge(features_seq_1, features_seq_2,
                               lens1=lens_seq_1, lens2=lens_seq_2,
                               lam1=mean, lam2=std)

    # Update the path by the mode
    new_path = []
    for (x_indices, y_indices, mode) in path:
        if mode == "merge":
            new_path.append((x_indices, y_indices))
        elif mode == "independent":
            for x in x_indices:
                for y in y_indices:
                    new_path.append(([x], [y]))
        else:
            raise Exception(f'Unknown mode: {mode}')
    path = new_path

    if PLOT_MODE:
        plot_two_sequences(seq1, seq2, suptitle=f"GDTW Mapping Annotation | Distance: {distance:.3f}",
                           vlines1=seq_1_change_points, vlines2=seq_2_change_points,
                           vlines_label="Change Points", matching=path)

    return distance, path


def debug_change_points(seq):
    # Normalize sequences to be in [0, 1]
    seq1 = normalize_sequence(seq)

    # Detect change points
    annotate_change_points_selection(seq1)


def main():
    # Load the sequences
    seqs, _ = load_sequences(join("data", "six_cps_sequences.npz"),
                             join("data", "six_cps_sequences_meta.json"))

    # Calc eps and delta
    # epsilon = np.min([np.std(s) for s in seqs])
    # delta = np.mean([len(s) for s in seqs]) * 0.2

    # Pick two sequences to compare
    # seq1, seq2 = 0, 1871
    # plot_dtw_alignment(seqs[seq1], seqs[seq2], suptitle="DTW(0, 1871) Alignment")
    # seq1, seq2 = 0, 4147
    # plot_lcss_alignment(seqs[seq1], seqs[seq2], eps=epsilon, delta=int(delta), suptitle="LCSS(0, 4147) Alignment")

    # Compute the distance and path between two sequences
    seq1, seq2 = 3050, 885
    seq_distance(seqs[seq1], seqs[seq2])

if __name__ == "__main__":
    main()
