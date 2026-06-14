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
import pickle
import numpy as np
from scipy.spatial.distance import cdist

# Constants
rcParams['font.family'] = 'Times New Roman'
PLOT_MODE = False

# Model's parameters
ALPHA = 5

# Full 12-feature weights (original combined metric).
# Order matches the full feature vector in extract_segment_features.
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

# Shape-based metric: only the shape features carry weight. The zero-weighted
# spectral features are still carried through the DTW for an accurate merge.
SHAPES_FEATURE_WEIGHTS = np.array([
    0.089638,  # mean_curvature
    0.155970,  # mean_diff
    0.0,       # mean_abs_diff   (spectral)
    0.0,       # sum_abs_diff    (spectral)
    0.0,       # mean_value      (spectral)
    0.0,       # amplitude       (spectral)
    0.0,       # length          (spectral)
    0.084600,  # sharp_increasing
    0.011294,  # light_increasing
    0.083740,  # sharp_decreasing
    0.022768,  # light_decreasing
    0.083542,  # constant
])

# Spectral-based metric: only the spectral features carry weight. mean_diff and
# the trend ratios remain in the vector (weight 0) so the amplitude-direction
# merge and the correlation penalty stay accurate.
SPECTRAL_FEATURE_WEIGHTS = np.array([
    0.0,       # mean_curvature    (shape)
    0.0,       # mean_diff         (aux: drives amplitude-direction merge)
    0.149875,  # mean_abs_diff
    0.081570,  # sum_abs_diff
    0.166547,  # mean_value
    0.060416,  # amplitude
    0.010039,  # length
    0.0,       # sharp_increasing  (aux: correlation)
    0.0,       # light_increasing  (aux: correlation)
    0.0,       # sharp_decreasing  (aux: correlation)
    0.0,       # light_decreasing  (aux: correlation)
    0.0,       # constant          (aux: correlation)
])

# Make sure the feature weights sum to 1
FEATURE_WEIGHTS = FEATURE_WEIGHTS / np.sum(FEATURE_WEIGHTS)
SHAPES_FEATURE_WEIGHTS = SHAPES_FEATURE_WEIGHTS / np.sum(SHAPES_FEATURE_WEIGHTS)
SPECTRAL_FEATURE_WEIGHTS = SPECTRAL_FEATURE_WEIGHTS / np.sum(SPECTRAL_FEATURE_WEIGHTS)


def _seq_distance(seq1, seq2, feature_weights):
    """Shared core for all three metrics.

    Every metric extracts the full 12-feature vector; `feature_weights` (a
    full-length vector, zero on the features that should not count) selects
    what enters the distance. The weights are applied only inside the distance
    norm, so the zero-weighted features still inform the merge (mean_diff) and
    the correlation penalty (trend ratios).
    """
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
    # Derivative amplitude: half the signed peak-to-peak slope range,
    # matching the convention in change_points_detection.
    der_f1_amp = (np.max(der_f1) - np.min(der_f1)) / 2
    der_f2_amp = (np.max(der_f2) - np.min(der_f2)) / 2
    features_seq_1 = np.array(
        [extract_segment_features(seq1[seq_1_change_points[i]:seq_1_change_points[i + 1]],
                                  len_seq1, der_f1_amp)
         for i in range(len(seq_1_change_points) - 1)])
    features_seq_2 = np.array(
        [extract_segment_features(seq2[seq_2_change_points[i]:seq_2_change_points[i + 1]], len_seq2, der_f2_amp)
         for i in range(len(seq_2_change_points) - 1)])

    # Compute pairwise L2 distances between all rows: shape (n_rows_a, n_rows_b).
    # Weights are applied here (and inside dtw_merge) rather than baked into the
    # feature arrays, so the merge / correlation steps keep the full features.
    dist_matrix = cdist(features_seq_1 * feature_weights, features_seq_2 * feature_weights, metric='euclidean')
    # Find the mean and std of the distance matrix
    mean, std = dist_matrix.mean(), dist_matrix.std()

    # Find the length of each segment
    lens_seq_1 = [seq_1_change_points[i + 1] - seq_1_change_points[i] for i in range(len(seq_1_change_points) - 1)]
    lens_seq_2 = [seq_2_change_points[i + 1] - seq_2_change_points[i] for i in range(len(seq_2_change_points) - 1)]

    # Apply gdtw to the features
    distance, path = dtw_merge(features_seq_1, features_seq_2,
                               lens1=lens_seq_1, lens2=lens_seq_2,
                               lam1=mean, lam2=std, weights=feature_weights)

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
        # Plot the mapping
        plot_two_sequences(seq1, seq2, suptitle="GDTW Mapping Annotation",
                           vlines1=seq_1_change_points, vlines2=seq_2_change_points,
                           vlines_label="Change Points", matching=path)

    return distance, path


def seq_distance(seq1, seq2, feature_weights=FEATURE_WEIGHTS):
    """Original combined metric over the full 12-feature set."""
    return _seq_distance(seq1, seq2, feature_weights)


def shapes_based_seq_distance(seq1, seq2, feature_weights=SHAPES_FEATURE_WEIGHTS):
    """Shape-based metric (only the 7 shape features carry weight)."""
    return _seq_distance(seq1, seq2, feature_weights)


def spectral_based_seq_distance(seq1, seq2, feature_weights=SPECTRAL_FEATURE_WEIGHTS):
    """Spectral-based metric (only the 5 spectral features carry weight)."""
    return _seq_distance(seq1, seq2, feature_weights)


def main():
    # Load the sequences
    seqs, _ = load_sequences(join("data", "ten_plus_cps_sequences.npz"),
                             join("data", "ten_plus_cps_sequences_meta.json"))
    seqs = [np.interp(np.linspace(0, len(s) - 1, 1000), np.arange(len(s)), s) for s in seqs]

    # Pick two sequences to compare
    seq1, seq2 = 0, 9176

    # Compute the distance and path under each of the three metrics
    full_distance, _ = seq_distance(seqs[seq1], seqs[seq2])
    shape_distance, _ = shapes_based_seq_distance(seqs[seq1], seqs[seq2])
    spectral_distance, _ = spectral_based_seq_distance(seqs[seq1], seqs[seq2])

    print(f"Full distance:\t\t{full_distance:.6f}")
    print(f"Shape distance:\t\t{shape_distance:.6f}")
    print(f"Spectral distance:\t\t{spectral_distance:.6f}")


if __name__ == "__main__":
    main()
