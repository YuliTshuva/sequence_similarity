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
import numpy as np
from mine_data import load_sequences
import os
from os.path import join
import pickle
import numpy as np

# Constants
rcParams['font.family'] = 'Times New Roman'
PLOT_MODE = False

# Model's parameters
ALPHA = 5

FEATURE_WEIGHTS = np.array([
    0.1,  # mean_curvature
    0.1,  # mean_diff
    0.1,  # mean_abs_diff
    0.1,  # sum_abs_diff
    0.1,  # mean_value
    0.1,  # amplitude
    0.1,  # length
    0.1,  # sharp_increasing
    0.1,  # light_increasing
    0.1,  # sharp_decreasing
    0.1,  # light_decreasing
    0.1,  # constant
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

    # Apply gdtw to the features
    distance, path = dtw_merge(features_seq_1, features_seq_2)

    if PLOT_MODE:
        plot_two_sequences(seq1, seq2, suptitle="Sequences with mapped segments",
                           vlines1=seq_1_change_points, vlines2=seq_2_change_points,
                           vlines_label="Change Points", matching=path)

    return distance, path


def debug_change_points(seq):
    # Normalize sequences to be in [0, 1]
    seq1 = normalize_sequence(seq)

    # Detect change points
    annotate_change_points_selection(seq1)


def test_seq_distance():
    print("Testing distance function...")

    # Load sequences
    paths = os.listdir(join("data", "tests"))
    for path in paths:
        with open(join("data", "tests", path), "rb") as f:
            lst = pickle.load(f)

        # Test distance function on pairs of sequences
        anchor = lst[0]
        # Make the sequence of length 1000
        anchor = np.interp(np.linspace(0, len(anchor) - 1, 1000), np.arange(len(anchor)), anchor)
        dists = []
        for i in range(1, len(lst)):
            dists.append(
                seq_distance(anchor, np.interp(np.linspace(0, len(lst[i]) - 1, 1000), np.arange(len(lst[i])), lst[i]))[
                    0])

        # Check that distances are in expected order (closest to farthest)
        dists_sorted = np.argsort(dists)
        if not (np.array_equal(dists_sorted, np.arange(len(dists)))):
            print(f"Test failed for {path}: distances are not in expected order.")
            print("Oder of distances:", dists_sorted)
        else:
            print(f"Test passed for {path}.")

        print("\n", "*" * 50, "\n")


def main():
    # seqs, _ = load_sequences()
    # seqs = [np.interp(np.linspace(0, len(s) - 1, 1000), np.arange(len(s)), s) for s in seqs]
    # seq1, seq2 = 57, 56
    # debug_change_points(seqs[seq1])
    #
    # distance, path = seq_distance(seqs[seq1], seqs[seq2])
    #
    # print(f"Distance between sequences {seq1} and {seq2}: {distance:.4f}")

    test_seq_distance()


if __name__ == "__main__":
    main()
