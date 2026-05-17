"""
Yuli Tshuva
Utility functions for similarity.
"""

# Imports
import cv2
import os
import pandas as pd
import numpy as np
from pyts.approximation import SymbolicAggregateApproximation
from pyts.preprocessing.discretizer import _uniform_bins
import matplotlib.pyplot as plt
from matplotlib import rcParams
from os.path import join

# Constants
STRATEGY = "uniform"
TIMEOUT = 10  # seconds
rcParams['font.family'] = 'Times New Roman'

# Hyperparameters
AMPLITUDE_PERCENTAGE, ROBUSTNESS_PERCENTAGE = 3, 3
MIN_SEGMENT_PERCENTAGE = 5
SEGMENT_PERCENTAGE = 3
CONVOLVE_KERNEL_SIZE = 10
CHANGE_THRESHOLD = 5

# Old hyperparameters
SEGMENT_THRESHOLD = 10  # minimum length (in percentages) of a segment to be considered for similarity
SAX_N_BINS = 5  # number of bins for SAX transformation
PL_ALPHA = 0.5  # weight for combining similarity scores
CHANGE_POINTS_PEN = 10  # penalty for DTW distance
EPSILON = 0.1  # threshold for extending the best match


def sawtooth_k_cycles(n_points=1000, k=5):
    x = np.linspace(0, 1, n_points)  # normalize domain to [0,1]
    saw = (k * x) - np.floor(k * x)  # k cycles
    return saw


def sign_func(x, threshold=0):
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0


def _compute_bins(X, n_samples, n_bins):
    "from KBinsDiscretizer "
    sample_min, sample_max = np.min(X, axis=1), np.max(X, axis=1)
    bin_edges = _uniform_bins(
        sample_min, sample_max, n_samples, n_bins).T
    return bin_edges


def sax_transform(data, n_bins=5):
    # Adjust the shape for calculation
    original_shape = data.shape
    data = data.reshape(1, -1)

    # Apply SAX
    sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy=STRATEGY)
    bins = _compute_bins(X=data,
                         n_samples=len(data),
                         n_bins=n_bins)
    data_sax = sax.fit_transform(data)
    # bottom_bool = np.r_[True, data_sax[0, 1:] > data_sax[0, :-1]]

    # Reshape back to original shape
    data = data.reshape(original_shape)
    data_sax = data_sax.reshape(original_shape)

    return data_sax, bins, sax


def extract_segments(points, change_points, segment_threshold):
    segments = []
    for start_idx in range(len(change_points) - 1):
        start = change_points[start_idx]
        for i in range(1, len(change_points)):
            end = change_points[i]
            segment = points[start:end]
            if len(segment) > segment_threshold:
                segments.append((segment, (start, end)))
    return segments


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df["y"].to_numpy()


def feature_points(f):
    # Find the amplitude of f
    amp = np.max(f) - np.min(f)

    # Set a set of feature points
    feature_pts = []

    # Iterate over f to find feature points
    segment_max, segment_min = f[0], f[0]
    segment_start = 0
    segment_size = 1
    for i in range(1, len(f)):
        if segment_max - segment_min > amp * AMPLITUDE_PERCENTAGE / 100:
            # Update segment size
            segment_size -= 1
            # Check segment size
            if segment_size > len(f) * SEGMENT_PERCENTAGE / 100:
                # Add feature points
                feature_pts.append(segment_start)  # Start of segment
                feature_pts.append(i - 2)  # End of segment
            # Start a new segment
            segment_max, segment_min = max(f[i - 1], f[i]), min(f[i - 1], f[i])
            segment_size = 2
            segment_start = i - 1
        else:
            segment_max = max(segment_max, f[i])
            segment_min = min(segment_min, f[i])
            segment_size += 1

    if (segment_max - segment_min < amp * AMPLITUDE_PERCENTAGE / 100 and
            segment_size > len(f) * SEGMENT_PERCENTAGE / 100):
        feature_pts.append(segment_start)
        feature_pts.append(len(f) - 1)

    # If no feature points were found, return the whole sequence as one segment
    if len(feature_pts) == 0:
        return [0, len(f) - 1]

    # Make sure the first and last points are included as feature points
    if 0 not in feature_pts:
        feature_pts[0] = 0
    if len(f) - 1 not in feature_pts:
        feature_pts.append(feature_pts[-1] + 1)
        feature_pts.append(len(f) - 1)

    return feature_pts


def robust_partition(f, feature_pts):
    result = list(feature_pts)  # Work on a copy
    i = 0
    while True:
        n = len(result)
        if 2 * i + 2 >= n:
            break
        segment_value = f[result[2 * i]]
        next_segment_value = f[result[2 * i + 2]]
        if segment_value == next_segment_value:
            if result[2 * i + 2] - result[2 * i + 1] < len(f) * ROBUSTNESS_PERCENTAGE / 100:
                # Drop elements at 2*i+1 and 2*i+2
                result = result[:2 * i + 1] + result[2 * i + 3:]
                # Don't increment i — recheck from same position after removal
                continue
        i += 1
    return result


def change_points_detection(input_sequence):
    # Copy the input sequence to avoid modifying the original data
    f = input_sequence.copy()

    # Smooth the data
    kernel = np.array(list(range(1, CONVOLVE_KERNEL_SIZE // 2 + 1)) + list(range(CONVOLVE_KERNEL_SIZE // 2 - 1, 0, -1)))
    kernel *= kernel
    # Normalize the kernel
    kernel = kernel / kernel.sum()
    convolved_f = np.convolve(f, kernel, mode='same')
    f[(CONVOLVE_KERNEL_SIZE - 1) // 2:(CONVOLVE_KERNEL_SIZE - 1) // 2 * (-1)] = convolved_f[
        (CONVOLVE_KERNEL_SIZE - 1) // 2:(CONVOLVE_KERNEL_SIZE - 1) // 2 * (-1)]

    # Calculate first derivative of f
    der_f = np.concat([np.array([0]), np.diff(f, n=1)])

    # Calculate the derivative amplitude
    der_amp = np.max(der_f) - np.min(der_f)

    # Track change points in the derivative
    threshold = CHANGE_THRESHOLD * der_amp / 100

    # Apply sign_func over der_f
    signs = [sign_func(x, threshold) for x in der_f]
    # Level up the abstraction
    signs = [2 * sign if sign != 0 else sign_func(x, threshold / 3) for sign, x in zip(signs, der_f)]

    # Filter the edges
    signs = ([signs[(CONVOLVE_KERNEL_SIZE - 1) // 2]] * ((CONVOLVE_KERNEL_SIZE - 1) // 2) +
             signs[(CONVOLVE_KERNEL_SIZE - 1) // 2:(CONVOLVE_KERNEL_SIZE - 1) // 2 * (-1)] +
             [signs[-(CONVOLVE_KERNEL_SIZE - 1) // 2 - 1]] * ((CONVOLVE_KERNEL_SIZE - 1) // 2))
    if len(signs) < len(f):
        signs = signs + [signs[-1]] * (len(f) - len(signs))
    signs = np.array(signs)

    # Extract feature points out of signs
    signs_fps = feature_points(signs)
    # Robust the partition
    signs_fps = robust_partition(signs, signs_fps)

    return signs_fps


def mark_nodes_limits(f, len_seq, change_points):
    """
    Return a list of tuples representing the limits of the nodes in the graph, based on the change points.
    The tuples are in the form of (start_index, end_index).
    :param len_seq:
    :param change_points:
    :return:
    """
    nodes_limits = []
    accumulated_gap = False
    for i in range(0, len(change_points) - 1, 2):
        # Extract the start and end indices of the current segment
        start = change_points[i]
        if start == 760:
            pass
        # Check if we need to attend gap from the previous segment
        if accumulated_gap:
            accumulated_gap = False
            start = start - gap // 2
        end = change_points[i + 1]

        if i + 2 == len(change_points):
            nodes_limits.append((start, end))
            break

        next_start, next_end = change_points[i + 2], change_points[i + 3]
        # Calculate the difference between one segment's end and the next segment's start
        gap = int((next_start - 1) - (end + 1) + 1)
        next_segment_y_size = f[next_start:next_end + 1].max() - f[next_start:next_end + 1].min()
        # If the gap is 1, we can merge the two segments into one
        if gap <= 1:
            nodes_limits.append((start, end))
        elif gap > len_seq * MIN_SEGMENT_PERCENTAGE / 100:
            nodes_limits.append((start, end))
            nodes_limits.append((end + 1, end + gap))
        else:
            nodes_limits.append((start, end + gap // 2))
            accumulated_gap = True

    return nodes_limits


def extract_node_features(segment, len_sequence):
    """
    Given a segment, extract features for the node.
    Specifically, we are interested in:
    1) The mean curvature of the segment.
    2) The mean difference between consecutive points in the segment.
    3) The mean of the segment.
    4) Segment amplitude (max - min).
    5) The length of the segment.
    6) The percentage of the segment that is increasing, decreasing, or constant.
    :param sequence: The full sequence from which the segment is extracted.
           Normalized to [0,1]!
    :param segment: A segment of values representing a segment of the original segment.
    :return: A feature vector containing the extracted features.
    """
    curvature = np.mean(np.abs(np.diff(segment, n=2)))

    # Find the first derivative of the segment
    der_f = np.diff(segment)

    # Calculate the mean difference between consecutive points in the segment
    mean_diff = np.mean(der_f)
    mean_abs_diff = np.mean(np.abs(der_f))

    # Calculate the segment amplitude (max - min)
    amplitude = np.max(segment) - np.min(segment)

    # Calculate the mean of the segment
    mean_value = np.mean(segment)

    # Calculate the length of the segment
    length = len(segment) / len_sequence
    # Track change points in the derivative
    der_amp = np.max(der_f) - np.min(der_f)
    threshold = CHANGE_THRESHOLD * der_amp / 100

    # Calculate the percentage of the segment that is increasing, decreasing, or constant
    sharp_increasing = np.sum(np.diff(segment) > threshold) / len(segment)
    light_increasing = np.sum(np.diff(segment) > threshold / 3) / len(segment) - sharp_increasing
    sharp_decreasing = np.sum(np.diff(segment) < threshold * (-1)) / len(segment)
    light_decreasing = np.sum(np.diff(segment) < threshold / 3 * (-1)) / len(segment) - sharp_decreasing
    constant = 1 - sharp_increasing - light_increasing - sharp_decreasing - light_decreasing

    # Summarize the features in a vector
    features_vector = np.array([curvature, mean_diff, mean_abs_diff, mean_value, amplitude,
                                length, sharp_increasing, light_increasing, sharp_decreasing,
                                light_decreasing, constant])

    # Balance the features to be in the same scale
    features_vector = features_vector / np.array([0.018, 0.002, 0.018, 0.550, 0.32193304,
                                                  0.18082032, 0.2, 0.2, 0.2, 0.2, 0.2])

    return features_vector


def annotate_change_points_selection(input_sequence):
    # Set a 2x1 grid for plots
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))

    # Read data
    f = input_sequence.copy()

    # Smooth the data
    kernel = np.array(
        list(range(1, CONVOLVE_KERNEL_SIZE // 2 + 1)) + list(range(CONVOLVE_KERNEL_SIZE // 2 - 1, 0, -1)))
    kernel *= kernel
    # Normalize the kernel
    kernel = kernel / kernel.sum()
    convolved_f = np.convolve(f, kernel, mode='same')
    f[(CONVOLVE_KERNEL_SIZE - 1) // 2:(CONVOLVE_KERNEL_SIZE - 1) // 2 * (-1)] = convolved_f[
        (CONVOLVE_KERNEL_SIZE - 1) // 2:(CONVOLVE_KERNEL_SIZE - 1) // 2 * (-1)]

    # Calculate first derivative of f
    der_f = np.concat([np.array([0]), np.diff(f, n=1)])

    # Calculate the derivative amplitude
    der_amp = np.max(der_f) - np.min(der_f)

    # Track change points in the derivative
    threshold = CHANGE_THRESHOLD * der_amp / 100

    # Apply sign_func over der_f
    signs = [sign_func(x, threshold) for x in der_f]
    # Level up the abstraction
    signs = [2 * sign if sign != 0 else sign_func(x, threshold / 3) for sign, x in zip(signs, der_f)]

    # Filter the edges
    signs = ([signs[(CONVOLVE_KERNEL_SIZE - 1) // 2]] * ((CONVOLVE_KERNEL_SIZE - 1) // 2) +
             signs[(CONVOLVE_KERNEL_SIZE - 1) // 2:(CONVOLVE_KERNEL_SIZE - 1) // 2 * (-1)] +
             [signs[-(CONVOLVE_KERNEL_SIZE - 1) // 2 - 1]] * ((CONVOLVE_KERNEL_SIZE - 1) // 2))
    if len(signs) < len(f):
        signs = signs + [signs[-1]] * (len(f) - len(signs))
    signs = np.array(signs)

    # Extract feature points out of signs
    signs_fps = feature_points(signs)
    # Robust the partition
    signs_fps = robust_partition(signs, signs_fps)

    # Plot the data
    ax[0].scatter(range(len(signs)), signs, color='royalblue')
    ax[0].scatter(signs_fps, signs[signs_fps], color='red', s=70)

    # Set title
    ax[0].set_title("Sequence's Trend", fontsize=30)
    ax[0].set_ylabel("Value", fontsize=22)

    # Plot the data
    ax[1].plot(range(len(f)), f, color='royalblue')
    ax[1].scatter(signs_fps, f[signs_fps], color='hotpink', s=70)

    # Set title
    ax[1].set_title("Input Sequence", fontsize=30)

    # Set labels
    ax[1].set_xlabel("Time", fontsize=22)
    ax[1].set_ylabel("Value", fontsize=22)

    # Set suptitle
    plt.suptitle(f"Change point detection process", fontsize=42)
    # Tight layout
    plt.tight_layout()
    # Save the figure
    plt.show()


def sinkhorn(A, max_iter=1000, tol=1e-9):
    """Normalize a positive matrix so rows and columns all sum to 1."""
    A = A.astype(float)
    for _ in range(max_iter):
        # Normalize rows
        A /= A.sum(axis=1, keepdims=True)
        # Normalize columns
        A /= A.sum(axis=0, keepdims=True)

        # Check convergence
        if np.allclose(A.sum(axis=1), 1, atol=tol) and \
                np.allclose(A.sum(axis=0), 1, atol=tol):
            break
    return A


def merge_intervals(intervals: list[tuple[int, int]], target_size: int) -> list[tuple[int, int]]:
    """
    Reduce a list of intervals to target_size by agglomeratively merging
    the closest adjacent pairs until the desired length is reached.

    Merge criterion: smallest combined span (end - start) of two adjacent intervals.
    """
    if target_size >= len(intervals):
        return intervals
    if target_size <= 1:
        return [(intervals[0][0], intervals[-1][1])]

    intervals = [list(i) for i in intervals]  # work with mutable copies

    while len(intervals) > target_size:
        # Find the adjacent pair with the smallest merged span
        best_idx = None
        best_span = float('inf')

        for i in range(len(intervals) - 1):
            merged_span = intervals[i + 1][1] - intervals[i][0]
            if merged_span < best_span:
                best_span = merged_span
                best_idx = i

        # Merge the best pair into one interval
        intervals[best_idx] = [intervals[best_idx][0], intervals[best_idx + 1][1]]
        intervals.pop(best_idx + 1)

    return [tuple(i) for i in intervals]


def annotate_mapping(seq1, seq2, mapping, title="Mapping of segments in seq1 to seq2", save_path=None):
    # Normalize sequences to be in [0, 1]
    if np.max(seq1) - np.min(seq1) > 0:
        seq1 = (seq1 - np.min(seq1)) / (np.max(seq1) - np.min(seq1))
    else:
        seq1 = np.zeros_like(seq1) + 0.5
    if np.max(seq2) - np.min(seq2) > 0:
        seq2 = (seq2 - np.min(seq2)) / (np.max(seq2) - np.min(seq2))
    else:
        seq2 = np.zeros_like(seq2) + 0.5

    # Find the change points in both sequences
    seq_1_change_points = change_points_detection(seq1)
    seq_2_change_points = change_points_detection(seq2)

    # Create nodes based on change points
    nodes_1 = mark_nodes_limits(seq1, len(seq1), seq_1_change_points)
    nodes_2 = mark_nodes_limits(seq2, len(seq2), seq_2_change_points)

    # Get the separating lines between the nodes
    vlines1 = [n[0] for n in nodes_1] + [nodes_1[-1][1]]
    vlines2 = [n[0] for n in nodes_2] + [nodes_2[-1][1]]

    # Plot the probability map of each node in seq1 to seq2
    # Get a figure for the current node
    plt.subplots(len(nodes_1), 2, figsize=(25, 5 * len(nodes_1)))
    for i in range(len(nodes_1)):
        # Get the subplot of row i+1 column 1
        plt.subplot(len(nodes_1), 2, 2 * i + 1)
        plt.plot(seq1, color='royalblue')
        plt.vlines(vlines1, ymin=0, ymax=1, color='turquoise', linestyle='--')
        plt.axvspan(nodes_1[i][0], nodes_1[i][1], color='salmon')
        plt.title(f"Node {i} in Sequence 1", fontsize=20)
        if i + 1 == len(nodes_1):
            plt.xlabel("Time", fontsize=15)
        plt.ylabel("Value", fontsize=15)
        # Get the mapping probabilities for the current node
        mapping_probs = mapping[i]
        # Plot the second sequence with the mapping probabilities
        # Get the subplot of row i+1 column 2
        plt.subplot(len(nodes_1), 2, 2 * i + 1 + 1)
        plt.plot(seq2, color='dodgerblue')
        plt.vlines(vlines2, ymin=0, ymax=1, color='turquoise', linestyle='--')
        for j in range(len(nodes_2)):
            plt.axvspan(nodes_2[j][0], nodes_2[j][1], color='salmon', alpha=mapping_probs[j].item())
        plt.title(f"Mapping of Node {i} to Sequence 2", fontsize=20)
        if i + 1 == len(nodes_1):
            plt.xlabel("Time", fontsize=15)

    plt.suptitle(title, fontsize=30)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def make_a_video_from_a_set_of_images(image_folder, output_video_path, fps=1):
    # Get all image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort the images by name

    # Read the first image to get the dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video writer object
    video.release()


def increase_sequence_resolution(seq, new_length):
    """
    Increases the resolution of a sequence by linear interpolation.
    """
    old_length = len(seq)
    if old_length == new_length:
        return seq
    x_old = np.arange(old_length)
    x_new = np.linspace(0, old_length - 1, new_length)
    seq_new = np.interp(x_new, x_old, seq)
    return seq_new
