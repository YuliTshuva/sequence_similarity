"""
Yuli Tshuva
Utility functions for similarity.
"""

# Imports
import pandas as pd
import numpy as np
import ruptures as rpt
from pyts.approximation import SymbolicAggregateApproximation
from pyts.preprocessing.discretizer import _uniform_bins
from sklearn.metrics import mean_absolute_error
from tslearn.metrics import dtw
import time

# Constants
STRATEGY = "uniform"
TIMEOUT = 10  # seconds

# Hyperparameters
AMPLITUDE_PERCENTAGE, SEGMENT_PERCENTAGE, MIN_SEGMENT_PERCENTAGE = 3, 5, 2
CONVOLVE_KERNEL_SIZE = 10
CHANGE_THRESHOLD = 3

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


def mae_distance(sax1, sax2):
    sax1_int = [ord(x) - 96 for x in sax1]
    sax2_int = [ord(x) - 96 for x in sax2]
    return mean_absolute_error(sax1_int, sax2_int)


def dtw_distance(sax1, sax2):
    sax1_int = [ord(x) - 96 for x in sax1]
    sax2_int = [ord(x) - 96 for x in sax2]
    return dtw(sax1_int, sax2_int)


def convert_subsection_to_proportion(subsection):
    result = []
    start_idx = 0
    for i in range(len(subsection)):
        if i + 1 == len(subsection) or subsection[i] != subsection[i + 1]:
            segment_length = i - start_idx + 1
            proportion = segment_length / len(subsection)
            result.append((subsection[i], proportion))
            start_idx = i + 1
    return result


def proportion_loss(subsection1, subsection2):
    cs1 = convert_subsection_to_proportion(subsection1)
    cs2 = convert_subsection_to_proportion(subsection2)

    result = 0
    for i in range(max(len(cs1), len(cs2))):
        if i < len(cs1) and i < len(cs2):
            result += abs(cs1[i][1] - cs2[i][1])
        elif i < len(cs1):
            result += cs1[i][1]
        else:
            result += cs2[i][1]

    return result / 2


def dist(abstraction1, abstraction2, alpha):
    """Get two abstractions and return Dist(t, c) = Fdist(t, c) + α · PL(t, c)"""
    return dtw_distance(abstraction1, abstraction2) + alpha * proportion_loss(abstraction1, abstraction2)


def jony_change_points(points, pen=10, model="rbf"):
    algo = rpt.Pelt(model=model).fit(points)
    result = algo.predict(pen=pen)  # pen is the HP beta
    return result


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


def increase_sample_resolution(x, n):
    """
    Given a sequence x and a target length n, increase the sample resolution of x to length n by linear interpolation.
    """
    current_length = len(x)
    if current_length >= n:
        return x

    # Create new indices
    new_indices = np.linspace(0, current_length - 1, n)
    new_x = np.interp(new_indices, np.arange(current_length), x)
    return new_x


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

    if 0 not in feature_pts:
        feature_pts.append(0)
    if len(f) - 1 not in feature_pts:
        feature_pts.append(len(f) - 1)

    return feature_pts


def original_sim_score(f1, f2):
    """
    Calculate similarity score between two sequences.
    1) Identify change points and extract sequences of consecutive segments.
    2) Normalize (by scaling) and smooth all segments.
    3) Look for partial matches: Compute similarity of each segment of Ck with each segment of T.
    """
    # Identify change points
    change_points_f1 = jony_change_points(f1, pen=CHANGE_POINTS_PEN)
    change_points_f2 = jony_change_points(f2, pen=CHANGE_POINTS_PEN)

    # Extract sequences of consecutive segments
    segments_f1 = extract_segments(f1, change_points_f1, segment_threshold=SEGMENT_THRESHOLD * len(f1) / 100)
    segments_f2 = extract_segments(f2, change_points_f2, segment_threshold=SEGMENT_THRESHOLD * len(f2) / 100)

    # Apply SAX
    sax_segments_f1 = [(sax_transform(segment[0])[0], segment[1]) for segment in segments_f1]
    sax_segments_f2 = [(sax_transform(segment[0])[0], segment[1]) for segment in segments_f2]

    # Look for partial matches
    best_similarity = np.inf
    index_f1, index_f2 = None, None
    for segment_f1, idxs1 in sax_segments_f1:
        for segment_f2, idxs2 in sax_segments_f2:
            similarity = dist(segment_f1, segment_f2, alpha=PL_ALPHA)
            if similarity < best_similarity:
                best_similarity = similarity
                index_f1, index_f2 = idxs1, idxs2

    # Try and extend the best match by checking the neighboring segments
    start = time.time()
    new_sim = best_similarity
    left_step1, right_step1 = index_f1[0], index_f1[1]
    left_step2, right_step2 = index_f2[0], index_f2[1]
    f1_percentage, f2_percentage = int(len(f1) / 100), int(len(f2) / 100)
    while new_sim <= best_similarity + EPSILON and time.time() - start < TIMEOUT:
        # Try to extend the segments in both directions
        if left_step1 > 0:
            left_step1 -= f1_percentage
        if left_step2 > 0:
            left_step2 -= f2_percentage
        if right_step1 < len(f1):
            right_step1 += f1_percentage
        if right_step2 < len(f2):
            right_step2 += f2_percentage

        # Abstract the extended segments and calculate the new similarity
        extended_segment_f1 = sax_transform(f1[left_step1:right_step1])[0]
        extended_segment_f2 = sax_transform(f2[left_step2:right_step2])[0]
        new_sim = dist(extended_segment_f1, extended_segment_f2, alpha=PL_ALPHA)

    return new_sim, (left_step1, right_step1), (left_step2, right_step2)


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

    # Filter the edges
    signs = ([signs[(CONVOLVE_KERNEL_SIZE - 1) // 2]] * ((CONVOLVE_KERNEL_SIZE - 1) // 2) +
             signs[(CONVOLVE_KERNEL_SIZE - 1) // 2:(CONVOLVE_KERNEL_SIZE - 1) // 2 * (-1)] +
             [signs[-(CONVOLVE_KERNEL_SIZE - 1) // 2 - 1]] * ((CONVOLVE_KERNEL_SIZE - 1) // 2))
    if len(signs) < len(f):
        signs = signs + [signs[-1]] * (len(f) - len(signs))
    signs = np.array(signs)

    # Extract feature points out of signs
    signs_fps = feature_points(signs)

    return signs_fps


def mark_nodes_limits(len_seq, change_points):
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
        # Check if we need to attend gap from the previous segment
        if accumulated_gap:
            accumulated_gap = False
            start = start - gap // 2
        end = change_points[i + 1]

        if i + 2 == len(change_points):
            nodes_limits.append((start, end))
            break

        next_start = change_points[i + 2]
        # Calculate the difference between one segment's end and the next segment's start
        gap = int((next_start - 1) - (end + 1) + 1)
        # If the gap is 1, we can merge the two segments into one
        if gap == 1:
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
    # Calculate the mean curvature of the segment
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
    increasing = np.sum(np.diff(segment) > threshold) / len(segment)
    decreasing = np.sum(np.diff(segment) < threshold * (-1)) / len(segment)
    constant = 1 - increasing - decreasing

    # Summarize the features in a vector
    features_vector = np.array([curvature, mean_diff, mean_abs_diff, mean_value, amplitude,
                                length, increasing, decreasing, constant])
    return features_vector
