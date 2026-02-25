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
AMPLITUDE_PERCENTAGE, SEGMENT_PERCENTAGE = 3, 10
SEGMENT_THRESHOLD = 10  # minimum length (in percentages) of a segment to be considered for similarity
SAX_N_BINS = 5  # number of bins for SAX transformation
PL_ALPHA = 0.5  # weight for combining similarity scores
CHANGE_POINTS_PEN = 10  # penalty for DTW distance
EPSILON = 0.1  # threshold for extending the best match
EXTREMUM_LENGTH = 5  # minimum length of a segment to be considered an extremum


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


def change_points(points, pen=10, model="rbf"):
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
    change_points_f1 = change_points(f1, pen=CHANGE_POINTS_PEN)
    change_points_f2 = change_points(f2, pen=CHANGE_POINTS_PEN)

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
