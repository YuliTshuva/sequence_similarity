"""
Yuli Tshuva
Utility functions for similarity.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Constants
STRATEGY = "uniform"
TIMEOUT = 10  # seconds
rcParams['font.family'] = 'Times New Roman'

# Hyperparameters
MIN_SEGMENT_PERCENTAGE = 4
SEGMENT_PERCENTAGE = 4
MIN_DISTANCE_BETWEEN_FEATURE_POINTS = 2
CHANGE_THRESHOLD = 10
SLIDING_WINDOW_SIZE = 3
SKIP_PERCENTAGE = 3


def sign_func(x, threshold=0):
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0


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
        if segment_max - segment_min > 0:
            # Update segment size
            segment_size -= 1
            # Check segment size
            if segment_size > len(f) * SEGMENT_PERCENTAGE / 100:
                # Add feature points
                feature_pts.append(segment_start)  # Start of segment
                feature_pts.append(i - 2)  # End of segment
                segment_size = 2
                segment_start = i - 1
            # Start a new segment
            segment_max, segment_min = max(f[i - 1], f[i]), min(f[i - 1], f[i])
        else:
            segment_max = max(segment_max, f[i])
            segment_min = min(segment_min, f[i])
            segment_size += 1

    if (segment_max - segment_min < 0.01 and
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
        feature_pts[-1] = len(f) - 1

    return feature_pts


def robust_partition(f, feature_pts):
    result = list(feature_pts)  # Work on a copy
    i = 0
    while True:
        n = len(result)
        if 2 * i + 2 >= n:
            break
        segment_value = np.round(np.mean(f[result[2 * i]:result[2 * i + 1] + 1]))
        next_segment_value = np.round(np.mean(f[result[2 * i + 2]:result[2 * i + 3] + 1]))
        if segment_value == next_segment_value and result[2 * i + 2] - result[2 * i + 1] < len(
                f) * SKIP_PERCENTAGE / 100:
            result = result[:2 * i + 1] + result[2 * i + 3:]
            continue
        i += 1

    return result


def merge_nearby_points(fps, f, min_distance=None):
    if min_distance is None:
        min_distance = max(5, MIN_DISTANCE_BETWEEN_FEATURE_POINTS * len(f) // 100)

    result = list(fps)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result) - 1:
            if result[i + 1] - result[i] < min_distance:
                # Keep the point further from local mean
                lo = max(0, result[i] - min_distance)
                hi = min(len(f), result[i + 1] + min_distance)
                local_mean = np.mean(f[lo:hi])
                dev_i = abs(f[result[i]] - local_mean)
                dev_j = abs(f[result[i + 1]] - local_mean)
                # Drop the less extreme point
                drop = i if dev_i < dev_j else i + 1
                result.pop(drop)
                changed = True
            else:
                i += 1

    # Merge neighboring points that are in the same height
    new_results = []
    amp_f = np.max(f) - np.min(f)
    skip = False
    for i in range(len(result) - 1):
        if skip:
            skip = False
            continue
        if abs(f[result[i]] - f[result[i + 1]]) < amp_f * 0.01:
            new_results.append((result[i] + result[i + 1]) // 2)
            skip = True
        else:
            new_results.append(result[i])
    result = new_results

    return result


def change_points_detection(f, return_signs=False):
    # Calculate first derivative of f
    der_f = np.diff(f, n=1)
    der_f = np.concat([der_f[0], der_f])

    # Calculate the derivative amplitude
    der_amp = np.max(der_f) - np.min(der_f)

    # Track change points in the derivative
    threshold = CHANGE_THRESHOLD * der_amp / 100

    # Apply sign_func over der_f
    signs = [sign_func(x, threshold) for x in der_f]

    if len(signs) < len(f):
        signs = signs + [signs[-1]] * (len(f) - len(signs))
    signs = np.array(signs)

    # Apply most decision to the raw signs
    window = SLIDING_WINDOW_SIZE * len(f) // 100
    signs = [return_most_common_sign(signs[i:i + window]) for i in range(len(signs) - window)]
    signs = ([signs[0]] * (window // 2) + signs + [signs[-1]] * (window // 2))
    signs = signs + [signs[-1]] * (len(f) - len(signs)) if len(signs) < len(f) else signs[:len(f)]
    signs = np.array(signs)

    # Extract feature points out of signs - in pairs [start1, end1, start2, end2, ...]
    signs_fps = feature_points(signs)

    # Merge adjacent segments with the same trend
    signs_fps = robust_partition(signs, signs_fps)

    # Mark node limits
    signs_fps = mark_nodes_limits(f, len(f), signs_fps)

    # signs_fps = merge_nearby_points(signs_fps, f)
    if len(f) - 1 not in signs_fps:
        if len(f) - 1 - signs_fps[-1] >= len(f) * SEGMENT_PERCENTAGE / 100:
            signs_fps.append(len(f) - 1)
        else:
            signs_fps[-1] = len(f) - 1
    if 0 not in signs_fps:
        if signs_fps[0] >= len(f) * SEGMENT_PERCENTAGE / 100:
            signs_fps.insert(0, 0)
        else:
            signs_fps[0] = 0

    if return_signs:
        return signs_fps, signs
    return signs_fps


def return_most_common_sign(segment):
    if len(segment) == 0:
        return 0
    counts = np.bincount(segment + 1)  # Shift to make -1 -> 0, 0 -> 1, 1 -> 2
    most_common = np.argmax(counts) - 1  # Shift back
    return most_common


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
        # Check if we need to attend gap from the previous segment
        if accumulated_gap:
            accumulated_gap = False
            start = start - gap // 2
        end = change_points[i + 1]

        if i + 2 == len(change_points):
            nodes_limits += [start, end]
            break

        next_start, next_end = change_points[i + 2], change_points[i + 3]
        # Calculate the difference between one segment's end and the next segment's start
        gap = int((next_start - 1) - (end + 1) + 1)
        # If the gap is 1, we can merge the two segments into one
        if gap <= 1:
            nodes_limits += [start, end]
        elif gap > len_seq * MIN_SEGMENT_PERCENTAGE / 100:
            nodes_limits += [start, end]
            nodes_limits += [end + 1, end + gap]
        else:
            nodes_limits += [start, end + gap // 2]
            accumulated_gap = True

    return nodes_limits


def extract_node_features(segment, len_sequence):
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

    # Robust the partition
    signs_fps, signs = change_points_detection(input_sequence, return_signs=True)

    # Plot the data
    ax[0].scatter(range(len(signs)), signs, color='royalblue')
    ax[0].scatter(signs_fps, signs[signs_fps], color='red', s=70)

    # Set title
    ax[0].set_title("Sequence's Trend", fontsize=30)
    ax[0].set_ylabel("Value", fontsize=22)

    # Plot the data
    ax[1].plot(range(len(input_sequence)), input_sequence, color='royalblue')
    ax[1].scatter(signs_fps, input_sequence[signs_fps], color='hotpink', s=70)

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
