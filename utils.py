"""
Yuli Tshuva
Utility functions for similarity.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import string

# Constants
STRATEGY = "uniform"
TIMEOUT = 10  # seconds
rcParams['font.family'] = 'Times New Roman'

# Hyperparameters
SEGMENT_PERCENTAGE = 2
MIN_DISTANCE_BETWEEN_FEATURE_POINTS = 1
CHANGE_THRESHOLD = 10
EPSILON = 1e-6
INF = float('inf')


def sign_func(x, threshold=0):
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0


def normalize_sequence(seq):
    if np.max(seq) - np.min(seq) > 0:
        seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
    else:
        seq = np.zeros_like(seq) + 0.5
    return seq


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
        if segment_max - segment_min > EPSILON:
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

    if (segment_max - segment_min < EPSILON and
            segment_size > len(f) * SEGMENT_PERCENTAGE / 100):
        feature_pts.append(segment_start)
        feature_pts.append(len(f) - 1)

    # If no feature points were found, return the whole sequence as one segment
    if len(feature_pts) == 0:
        return [0, len(f) - 1]

    return feature_pts


def find_local_extrema(der_f, f, min_prominence):
    # Smooth the derivative with a wide window to suppress noise
    smooth_win = max(11, len(f) // 30)
    kernel = np.ones(smooth_win) / smooth_win
    smooth_der = np.convolve(der_f, kernel, mode='same')

    # Find zero-crossings of the smoothed derivative
    extrema = []
    for i in range(1, len(smooth_der)):
        if smooth_der[i - 1] > 0 and smooth_der[i] <= 0:
            extrema.append(i)  # peak
        elif smooth_der[i - 1] < 0 and smooth_der[i] >= 0:
            extrema.append(i)  # valley

    snapped = []
    half_snap = max(5, smooth_win // 2)
    for idx in extrema:
        lo = max(0, idx - half_snap)
        hi = min(len(f), idx + half_snap + 1)
        window = f[lo:hi]
        mean = np.mean(window)
        # Pick whichever is further from the mean — max or min
        if np.max(window) - mean >= mean - np.min(window):
            snapped.append(lo + np.argmax(window))  # peak
        else:
            snapped.append(lo + np.argmin(window))  # valley
    extrema = snapped

    # Filter by prominence using a tight local window
    # A smaller window ensures we measure the local stand-out of the extremum
    # rather than the global amplitude, catching mid-sequence peaks/valleys
    half_win = max(5, len(f) // 50)
    filtered = []
    for idx in extrema:
        lo = max(0, idx - half_win)
        hi = min(len(f), idx + half_win)
        local_range = np.max(f[lo:hi]) - np.min(f[lo:hi])
        if local_range >= min_prominence:
            filtered.append(idx)

    return filtered


def drop_false_extrema(signs_fps, signs):
    if len(signs_fps) < 4:
        return signs_fps
    result = list(signs_fps)
    i = 0
    while i < len(result) - 2:
        # Dominant sign of segment i and segment i+1
        seg_a = signs[result[i]:result[i + 1] + 1]
        seg_b = signs[result[i + 1]:result[i + 2] + 1] if i + 2 < len(result) else []
        if len(seg_a) == 0 or len(seg_b) == 0:
            i += 1
            continue
        dir_a = np.sign(np.round(np.mean(seg_a)))
        dir_b = np.sign(np.round(np.mean(seg_b)))
        if dir_a == dir_b and dir_a != 0:
            # Same direction on both sides — drop the boundary point between them
            result = result[:i + 1] + result[i + 2:]
        else:
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


def change_points_detection(input_sequence, return_signs=False):
    f = input_sequence

    # Calculate first derivative of f
    der_f = np.diff(f, n=1)
    der_f = np.concat([[der_f[0]], der_f])

    # Calculate the derivative amplitude
    der_amp = (np.max(der_f) - np.min(der_f)) / 2

    # Track change points in the derivative
    threshold = CHANGE_THRESHOLD * der_amp / 100

    # Apply sign_func over der_f
    signs = np.array([sign_func(x, threshold) for x in der_f])
    signs = signs + np.array([sign_func(x, threshold / 3) for x in der_f])

    # Extract feature points out of signs - in pairs [start1, end1, start2, end2, ...]
    signs_fps = feature_points(signs)

    # Extract local extrema
    amp = np.max(f) - np.min(f)
    min_prominence = amp * 0.05
    extrema = find_local_extrema(der_f, f, min_prominence)

    # Merge extrema into existing feature points and re-sort
    if extrema:
        combined = []
        i, j = 0, 0
        while i < len(signs_fps) and j < len(extrema):
            if signs_fps[i] < extrema[j]:
                combined.append(signs_fps[i])
                i += 1
            elif signs_fps[i] == extrema[j]:
                combined.append(signs_fps[i])
                i += 1
                j += 1
            else:
                combined.append(extrema[j] - 1)
                combined.append(extrema[j])
                j += 1
        # Add any remaining points
        combined += signs_fps[i:]

    signs_fps = drop_false_extrema(signs_fps, signs)
    signs_fps = merge_nearby_points(signs_fps, f)

    if len(input_sequence) - 1 not in signs_fps:
        if len(input_sequence) - 1 - signs_fps[-1] < len(input_sequence) * SEGMENT_PERCENTAGE / 100:
            signs_fps[-1] = len(input_sequence) - 1
        else:
            signs_fps.append(len(input_sequence) - 1)
    if 0 not in signs_fps:
        if signs_fps[0] < len(input_sequence) * SEGMENT_PERCENTAGE / 100:
            signs_fps[0] = 0
        else:
            signs_fps.insert(0, 0)

    if return_signs:
        return signs_fps, signs
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
        # If the gap is 1, we can merge the two segments into one
        if gap <= 1:
            nodes_limits.append((start, end))
        elif gap > len_seq * SEGMENT_PERCENTAGE / 100:
            nodes_limits.append((start, end))
            nodes_limits.append((end + 1, end + gap))
        else:
            nodes_limits.append((start, end + gap // 2))
            accumulated_gap = True

    return nodes_limits


def extract_segment_features(segment, len_sequence, der_amp):
    # Calculate the curvature of the segment (mean of absolute second derivative)
    mean_curvature = np.mean(np.abs(np.diff(segment, n=2)))

    # Find the first derivative of the segment
    der_f = np.diff(segment)

    # Calculate the mean difference between consecutive points in the segment
    mean_diff = np.mean(der_f)
    mean_abs_diff = np.mean(np.abs(der_f))
    sum_abs_diff = np.sum(np.abs(der_f))

    # Calculate the segment amplitude (max - min)
    amplitude = np.max(segment) - np.min(segment)

    # Calculate the mean of the segment
    mean_value = np.mean(segment)

    # Calculate the length of the segment
    length = len(segment) / len_sequence

    # Track change points in the derivative
    threshold = CHANGE_THRESHOLD * der_amp / 100

    # Calculate the percentage of the segment that is increasing, decreasing, or constant
    sharp_increasing = np.sum(np.diff(segment) > threshold) / len(segment)
    light_increasing = np.sum(np.diff(segment) > threshold / 3) / len(segment) - sharp_increasing
    sharp_decreasing = np.sum(np.diff(segment) < threshold * (-1)) / len(segment)
    light_decreasing = np.sum(np.diff(segment) < threshold / 3 * (-1)) / len(segment) - sharp_decreasing
    constant = 1 - sharp_increasing - light_increasing - sharp_decreasing - light_decreasing

    # Summarize the features in a vector
    features_vector = np.array([mean_curvature, mean_diff, mean_abs_diff, sum_abs_diff,
                                mean_value, amplitude,
                                length, sharp_increasing, light_increasing, sharp_decreasing,
                                light_decreasing, constant])

    # Pre-computed means and standard deviations
    means = np.array([1.29917091e-04, 6.88773468e-05, 2.15043790e-03, 2.47959340e-01,
                      4.91384676e-01, 2.41590926e-01, 1.07280928e-01, 3.82233768e-01,
                      8.75325723e-02, 3.36648956e-01, 8.52701912e-02, 1.08314512e-01])
    stds = np.array([8.64548709e-05, 2.54086189e-03, 1.45989651e-03, 2.42832550e-01,
                     2.83370148e-01, 2.43137777e-01, 7.82924368e-02, 3.93206966e-01,
                     1.25110193e-01, 3.85479883e-01, 1.28169222e-01, 1.30012737e-01])

    # Normalize features by subtracting the mean and dividing by the standard deviation
    features_vector = (features_vector - means) / stds

    return features_vector


def merge_segments_features(features_1, features_2):
    # Averaging by default
    merged_features = (features_1 + features_2) / 2
    # sum_abs_diff should be summed
    merged_features[3] = features_1[3] + features_2[3]
    # Amplitude
    if features_1[1] * features_2[1] > 0:
        merged_features[5] = features_1[5] + features_2[5]
    if features_1[1] * features_2[1] < 0:
        merged_features[5] = abs(features_1[5] - features_2[5])
    # length should be summed
    merged_features[6] = features_1[6] + features_2[6]
    return merged_features


def merge_sequence(vecs):
    """Merge a list of vectors iteratively using merge_fn."""
    result = vecs[0]
    for v in vecs[1:]:
        result = merge_segments_features(result, v)
    return result


def features_correlation(features):
    # Edge case: if there's only one feature vector, return 1.0
    if features.shape[0] == 1:
        return 1.0

    # Filter irrelevant features
    relevant_features = [7, 8, 9, 10, 11]
    # Filter features to only the relevant ones
    filtered = features[:, relevant_features]
    # Calculate the minimal pairwise correlation between the features
    corr_matrix = np.corrcoef(filtered)  # shape: (n_rows, n_rows)

    # Mask the diagonal (self-correlations = 1.0) to ignore them
    np.fill_diagonal(corr_matrix, np.nan)

    # Return the minimum correlation across all pairs
    return np.nanmin(corr_matrix)


def dtw_merge(X, Y, lens1, lens2, lam1, lam2):
    n, m = len(X), len(Y)
    max_merge = max(n, m)

    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    back = {}  # (i, j) -> (di, dj, mode) where mode is 'merge' or 'independent'

    # Extract total length for both sequences
    len_x, len_y = np.sum(lens1), np.sum(lens2)
    # Normalize lens1 and lens2
    lens1 = lens1 / len_x
    lens2 = lens2 / len_y

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            best = INF
            best_step = None

            for di in range(1, min(i, max_merge) + 1):
                for dj in range(1, min(j, max_merge) + 1):
                    prev = D[i - di, j - dj]
                    if prev == INF:
                        continue

                    # Calculate the correlation between the features in the current segments
                    seg_x, seg_y = X[i - di:i], Y[j - dj:j]
                    corr_x, corr_y = features_correlation(seg_x), features_correlation(seg_y)
                    penalty = lam1 * ((1 - 2 * lam2 * corr_x) * (di - 1) + (1 - 2 * lam2 * corr_y) * (dj - 1))

                    # --- Mode 1: merge ---
                    merged_x = merge_sequence(X[i - di:i])
                    merged_y = merge_sequence(Y[j - dj:j])
                    dist_merge = np.linalg.norm(merged_x - merged_y, 2)

                    # Calculate the length of the segments
                    len_seg_x, len_seg_y = np.sum(lens1[i - di:i]), np.sum(lens2[j - dj:j])

                    # Evaluate the cost of merging these segments
                    # cost_merge = prev + (dist_merge + penalty)
                    cost_merge = prev + ((len_seg_x + len_seg_y) / 2) * (dist_merge + penalty)

                    if cost_merge < best:
                        best = cost_merge
                        best_step = (di, dj, 'merge')

                    # --- Mode 2: independent (only meaningful if asymmetric) ---
                    if di == 1 and dj > 1:
                        dist_indep = sum(np.linalg.norm(X[i - 1] - Y[j - dj + k], 2) *
                                         (lens1[i - 1] + lens2[j - dj + k]) / 2 for k in range(dj))
                        cost_indep = prev + dist_indep
                        if cost_indep < best:
                            best = cost_indep
                            best_step = (di, dj, 'independent')

                    elif dj == 1 and di > 1:
                        dist_indep = sum(np.linalg.norm(X[i - di + k] - Y[j - 1], 2) *
                                         (lens1[i - di + k] + lens2[j - 1]) / 2 for k in range(di))
                        cost_indep = prev + dist_indep
                        if cost_indep < best:
                            best = cost_indep
                            best_step = (di, dj, 'independent')

            D[i, j] = best
            if best_step is not None:
                back[i, j] = best_step

    # Backtrack
    matching = []
    i, j = n, m
    while i > 0 or j > 0:
        di, dj, mode = back[i, j]
        matching.append((list(range(i - di, i)), list(range(j - dj, j)), mode))
        i -= di
        j -= dj

    matching.reverse()
    return D[n, m], matching


def annotate_change_points_selection(input_sequence):
    # Set a 2x1 grid for plots
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))

    # Robust the partition
    signs_fps, signs = change_points_detection(input_sequence, return_signs=True)
    f = input_sequence

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


def _segment_label(idx):
    """Excel-style labels: A, B, ..., Z, AA, AB, ..."""
    label = ''
    n = idx
    while True:
        label = string.ascii_uppercase[n % 26] + label
        n = n // 26 - 1
        if n < 0:
            return label


def plot_two_sequences(seq1, seq2, suptitle="", vlines1=None, vlines2=None,
                       vlines_label="", matching=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(seq1, color='royalblue', label='Sequence 1')
    axes[1].plot(seq2, color='hotpink', label='Sequence 2')
    if vlines1 is not None:
        axes[0].vlines(vlines1, ymin=min(seq1), ymax=max(seq1),
                       colors='turquoise', linestyles='dashed', label=vlines_label)
    if vlines2 is not None:
        axes[1].vlines(vlines2, ymin=min(seq2), ymax=max(seq2),
                       colors='turquoise', linestyles='dashed', label=vlines_label)

    if matching is not None:
        colors = plt.cm.tab20.colors
        # Per-axis list of (seg_start, seg_end, y) for each placed label
        placed = {0: [], 1: []}
        Y_BASE = 1.02  # above the plot's top edge
        Y_STEP = 0.11
        Y_TOL = 0.025  # two labels are "on the same row" if |Δy| < this

        for idx, (x_indices, y_indices) in enumerate(matching):
            color = colors[idx % len(colors)]
            x_start, x_end = vlines1[x_indices[0]], vlines1[x_indices[-1] + 1]
            y_start, y_end = vlines2[y_indices[0]], vlines2[y_indices[-1] + 1]

            axes[0].axvspan(x_start, x_end, alpha=0.3, color=color)
            axes[1].axvspan(y_start, y_end, alpha=0.3, color=color)

            label = _segment_label(idx)

            for ax_idx, ax, seg_lo, seg_hi in [
                (0, axes[0], x_start, x_end),
                (1, axes[1], y_start, y_end),
            ]:
                # Find the lowest row whose existing segments don't overlap
                # the new one. Touching endpoints don't count as overlap.
                y = Y_BASE
                while any(
                        abs(y - pl_y) < Y_TOL
                        and seg_lo < pl_hi and pl_lo < seg_hi
                        for pl_lo, pl_hi, pl_y in placed[ax_idx]
                ):
                    y += Y_STEP
                placed[ax_idx].append((seg_lo, seg_hi, y))

                label_style = dict(ha='center', va='bottom', fontsize=13,
                                   fontweight='bold', color='black',
                                   bbox=dict(boxstyle='round,pad=0.25',
                                             facecolor='white', edgecolor=color,
                                             linewidth=1.5, alpha=0.9),
                                   clip_on=False)  # allow labels above the plot
                ax.text((seg_lo + seg_hi) / 2, y, label,
                        transform=ax.get_xaxis_transform(), **label_style)

    plt.suptitle(suptitle, fontsize=30)
    for ax_i in axes:
        ax_i.set_xlabel("Timestep", fontsize=21)
    axes[0].set_ylabel("Value", fontsize=21)
    if vlines1 and vlines2:
        axes[0].set_xticks(vlines1)
        axes[1].set_xticks(vlines2)
        axes[0].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='x', rotation=45)
    # axes[0].legend(fontsize=15)
    # axes[1].legend(fontsize=15)
    plt.tight_layout()
    plt.show()


def _stack_offset(seq1, seq2, gap_frac=0.3):
    """Vertical offset that lifts seq2 clear above seq1 for an alignment plot.

    Returns the amount to add to seq2 so its lowest point sits a `gap_frac`
    fraction of the larger amplitude above seq1's highest point.
    """
    span1 = np.max(seq1) - np.min(seq1)
    span2 = np.max(seq2) - np.min(seq2)
    span = max(span1, span2, EPSILON)
    return (np.max(seq1) - np.min(seq2)) + gap_frac * span


def plot_dtw_alignment(seq1, seq2, suptitle="DTW Alignment", path=None,
                       gap_frac=0.3, cmap="viridis"):
    """Explain the DTW distance by drawing the warping path between two
    sequences.

    The two curves are stacked (seq2 shifted vertically clear of seq1) and a
    thin line is drawn for every matched pair (i, j) on the optimal warping
    path. Each link is coloured by its local cost |seq1[i] - seq2[j]|, so the
    one-to-many fan-outs reveal *where time was warped* and the colours reveal
    *where the distance accumulates*.

    The reported distance comes from `compare_baselines.dtw_distance`, so it is
    consistent with the value used everywhere else in the project; the path
    (which `dtw_distance` does not expose) is recovered via tslearn's
    `dtw_path`, the same DTW underneath.

    :param path: optional precomputed list of (i, j) pairs. If None, it is
        recovered with tslearn.metrics.dtw_path.
    :return: (distance, path)
    """
    from tslearn.metrics import dtw_path
    from compare_baselines import dtw_distance

    seq1 = np.asarray(seq1, dtype=float)
    seq2 = np.asarray(seq2, dtype=float)

    if path is None:
        path, _ = dtw_path(seq1, seq2)
        distance = dtw_distance(seq1, seq2)
    else:
        distance = float(sum(abs(seq1[i] - seq2[j]) for i, j in path))

    offset = _stack_offset(seq1, seq2, gap_frac)
    y2 = seq2 + offset

    # Per-link cost drives the colour, so the build-up of the distance is visible.
    link_costs = np.array([abs(seq1[i] - seq2[j]) for i, j in path])
    lo, hi = float(link_costs.min()), float(link_costs.max())
    if hi - lo < EPSILON:
        hi = lo + EPSILON
    norm = plt.Normalize(lo, hi)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(16, 7))
    for (i, j), c in zip(path, link_costs):
        ax.plot([i, j], [seq1[i], y2[j]], color=cmap_obj(norm(c)),
                linewidth=1, alpha=0.6, zorder=1)

    ax.plot(range(len(seq1)), seq1, color='royalblue', linewidth=5,
            label='Sequence 1', zorder=3)
    ax.plot(range(len(seq2)), y2, color='hotpink', linewidth=5,
            label='Sequence 2 (offset)', zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Per-link cost  |seq1[i] - seq2[j]|", fontsize=25)

    ax.set_title(f"{suptitle} | Distance: {distance:.3f}", fontsize=30)
    ax.set_xlabel("Timestep", fontsize=25)
    ax.set_ylabel("Value", fontsize=25)
    ax.set_yticks([])  # the vertical offset is cosmetic, hide absolute values
    ax.legend(fontsize=18, loc='upper right')
    plt.tight_layout()
    plt.show()

    return distance, path


def plot_lcss_alignment(seq1, seq2, eps=0.1585, delta=250, n_quantiles=7,
                        translate=True, suptitle="LCSS Alignment", gap_frac=0.3):
    """Explain the LCSS distance by drawing the matched subsequence and the
    points it skips.

    The two curves are stacked and a green line is drawn for every matched pair
    in the longest common subsequence (points within `eps` in value and `delta`
    in time). Matched points are filled green; unmatched points are marked with
    red crosses. Those skipped points are the whole story: they are why LCSS is
    robust to noise where DTW is not.

    By default this reproduces the translation-invariant D2 (`lcss_d2`) that the
    project actually uses: seq2 is shifted by the best translation found with
    the *same* search as `compare_baselines.similarity_s2`
    (`_candidate_translations_1d` + `similarity_s1`), so the drawn matching is
    exactly the one `lcss_d2` scores. The reported distance comes from your
    `lcss_d2` (or `distance_d1` when `translate=False`).

    :param eps: value tolerance (matches compare_baselines).
    :param delta: time tolerance, passed as sakoe_chiba_radius.
    :param n_quantiles: number of candidate translations (as in similarity_s2).
    :param translate: if False, use the plain S1 matching / `distance_d1`.
    :return: (distance, path).
    """
    from tslearn.metrics import lcss_path
    from compare_baselines import (lcss_d2, distance_d1, similarity_s1,
                                   _candidate_translations_1d)

    seq1 = np.asarray(seq1, dtype=float)
    seq2 = np.asarray(seq2, dtype=float)

    # Find the best translation exactly as similarity_s2 does, so the drawn
    # matching corresponds to the lcss_d2 number reported below.
    best_c = 0.0
    if translate:
        best_sim = -1.0
        for cc in _candidate_translations_1d(seq1, seq2, delta, eps, n_quantiles):
            s = similarity_s1(seq1, seq2 + cc, delta, eps)
            if s > best_sim:
                best_sim, best_c = s, cc
                if best_sim == 1.0:
                    break

    seq2_shift = seq2 + best_c
    path, sim = lcss_path(seq1, seq2_shift, eps=eps,
                          global_constraint="sakoe_chiba",
                          sakoe_chiba_radius=delta)

    if translate:
        distance = lcss_d2(seq1, seq2, delta=delta, eps=eps, n_quantiles=n_quantiles)
    else:
        distance = distance_d1(seq1, seq2, delta, eps)

    matched_i = sorted({i for i, j in path})
    matched_j = sorted({j for i, j in path})
    unmatched_i = [k for k in range(len(seq1)) if k not in set(matched_i)]
    unmatched_j = [k for k in range(len(seq2)) if k not in set(matched_j)]

    offset = _stack_offset(seq1, seq2_shift, gap_frac)
    y2 = seq2_shift + offset

    fig, ax = plt.subplots(figsize=(16, 7))

    # Matched links — the longest common subsequence within (eps, delta).
    for i, j in path:
        ax.plot([i, j], [seq1[i], y2[j]], color='mediumseagreen',
                linewidth=1, alpha=0.6, zorder=1)

    ax.plot(range(len(seq1)), seq1, color='royalblue', linewidth=5,
            label='Sequence 1', zorder=2)
    ax.plot(range(len(seq2)), y2, color='hotpink', linewidth=5,
            label='Sequence 2 (offset)', zorder=2)

    # Matched points filled; unmatched points are the ones LCSS skips.
    ax.scatter(matched_i, [seq1[k] for k in matched_i],
               color='mediumseagreen', s=15, zorder=4, label='Matched')
    ax.scatter(matched_j, [y2[k] for k in matched_j],
               color='mediumseagreen', s=15, zorder=4)
    ax.scatter(unmatched_i, [seq1[k] for k in unmatched_i],
               color='red', marker='x', s=15, zorder=4, label='Skipped')
    ax.scatter(unmatched_j, [y2[k] for k in unmatched_j],
               color='red', marker='x', s=15, zorder=4)

    ax.set_title(f"{suptitle} | Similarity: {1.0 - distance:.3f} | Distance: {distance:.3f}",
                 fontsize=30)
    ax.set_xlabel("Timestep", fontsize=25)
    ax.set_ylabel("Value", fontsize=25)
    ax.set_yticks([])  # the vertical offset is cosmetic, hide absolute values
    ax.legend(fontsize=18, loc='upper right')
    plt.tight_layout()
    plt.show()

    return distance, path
