"""
Yuli Tshuva
Measure how visually complex a mined sequence is, so the study can keep the
middle band and drop curves that are too plain or too busy to judge.

Three complementary measures, all computed on the curve resampled to the length
a rater actually sees -- complexity of the rendered line, not of the raw ticks:

  turning_points  -- how many direction changes; the most direct proxy for
                     "how many moves does the eye have to track"
  perm_entropy    -- Weighted-free permutation entropy (Bandt & Pompe 2002),
                     normalised to [0, 1]; 0 is a monotone ramp, 1 is noise
  efficiency      -- |end - start| / total travel; 1 is a straight line,
                     near 0 is a curve that wanders and returns

They disagree in useful ways: a clean sine has few turning points but low
efficiency, while a noisy ramp has high efficiency and high entropy.
"""

import math

import numpy as np

EPS = 1e-12
DEFAULT_LENGTH = 128
DEFAULT_PERM_ORDER = 4
TURNING_POINT_TOL = 0.02  # ignore wiggles smaller than 2% of the curve's range


def _prepare(seq, length=DEFAULT_LENGTH):
    """Resample to the rendered length and scale to [0, 1]."""
    seq = np.asarray(seq, dtype=float)
    out = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(seq)), seq)
    span = out.max() - out.min()
    return (out - out.min()) / span if span > EPS else out * 0.0


def turning_points(curve, tol=TURNING_POINT_TOL):
    """
    Count direction changes, ignoring reversals smaller than `tol` of the range.

    Walks the curve and only registers a turn once the move away from the last
    extreme is large enough to be visible, which keeps smoothing noise from
    inflating the count.
    """
    count, direction, last_extreme = 0, 0, curve[0]
    for value in curve[1:]:
        if direction == 0:
            # Still flat: wait for the first move big enough to define a direction.
            if abs(value - last_extreme) > tol:
                direction = 1 if value > last_extreme else -1
                last_extreme = value
        elif direction == 1:
            if value > last_extreme:
                last_extreme = value           # still climbing
            elif last_extreme - value > tol:
                count += 1                     # fell far enough to count as a turn
                direction, last_extreme = -1, value
        else:
            if value < last_extreme:
                last_extreme = value           # still falling
            elif value - last_extreme > tol:
                count += 1
                direction, last_extreme = 1, value
    return count


def perm_entropy(curve, order=DEFAULT_PERM_ORDER):
    """Normalised permutation entropy: 0 for a monotone curve, 1 for white noise."""
    n = len(curve) - order + 1
    windows = np.lib.stride_tricks.sliding_window_view(curve, order)[:n]
    patterns = np.argsort(windows, axis=1)
    codes = np.zeros(n, dtype=np.int64)
    for position in range(order):
        codes = codes * order + patterns[:, position]
    counts = np.bincount(codes)
    probs = counts[counts > 0] / n
    return float(-(probs * np.log(probs)).sum() / np.log(math.factorial(order)))


def efficiency(curve):
    """Net displacement over total travel: 1 is a straight line, 0 wanders back."""
    travel = np.abs(np.diff(curve)).sum()
    return float(abs(curve[-1] - curve[0]) / travel) if travel > EPS else 1.0


def measure(seq, length=DEFAULT_LENGTH):
    """All three measures for one sequence."""
    curve = _prepare(seq, length)
    return {
        "turning_points": turning_points(curve),
        "perm_entropy": perm_entropy(curve),
        "efficiency": efficiency(curve),
    }


def measure_all(sequences, length=DEFAULT_LENGTH):
    """Measures for a list of sequences, as a dict of arrays."""
    rows = [measure(s, length) for s in sequences]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}
