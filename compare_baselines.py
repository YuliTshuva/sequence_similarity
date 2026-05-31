import numpy as np
from typing import Optional
from itertools import product


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Euclidean (L2) distance between two equal-length sequences.
    Returns None if sequences have different lengths.
    """
    if len(a) != len(b):
        return None
    return float(np.sqrt(np.sum((a - b) ** 2)))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Unconstrained Dynamic Time Warping (DTW) distance.
    Works with sequences of different lengths.
    """
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[n, m])


def dtw_sakoe_chiba(a: np.ndarray, b: np.ndarray, band: float = 0.1) -> float:
    """
    DTW with Sakoe-Chiba band constraint.
    band: fraction of max(len(a), len(b)) to use as window width (default 10%).
    """
    n, m = len(a), len(b)
    w = max(int(band * max(n, m)), abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[n, m])


def lcss_delta_eps(a, b, delta, eps):
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        j_lo = max(1, i - delta)
        j_hi = min(m, i + delta)
        for j in range(j_lo, j_hi + 1):
            if np.all(np.abs(a[i - 1] - b[j - 1]) <= eps):
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[n, m])


def similarity_s1(a, b, delta, eps):
    return lcss_delta_eps(a, b, delta, eps) / min(len(a), len(b))


def distance_d1(a, b, delta, eps):
    return 1.0 - similarity_s1(a, b, delta, eps)


def _candidate_translations_1d(ax, bx, delta, eps, n_quantiles):
    candidates = []
    for i in range(len(ax)):
        for j in range(len(bx)):
            if abs(i - j) <= delta:
                candidates.append(ax[i] - bx[j] - eps)
                candidates.append(ax[i] - bx[j] + eps)
    if not candidates:
        return np.array([0.0])
    candidates = np.unique(candidates)
    if n_quantiles >= len(candidates):
        return candidates
    indices = np.round(np.linspace(0, len(candidates) - 1, n_quantiles)).astype(int)
    return candidates[indices]


def similarity_s2(a, b, delta, eps, n_quantiles=7):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c_list = _candidate_translations_1d(a, b, delta, eps, n_quantiles)
    best = 0.0
    for c in c_list:
        s = similarity_s1(a, b + c, delta, eps)
        if s > best:
            best = s
            if best == 1.0:
                break
    return best


def lcss_d2(a, b, delta=250, eps=0.1585, n_quantiles=7):
    """
    In the paper they used:
    * delta = 20-30% of the sequence length
    * eps = the smallest standard deviation
    """
    return 1.0 - similarity_s2(a, b, delta, eps, n_quantiles)


def pearson_distance(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Pearson correlation distance: 1 - r.
    Requires equal-length sequences. Returns None if a sequence has zero variance.
    Range: 0 (identical) to 2 (perfectly anti-correlated).
    """
    if len(a) != len(b):
        return None
    std_a, std_b = np.std(a), np.std(b)
    if std_a == 0 or std_b == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return 1.0 - r