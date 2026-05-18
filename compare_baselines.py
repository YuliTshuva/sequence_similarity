import numpy as np
from typing import Optional
from seq_sim_alg import seq_distance


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


def lcss_distance(a: np.ndarray, b: np.ndarray, eps: float = 0.1) -> float:
    """
    LCSS-based distance: 1 - (longest common subsequence length / max sequence length).
    Two elements match if their absolute difference is within eps.
    Returns a value in [0, 1], where 0 = identical, 1 = no match.
    """
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(a[i - 1] - b[j - 1]) <= eps:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    similarity = dp[n, m] / max(n, m)
    return float(1.0 - similarity)


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


def all_distances(a, b, dtw_band: float = 0.1, lcss_eps: float = 0.1) -> dict:
    """
    Compute all baseline distances between sequences a and b.

    Parameters
    ----------
    a, b       : array-like, values in [0, 1]
    dtw_band   : Sakoe-Chiba band as a fraction of sequence length (default 0.1)
    lcss_eps   : matching tolerance for LCSS (default 0.1)

    Returns
    -------
    dict with keys: euclidean, dtw, dtw_sakoe_chiba, lcss, pearson
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return {
        "euclidean": euclidean_distance(a, b),
        "dtw": dtw_distance(a, b),
        "dtw_sakoe_chiba": dtw_sakoe_chiba(a, b, band=dtw_band),
        "lcss": lcss_distance(a, b, eps=lcss_eps),
        "pearson": pearson_distance(a, b),
        "ours": seq_distance(a, b)[0] + seq_distance(b, a)[0]
    }


# --- Example usage ---
def main():
    a = [0.1, 0.3, 0.6, 0.8, 0.9, 0.7, 0.4, 0.2]
    b = [0.0, 0.2, 0.5, 0.9, 0.8, 0.6, 0.3, 0.1]

    scores = all_distances(a, b)
    for method, score in scores.items():
        print(f"{method:<20} {score:.4f}" if score is not None else f"{method:<20} n/a")


if __name__ == "__main__":
    main()
