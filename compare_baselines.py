import numpy as np
from typing import Optional

from scipy.spatial.distance import (
    euclidean as _scipy_euclidean,
    correlation as _scipy_correlation,
)
from tslearn.metrics import dtw as _tslearn_dtw, lcss as _tslearn_lcss


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Euclidean (L2) distance between two equal-length sequences.
    Returns None if sequences have different lengths.

    Implementation: scipy.spatial.distance.euclidean.
    """
    if len(a) != len(b):
        return None
    return float(_scipy_euclidean(a, b))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Unconstrained Dynamic Time Warping (DTW) distance.
    Works with sequences of different lengths.

    Implementation: tslearn.metrics.dtw (Tavenard et al., 2020),
    following Sakoe & Chiba (1978).
    """
    return float(_tslearn_dtw(a, b))


def dtw_sakoe_chiba(a: np.ndarray, b: np.ndarray, band: float = 0.1) -> float:
    """
    DTW with Sakoe-Chiba band constraint.
    band: fraction of max(len(a), len(b)) to use as window width (default 10%).

    Implementation: tslearn.metrics.dtw with sakoe_chiba global constraint
    (Tavenard et al., 2020), following Sakoe & Chiba (1978). Note that
    tslearn's sakoe_chiba_radius is an integer radius in index units, so we
    convert from the fractional `band` here.
    """
    n, m = len(a), len(b)
    radius = max(int(band * max(n, m)), abs(n - m))
    return float(_tslearn_dtw(
        a, b,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=radius,
    ))


def lcss_delta_eps(a, b, delta, eps):
    """
    Raw LCSS count of Vlachos, Kollios & Gunopulos (2002): the length of the
    longest common subsequence within delta time tolerance and eps value
    tolerance.

    Implementation: tslearn.metrics.lcss returns the ratio normalized by
    min(len(a), len(b)); we multiply back to recover the integer count.
    Per tslearn's documentation, sakoe_chiba_radius corresponds to the
    parameter delta in Vlachos et al.
    """
    ratio = _tslearn_lcss(
        a, b,
        eps=eps,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=delta,
    )
    return int(round(ratio * min(len(a), len(b))))


def similarity_s1(a, b, delta, eps):
    """
    LCSS similarity S1 of Vlachos, Kollios & Gunopulos (2002):
    LCSS(a, b) / min(len(a), len(b)).

    Implementation: tslearn.metrics.lcss with sakoe_chiba global constraint
    (delta corresponds to sakoe_chiba_radius per tslearn's documentation).
    """
    return float(_tslearn_lcss(
        a, b,
        eps=eps,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=delta,
    ))


def distance_d1(a, b, delta, eps):
    """
    LCSS-based distance D1 = 1 - S1 of Vlachos, Kollios & Gunopulos (2002).
    """
    return 1.0 - similarity_s1(a, b, delta, eps)


# ---------------------------------------------------------------------------
# Translation-invariant LCSS (S2 / D2) of Vlachos, Kollios & Gunopulos (2002).
# No Python library exposes this variant. The closest author-maintained
# reference implementation is the R package `SimilarityMeasures` by
# K. Toohey (CRAN, 2015). The code below is a re-implementation that follows
# the paper, with candidate translations discretized via n_quantiles
# quantiles for tractability.
# ---------------------------------------------------------------------------


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

    Implementation: scipy.spatial.distance.correlation, which returns 1 - r.
    """
    if len(a) != len(b):
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(_scipy_correlation(a, b))