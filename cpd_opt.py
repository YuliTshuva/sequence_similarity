"""
Yuli Tshuva
Change point detection as a global optimization problem.

Replaces the heuristic cascade in utils.change_points_detection with an exact
dynamic program:

    min over K and 0 = t_0 < t_1 < ... < t_K = n  of
        sum_k c(t_k, t_{k+1}) + beta * K

Two segment costs are provided, both O(1) per interval via prefix sums:

  * LinearCost   -- SSE of a least-squares line fit on the raw segment.
                    Classical piecewise-linear segmentation; matches the
                    "a segment is a trend" semantics of the graph nodes.
  * FeatureCost  -- weighted within-segment scatter of the pointwise analogues
                    of the segment features used by the distance (level, slope,
                    abs slope, curvature), weighted by the tuned FEATURE_WEIGHTS.
                    Segments are then homogeneous in the same space the metric
                    measures.

The minimum-segment-length constraint is enforced inside the recursion, so it is
globally optimal rather than an order-dependent post-hoc merging pass.
"""

import numpy as np

INF = float('inf')

# Weights of (mean_value, mean_diff, mean_abs_diff, mean_curvature) taken from
# the tuned FEATURE_WEIGHTS in seq_sim_alg.py. Those four segment features are
# exactly the means of the four pointwise channels used by FeatureCost.
DEFAULT_CHANNEL_WEIGHTS = np.array([0.166547, 0.155970, 0.149875, 0.089638])


def _cumsum0(a):
    """Prefix sums with a leading zero, so sum(a[i:j]) == out[j] - out[i]."""
    out = np.zeros(len(a) + 1, dtype=float)
    np.cumsum(a, out=out[1:])
    return out


class LinearCost:
    """c(a, b) = residual sum of squares of the least-squares line on y[a:b]."""

    n_params = 2

    def __init__(self, y):
        y = np.asarray(y, dtype=float).ravel()
        self.y = y
        self.n = y.size
        x = np.arange(self.n, dtype=float)
        self._x = _cumsum0(x)
        self._y = _cumsum0(y)
        self._xx = _cumsum0(x * x)
        self._xy = _cumsum0(x * y)
        self._yy = _cumsum0(y * y)

    def __call__(self, a, b):
        """a: array of start indices, b: a single end index (exclusive)."""
        a = np.asarray(a)
        m = (b - a).astype(float)
        sx = self._x[b] - self._x[a]
        sy = self._y[b] - self._y[a]
        sxx = self._xx[b] - self._xx[a]
        sxy = self._xy[b] - self._xy[a]
        syy = self._yy[b] - self._yy[a]

        cxx = sxx - sx * sx / m
        cxy = sxy - sx * sy / m
        cyy = syy - sy * sy / m

        safe = cxx > 1e-12
        slope_term = np.where(safe, cxy * cxy / np.where(safe, cxx, 1.0), 0.0)
        return np.maximum(cyy - slope_term, 0.0)

    def noise_var(self):
        """Robust sigma^2 from second differences, using Var(d2 y) = 6 sigma^2."""
        d2 = np.diff(self.y, n=2)
        if d2.size == 0:
            return 1e-12
        mad = np.median(np.abs(d2 - np.median(d2)))
        return max((1.4826 * mad) ** 2 / 6.0, 1e-12)


class FeatureCost:
    """c(a, b) = sum_t || w * (psi(t) - mean of psi over [a, b)) ||^2.

    psi(t) = (level, slope, abs slope, curvature), each z-scored over the whole
    sequence so the channels are commensurable before weighting.
    """

    n_params = 4

    def __init__(self, y, weights=DEFAULT_CHANNEL_WEIGHTS):
        y = np.asarray(y, dtype=float).ravel()
        self.y = y
        self.n = y.size

        d1 = np.gradient(y)
        d2 = np.gradient(d1)
        psi = np.column_stack([y, d1, np.abs(d1), np.abs(d2)])

        sd = psi.std(axis=0)
        sd[sd < 1e-12] = 1.0
        psi = (psi - psi.mean(axis=0)) / sd
        psi = psi * np.asarray(weights, dtype=float)

        self.psi = psi
        self._s = np.vstack([np.zeros(psi.shape[1]), np.cumsum(psi, axis=0)])
        self._ss = np.vstack([np.zeros(psi.shape[1]), np.cumsum(psi ** 2, axis=0)])

    def __call__(self, a, b):
        a = np.asarray(a)
        m = (b - a).astype(float)[:, None]
        s = self._s[b] - self._s[a]
        ss = self._ss[b] - self._ss[a]
        return np.maximum((ss - s * s / m).sum(axis=1), 0.0)

    def noise_var(self):
        d2 = np.diff(self.psi, n=2, axis=0)
        if d2.shape[0] == 0:
            return 1e-12
        mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0)
        return max(float(np.sum((1.4826 * mad) ** 2 / 6.0)), 1e-12)


def optimal_change_points(sequence, cost="linear", beta=None, beta_rel=0.02,
                          bic_scale=2.0, min_len=None, n_bkps=None,
                          weights=DEFAULT_CHANNEL_WEIGHTS, return_cost=False):
    """Globally optimal segmentation of `sequence`.

    cost     : "linear" | "feature", or any object with __call__(a, b) and noise_var().
    beta     : per-change-point penalty. A float is used as-is; "bic" uses
               bic_scale * sigma^2 * p * log(n); None uses the relative penalty below.
    beta_rel : scale-free penalty, beta = beta_rel * c(0, n). A new change point must
               cut the total cost by at least this fraction of the one-segment cost,
               which makes the knob comparable across sequences. This is the single
               parameter to tune against the downstream objective.
    n_bkps   : if given, run the fixed-K dynamic program and ignore beta.
    min_len  : minimum samples per segment (default max(5, 1% of n)).

    Returns the boundary list in the same convention as
    utils.change_points_detection: starts at 0 and ends at len(sequence) - 1.
    """
    y = np.asarray(sequence, dtype=float).ravel()
    n = y.size
    if min_len is None:
        min_len = max(5, n // 100)

    if cost == "linear":
        cost_fn = LinearCost(y)
    elif cost == "feature":
        cost_fn = FeatureCost(y, weights)
    else:
        cost_fn = cost

    if n_bkps is not None:
        bounds, total = _dp_fixed_k(cost_fn, n, min_len, n_bkps)
    else:
        if beta is None:
            beta = beta_rel * float(cost_fn(np.array([0]), n)[0])
        elif beta == "bic":
            beta = bic_scale * cost_fn.noise_var() * cost_fn.n_params * np.log(n)
        bounds, total = _dp_penalized(cost_fn, n, min_len, beta)

    bounds = list(bounds)
    bounds[-1] = n - 1  # match the existing convention
    return (bounds, total) if return_cost else bounds


def cost_curve(sequence, cost="linear", kmax=12, min_len=None,
               weights=DEFAULT_CHANNEL_WEIGHTS):
    """Optimal cost and boundaries for every K from 0 to kmax, in one DP pass.

    Returns (costs, boundaries) where costs[k] is the lowest achievable cost with
    k change points and boundaries[k] is the segmentation that attains it.
    """
    y = np.asarray(sequence, dtype=float).ravel()
    n = y.size
    if min_len is None:
        min_len = max(5, n // 100)

    if cost == "linear":
        cost_fn = LinearCost(y)
    elif cost == "feature":
        cost_fn = FeatureCost(y, weights)
    else:
        cost_fn = cost

    n_seg = kmax + 1
    C = np.full((n_seg + 1, n + 1), INF)
    C[0, 0] = 0.0
    par = np.zeros((n_seg + 1, n + 1), dtype=int)

    for k in range(1, n_seg + 1):
        for t in range(k * min_len, n + 1):
            starts = np.arange((k - 1) * min_len, t - min_len + 1)
            if starts.size == 0:
                continue
            prev = C[k - 1, starts]
            ok = np.isfinite(prev)
            starts, prev = starts[ok], prev[ok]
            if starts.size == 0:
                continue
            total = prev + cost_fn(starts, t)
            j = int(np.argmin(total))
            C[k, t] = total[j]
            par[k, t] = starts[j]

    costs, boundaries = [], []
    for k in range(1, n_seg + 1):
        b, t = [n], n
        for kk in range(k, 0, -1):
            t = par[kk, t]
            b.append(t)
        b = list(reversed(b))
        b[-1] = n - 1
        costs.append(C[k, n])
        boundaries.append(b)
    return np.array(costs), boundaries


def change_points_by_tolerance(sequence, tol=0.05, cost="linear", kmax=12,
                               min_len=None, weights=DEFAULT_CHANNEL_WEIGHTS,
                               return_k=False):
    """Smallest segmentation that explains the sequence to within `tol`.

    Takes the fewest change points K such that the leftover cost is at most a
    fraction `tol` of the cost of the best single segment:

        K* = min { K : c_K / c_0 <= tol }

    So tol = 0.05 means "the fit must account for 95% of what one straight line
    leaves unexplained". Unlike a penalty on K, this does not get harsher on
    sequences with a large overall swing, so small features stay affordable.
    """
    costs, boundaries = cost_curve(sequence, cost=cost, kmax=kmax,
                                   min_len=min_len, weights=weights)
    c0 = costs[0]
    if c0 <= 0:
        return (boundaries[0], 0) if return_k else boundaries[0]
    ratio = costs / c0
    ok = np.flatnonzero(ratio <= tol)
    i = int(ok[0]) if ok.size else len(costs) - 1
    return (boundaries[i], i) if return_k else boundaries[i]


def _dp_penalized(cost_fn, n, min_len, beta):
    F = np.full(n + 1, INF)
    F[0] = 0.0
    parent = np.zeros(n + 1, dtype=int)

    all_starts = np.arange(n + 1)
    for t in range(min_len, n + 1):
        # Valid predecessors: s == 0, or s >= min_len, and t - s >= min_len.
        starts = all_starts[:t - min_len + 1]
        starts = starts[(starts == 0) | (starts >= min_len)]
        starts = starts[np.isfinite(F[starts])]
        if starts.size == 0:
            continue
        total = F[starts] + cost_fn(starts, t) + beta
        k = int(np.argmin(total))
        F[t] = total[k]
        parent[t] = starts[k]

    bounds, t = [n], n
    while t > 0:
        t = parent[t]
        bounds.append(t)
    # The penalty was also charged for the final (non-)cut, so remove it once.
    return list(reversed(bounds)), F[n] - beta


def _dp_fixed_k(cost_fn, n, min_len, n_bkps):
    n_seg = n_bkps + 1
    C = np.full((n_seg + 1, n + 1), INF)
    C[0, 0] = 0.0
    par = np.zeros((n_seg + 1, n + 1), dtype=int)

    for k in range(1, n_seg + 1):
        for t in range(k * min_len, n + 1):
            starts = np.arange((k - 1) * min_len, t - min_len + 1)
            if starts.size == 0:
                continue
            prev = C[k - 1, starts]
            ok = np.isfinite(prev)
            starts, prev = starts[ok], prev[ok]
            if starts.size == 0:
                continue
            total = prev + cost_fn(starts, t)
            j = int(np.argmin(total))
            C[k, t] = total[j]
            par[k, t] = starts[j]

    bounds, t, k = [n], n, n_seg
    while k > 0:
        t = par[k, t]
        bounds.append(t)
        k -= 1
    return list(reversed(bounds)), C[n_seg, n]
