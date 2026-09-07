"""Visual demo: how each Sec. 2.2 candidate segments sequences from data/six_cps_sequences.npz.

Rows = methods, columns = sequences. Red lines are the detected change points.
Every method is asked for the same number of change points (K_TARGET) so that what you
see is WHERE each method puts its boundaries, not how many it returns.

  current  -- utils.change_points_detection            (what the pipeline uses today)
  l1tf     -- l1 trend filtering (blue = the fitted piecewise-linear model)
  pelt     -- PELT, l2 cost on the derivative
  pelt-l1  -- PELT, l1 cost on the derivative (outlier-robust)
  pip      -- perceptually important points

Writes reports/figs/cpd_methods_clean.png and cpd_methods_noisy.png
"""
import sys, os, warnings, heapq
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ruptures as rpt

from utils import change_points_detection, normalize_sequence

K_TARGET = 6
SEQ_IDS = [0, 2, 5, 7]
PIP_SEQ_IDS = [2, 0, 5, 7, 13]          # rows of the PIP hierarchy figure

_d = np.load("data/six_cps_sequences.npz")
seqs = [normalize_sequence(np.asarray(_d[f"seq_{i}"], float)) for i in SEQ_IDS]
pip_seqs = [normalize_sequence(np.asarray(_d[f"seq_{i}"], float)) for i in PIP_SEQ_IDS]
n = len(seqs[0])
MIN_GAP = max(5, int(0.01 * n))


def thin(idx, min_gap=MIN_GAP):
    if len(idx) == 0:
        return np.array([], int)
    out, run = [], [idx[0]]
    for i in idx[1:]:
        if i - run[-1] <= min_gap:
            run.append(i)
        else:
            out.append(int(np.mean(run))); run = [i]
    out.append(int(np.mean(run)))
    return np.array(out, int)


# ------------------------------------------------------------- methods
def cps_current(y, K=None):
    cp = np.asarray(change_points_detection(y), int).ravel()
    return np.unique(cp[(cp > 0) & (cp < len(y) - 1)])


_e = np.ones(n)
_D = sp.diags([_e[:-2], -2 * _e[1:-1], _e[2:]], [0, 1, 2], shape=(n - 2, n), format="csc")
_FACT = spla.factorized((sp.identity(n, format="csc") + (_D.T @ _D)).tocsc())


def l1tf_fit(y, lam, iters=400):
    """min 1/2||y-x||^2 + lam*||D2 x||_1, solved by ADMM. Convex -> unique solution.
    x is piecewise linear; the support of z is its knot set."""
    z = np.zeros(n - 2); u = np.zeros(n - 2); x = np.asarray(y, float)
    for _ in range(iters):
        x = _FACT(y + _D.T @ (z - u))
        Dx = _D @ x
        v = Dx + u
        z = np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)
        u += Dx - z
    return x, z


def _bisect(count, lo, hi, K, iters=22):
    """Find the parameter giving ~K change points (both knobs are monotone)."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if count(mid) > K:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def pwl_refit(y, knots):
    """Exact continuous piecewise-linear least-squares fit with the given knots.
    Basis: {1, t, (t-k)_+}. ADMM's iterate is only approximately piecewise linear;
    this makes the returned trend model exactly so."""
    t = np.arange(n, dtype=float)
    B = [np.ones(n), t] + [np.maximum(t - k, 0.0) for k in knots]
    B = np.column_stack(B)
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return B @ coef


def cps_l1tf(y, K=K_TARGET, return_fit=False):
    # bisect on the RAW support size of z: it is monotone in lambda.
    # (thinning first would collapse a dense support to one point and break monotonicity)
    lam = _bisect(lambda l: int((np.abs(l1tf_fit(y, l)[1]) > 0).sum()), 1e-6, 1e1, K, 20)
    _, z = l1tf_fit(y, lam)
    cp = thin(np.where(np.abs(z) > 0)[0] + 1)
    return (cp, pwl_refit(y, cp)) if return_fit else cp


def _pelt(y, pen, model):
    d = np.diff(y).reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=MIN_GAP, jump=5).fit(d)
    return np.array([b for b in algo.predict(pen=max(pen, 1e-15)) if 0 < b < n - 1], int)


def cps_pelt(y, K=K_TARGET, model="l2"):
    pen = _bisect(lambda p: len(_pelt(y, p, model)), 1e-12, 1e2, K, 24)
    return _pelt(y, pen, model)


def cps_pip(y, K=K_TARGET, return_order=False):
    """Recursively take the point of maximal vertical deviation from the current chord."""
    def best(a, b):
        if b - a < 2:
            return None
        i = np.arange(a + 1, b)
        dev = np.abs((y[b] - y[a]) * (i - a) / (b - a) + y[a] - y[i])
        return (-dev.max(), int(i[np.argmax(dev)]), a, b)
    heap = [h for h in [best(0, n - 1)] if h]
    order = []
    while heap and len(order) < K:
        _, j, a, b = heapq.heappop(heap)
        order.append(j)
        for seg in ((a, j), (j, b)):
            h = best(*seg)
            if h:
                heapq.heappush(heap, h)
    o = np.array(sorted(order), int)
    return (o, order) if return_order else o


METHODS = [
    ("current",  cps_current),
    ("l1tf",     cps_l1tf),
    ("pelt",     lambda y, K=K_TARGET: cps_pelt(y, K, "l2")),
    ("pelt-l1",  lambda y, K=K_TARGET: cps_pelt(y, K, "l1")),
    ("pip",      cps_pip),
]

# ------------------------------------------------------------- plotting
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("reports/figs", exist_ok=True)


def panel(signals, fname, title):
    fig, axes = plt.subplots(len(METHODS), len(signals),
                             figsize=(3.3 * len(signals), 1.75 * len(METHODS)),
                             sharex=True, squeeze=False)
    for c, s in enumerate(signals):
        for r, (name, fn) in enumerate(METHODS):
            ax = axes[r][c]
            ax.plot(s, lw=0.8, color="0.4")
            if name == "l1tf":
                cp, fit = cps_l1tf(s, K_TARGET, return_fit=True)
                ax.plot(fit, lw=1.3, color="tab:blue")
            else:
                cp = fn(s)
            for p in cp:
                ax.axvline(p, color="crimson", lw=1.1)
            ax.set_yticks([])
            ax.set_title(f"K={len(cp)}", fontsize=8, pad=2)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
            if r == 0:
                ax.text(.5, 1.45, f"sequence {SEQ_IDS[c]}", transform=ax.transAxes,
                        ha="center", fontsize=11)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"reports/figs/{fname}", dpi=140)
    plt.close(fig)
    print("wrote reports/figs/" + fname)


panel(seqs, "cpd_methods_clean.png",
      f"Change-point detection on your sequences (each method asked for K={K_TARGET})")

rng = np.random.default_rng(0)
noisy = [y + rng.normal(0, 0.005, n) for y in seqs]
panel(noisy, "cpd_methods_noisy.png",
      "Same sequences + invisible noise ($\\sigma$=0.005): who keeps the same segmentation?")

# --------------------------- PIP hierarchy: one sequence, growing K
KS = [2, 4, 8, 16]
fig, axes = plt.subplots(len(pip_seqs), len(KS) + 1, figsize=(16, 2.1 * len(pip_seqs)),
                         sharex=True, squeeze=False)
for r, y in enumerate(pip_seqs):
    _, order = cps_pip(y, max(KS), return_order=True)
    axes[r][0].plot(y, lw=1.4, color="royalblue")
    axes[r][0].set_yticks([])
    axes[r][0].set_ylabel(f"sequence {PIP_SEQ_IDS[r]}", fontsize=10)
    if r == 0:
        axes[r][0].set_title("original", fontsize=11)
    for c, k in enumerate(KS, start=1):
        ax = axes[r][c]
        ax.plot(y, lw=1.4, color="0.7")
        pts = [0] + sorted(order[:k]) + [n - 1]
        ax.plot(pts, y[pts], "o-", color="crimson", ms=3, lw=1.0)
        ax.set_yticks([])
        if r == 0:
            ax.set_title(f"K={k}", fontsize=11)
fig.suptitle("PIP gives a nested hierarchy: coarser K is always a subset of finer K", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("reports/figs/cpd_pip_hierarchy.png", dpi=140)
print("wrote reports/figs/cpd_pip_hierarchy.png")

# --------------------------- l1tf: the model it fits, at three lambdas
fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
for ax, k in zip(axes, [3, 6, 12]):
    cp, fit = cps_l1tf(y, k, return_fit=True)
    ax.plot(y, lw=1.6, color="0.65")
    ax.plot(fit, lw=1.4, color="tab:blue")
    for p in cp:
        ax.axvline(p, color="crimson", lw=1.0)
    ax.set_title(f"$\\ell_1$-TF, K={len(cp)}", fontsize=10)
    ax.set_yticks([])
fig.suptitle("$\\ell_1$ trend filtering returns a fitted piecewise-linear model, "
             "not just cut positions", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("reports/figs/cpd_l1tf_model.png", dpi=140)
print("wrote reports/figs/cpd_l1tf_model.png")
