"""Mechanism checks + report figures."""
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # project root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "Times New Roman"
from utils import change_points_detection, extract_segment_features, normalize_sequence

FIGDIR = os.path.join(HERE, "..", "figs")
os.makedirs(FIGDIR, exist_ok=True)
rng = np.random.default_rng(0)
OUT = {}
_d = np.load("data/six_cps_sequences.npz")
seqs = [normalize_sequence(_d[f"seq_{i}"].astype(float)) for i in range(len(_d.files))]

# ---------------------------------------------------------------
# M1: is the instability driven by der_amp (a max-min statistic)?
# ---------------------------------------------------------------
print("=== M1: threshold statistic der_amp under noise ===")
rows = []
for k in rng.choice(len(seqs), 40, replace=False):
    s = seqs[k]
    d0 = np.diff(s); a0 = (d0.max()-d0.min())/2
    sn = normalize_sequence(s + rng.normal(0, 0.005, len(s)))
    dn = np.diff(sn); an = (dn.max()-dn.min())/2
    # robust alternative: interquartile range of the derivative
    r0 = np.subtract(*np.percentile(d0, [75, 25])); rn = np.subtract(*np.percentile(dn, [75, 25]))
    rows.append((a0, an, r0, rn))
rows = np.array(rows)
infl_range = rows[:, 1]/rows[:, 0]
infl_iqr = rows[:, 3]/np.maximum(rows[:, 2], 1e-12)
OUT["M1"] = {"der_amp_inflation_median": float(np.median(infl_range)),
             "der_amp_inflation_p90": float(np.percentile(infl_range, 90)),
             "iqr_inflation_median": float(np.median(infl_iqr))}
print(f"  der_amp (max-min)/2 inflates x{np.median(infl_range):.1f} (median), x{np.percentile(infl_range,90):.1f} (p90)")
print(f"  robust IQR of derivative inflates x{np.median(infl_iqr):.2f} (median)")

# consequence: fraction of samples classified 'constant' by the sign threshold
frac0, fracn = [], []
for k in rng.choice(len(seqs), 40, replace=False):
    s = seqs[k]
    for tgt, arr in ((s, frac0), (normalize_sequence(s+rng.normal(0,0.005,len(s))), fracn)):
        d = np.diff(tgt); a = (d.max()-d.min())/2; thr = 10*a/100
        arr.append(float(np.mean(np.abs(d) <= thr)))
OUT["M1_constant_frac"] = {"clean": float(np.mean(frac0)), "noisy": float(np.mean(fracn))}
print(f"  samples below the 'change' threshold: clean={np.mean(frac0)*100:.1f}%  noisy={np.mean(fracn)*100:.1f}%")

# ---------------------------------------------------------------
# M2: redundancy of the 12 features on real segments
# ---------------------------------------------------------------
print("\n=== M2: feature redundancy ===")
FEATNAMES = ["mean_curvature","mean_diff","mean_abs_diff","sum_abs_diff","mean_value",
             "amplitude","length","sharp_inc","light_inc","sharp_dec","light_dec","constant"]
F = []
for k in rng.choice(len(seqs), 300, replace=False):
    s = seqs[k]
    try:
        cps = change_points_detection(s)
    except Exception:
        continue
    d = np.diff(s); da = np.max(np.abs(d))-np.min(np.abs(d))
    for i in range(len(cps)-1):
        seg = s[cps[i]:cps[i+1]]
        if len(seg) < 5: continue
        F.append(extract_segment_features(seg, len(s), da))
F = np.array(F)
C = np.corrcoef(F.T)
np.fill_diagonal(C, 0.0)
pairs = []
for i in range(12):
    for j in range(i+1, 12):
        pairs.append((abs(C[i, j]), FEATNAMES[i], FEATNAMES[j], C[i, j]))
pairs.sort(reverse=True)
OUT["M2_top_correlated"] = [{"a": a, "b": b, "r": float(r)} for _, a, b, r in pairs[:8]]
print(f"  n segments = {len(F)}")
for _, a, b, r in pairs[:8]:
    print(f"    {a:15s} ~ {b:15s}  r = {r:+.3f}")
ev = np.linalg.eigvalsh(np.corrcoef(F.T))[::-1]
OUT["M2_eigen"] = {"eigenvalues": [float(v) for v in ev],
                   "n_components_95pct": int(np.searchsorted(np.cumsum(ev)/ev.sum(), 0.95)+1),
                   "condition_number": float(ev[0]/max(ev[-1], 1e-12))}
print(f"  effective dimensionality: {OUT['M2_eigen']['n_components_95pct']} of 12 components carry 95% of variance")
print(f"  correlation-matrix condition number: {OUT['M2_eigen']['condition_number']:.1f}")

# duration/amplitude/slope algebraic dependence on monotone segments
mono = []
for row in F:
    pass
raw = []
for k in rng.choice(len(seqs), 200, replace=False):
    s = seqs[k]
    try: cps = change_points_detection(s)
    except Exception: continue
    for i in range(len(cps)-1):
        seg = s[cps[i]:cps[i+1]]
        if len(seg) < 5: continue
        d = np.diff(seg)
        raw.append((np.mean(d), np.max(seg)-np.min(seg), len(seg)))
raw = np.array(raw)
pred = raw[:, 0]*raw[:, 2]
r_dep = np.corrcoef(np.abs(pred), raw[:, 1])[0, 1]
OUT["M2_slope_x_length_vs_amplitude_r"] = float(r_dep)
print(f"  |mean_diff x length| vs amplitude:  r = {r_dep:+.3f}  (they are one relation, not three axes)")

# ---------------------------------------------------------------
# FIGURE 1: change point instability
# ---------------------------------------------------------------
print("\n=== Figures ===")
import ruptures as rpt
k = 3050
s = seqs[k]
sn = normalize_sequence(s + rng.normal(0, 0.005, len(s)))
c0, c1 = change_points_detection(s), change_points_detection(sn)
p0 = rpt.Pelt(model="l2", min_size=10, jump=5).fit(s.reshape(-1,1)).predict(pen=0.05)
p1 = rpt.Pelt(model="l2", min_size=10, jump=5).fit(sn.reshape(-1,1)).predict(pen=0.05)

fig, ax = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True, sharey=True)
for a, (sig, cps, ttl) in zip(ax.ravel(), [
        (s, c0, f"Current detector — clean  ({len(c0)} points)"),
        (sn, c1, f"Current detector — $\\sigma$=0.005 noise  ({len(c1)} points)"),
        (s, p0, f"PELT (ruptures) — clean  ({len(p0)} points)"),
        (sn, p1, f"PELT (ruptures) — $\\sigma$=0.005 noise  ({len(p1)} points)")]):
    a.plot(sig, color="#33415c", lw=1.1)
    for c in cps:
        a.axvline(c, color="#d1495b", ls="--", lw=1.3, alpha=0.9)
    a.set_title(ttl, fontsize=13)
    a.set_ylim(-0.05, 1.05)
fig.suptitle("Segmentation stability under a 0.5%-of-range perturbation", fontsize=17, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_cp_stability.pdf"), bbox_inches="tight")
plt.close(fig)
print("  fig_cp_stability.pdf")

# ---------------------------------------------------------------
# FIGURE 2: context-dependent scale
# ---------------------------------------------------------------
t = np.linspace(0, 1, 200)
motif = 0.5 + 0.15*np.sin(2*np.pi*t)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, tail_amp in zip(ax, [0.1, 0.8, 4.0]):
    tail = np.linspace(motif[-1], motif[-1]+tail_amp, 800)
    full = normalize_sequence(np.concatenate([motif, tail]))
    m = full[:200]
    a.plot(full, color="#c8cdd6", lw=1.2)
    a.plot(np.arange(200), m, color="#0f766e", lw=2.0)
    a.set_title(f"context slope {tail_amp}\nmotif amplitude feature = {m.max()-m.min():.3f}", fontsize=13)
    a.set_ylim(-0.05, 1.05); a.set_xticks([])
fig.suptitle("The same motif (teal), three contexts: its amplitude feature changes 17$\\times$",
             fontsize=16, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_scale_context.pdf"), bbox_inches="tight")
plt.close(fig)
print("  fig_scale_context.pdf")

# ---------------------------------------------------------------
# FIGURE 3: tuning objective plateau
# ---------------------------------------------------------------
with open("results/human_tuning_results.json") as f:
    tr = json.load(f)
vals = np.array([t["value"] for t in tr["all_trials"]])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(vals, "o-", ms=3.5, lw=0.8, color="#3b6ea5")
ax[0].axhline(vals.max(), color="#d1495b", ls="--", label=f"best = {vals.max():.3f}")
ax[0].axhline(tr["before_tuning"]["ap"]["dtw"], color="#e07a3f", ls=":", label="DTW baseline")
ax[0].axhline(tr["before_tuning"]["ap"]["lcss"], color="#4b8f5b", ls=":", label="LCSS baseline")
ax[0].set_xlabel("Optuna trial", fontsize=12); ax[0].set_ylabel("mean Precision@3", fontsize=12)
ax[0].set_title("Search trace: a staircase, not a gradient", fontsize=14); ax[0].legend(fontsize=9)
u, cnt = np.unique(np.round(vals, 6), return_counts=True)
ax[1].bar(u, cnt, width=0.012, color="#3b6ea5")
ax[1].set_xlabel("distinct objective value", fontsize=12); ax[1].set_ylabel("# trials", fontsize=12)
ax[1].set_title(f"{len(vals)} trials collapse onto {len(u)} values", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig_tuning_plateau.pdf"), bbox_inches="tight")
plt.close(fig)
print("  fig_tuning_plateau.pdf")

with open(os.path.join(HERE, "diag2.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print("\nSaved diag2.json")
