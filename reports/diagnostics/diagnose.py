"""Diagnostics for the four stated weaknesses of the GDTW method."""
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # project root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import kendalltau
from utils import (change_points_detection, extract_segment_features,
                   normalize_sequence, merge_segments_features, features_correlation)
rng = np.random.default_rng(0)
OUT = {}

print("Loading sequences...")
_d = np.load("data/six_cps_sequences.npz")
seqs = [_d[f"seq_{i}"] for i in range(len(_d.files))]
seqs = [normalize_sequence(np.asarray(s, float)) for s in seqs]
print(f"  {len(seqs)} sequences, len={len(seqs[0])}")

FEATNAMES = ["mean_curvature","mean_diff","mean_abs_diff","sum_abs_diff","mean_value",
             "amplitude","length","sharp_inc","light_inc","sharp_dec","light_dec","constant"]

# raw (unstandardised) feature extraction, to see the true scaling behaviour
def raw_features(segment, len_sequence, der_amp):
    seg = np.asarray(segment, float)
    d = np.diff(seg)
    thr = 10 * der_amp / 100
    n = len(seg)
    si = np.sum(d > thr)/n; li = np.sum(d > thr/3)/n - si
    sd = np.sum(d < -thr)/n; ld = np.sum(d < -thr/3)/n - sd
    return np.array([np.mean(np.abs(np.diff(seg, n=2))), np.mean(d), np.mean(np.abs(d)),
                     np.sum(np.abs(d)), np.mean(seg), np.max(seg)-np.min(seg),
                     n/len_sequence, si, li, sd, ld, 1-si-li-sd-ld])

def resample(x, n):
    x = np.asarray(x, float)
    return np.interp(np.linspace(0, len(x)-1, n), np.arange(len(x)), x)

# =====================================================================
# E1  x-axis scale: same SHAPE, different sampling resolution
# =====================================================================
print("\n=== E1: feature drift under pure time re-sampling (shape identical) ===")
base = seqs[3050]
cps = change_points_detection(base)
seg = base[cps[1]:cps[2]]
der_amp = np.max(np.abs(np.diff(base))) - np.min(np.abs(np.diff(base)))
rows = {}
for factor in [0.5, 1.0, 2.0, 4.0]:
    n = int(len(seg)*factor)
    s2 = resample(seg, n)
    # the containing sequence is resampled by the same factor -> shape unchanged
    b2 = resample(base, int(len(base)*factor))
    d2 = np.diff(b2)
    da2 = np.max(np.abs(d2)) - np.min(np.abs(d2))
    rows[factor] = raw_features(s2, len(b2), da2)
ref = rows[1.0]
e1 = {}
for i, nm in enumerate(FEATNAMES):
    e1[nm] = {str(f): float(rows[f][i]) for f in rows}
    r = [rows[f][i] for f in [0.5,1.0,2.0,4.0]]
    ratio = (max(np.abs(r))+1e-12)/(min(np.abs(r))+1e-12)
    e1[nm]["max_min_ratio"] = float(ratio)
    print(f"  {nm:15s} " + "  ".join(f"{v:10.5f}" for v in r) + f"   ratio={ratio:8.2f}")
OUT["E1_resampling"] = e1
print(f"  segment length at factor 1.0: {len(seg)}")

# =====================================================================
# E2  y-axis / context: identical motif inside different global contexts
# =====================================================================
print("\n=== E2: identical motif, different global context ===")
t = np.linspace(0, 1, 200)
motif = 0.5 + 0.15*np.sin(2*np.pi*t)          # fixed absolute shape
def context(motif, tail_amp, tail_len=800):
    tail = np.linspace(motif[-1], motif[-1]+tail_amp, tail_len)
    return np.concatenate([motif, tail])
e2 = {}
for tail_amp in [0.1, 0.5, 2.0, 5.0]:
    full = context(motif, tail_amp)
    fulln = normalize_sequence(full)
    m = fulln[:200]                            # the same motif after global normalisation
    d = np.diff(fulln); da = np.max(np.abs(d)) - np.min(np.abs(d))
    f = raw_features(m, len(fulln), da)
    e2[str(tail_amp)] = {nm: float(v) for nm, v in zip(FEATNAMES, f)}
    print(f"  tail_amp={tail_amp:4.1f}  amp={f[5]:.4f} mean={f[4]:.4f} "
          f"sharp_inc={f[7]:.3f} sharp_dec={f[9]:.3f} const={f[11]:.3f}")
OUT["E2_context"] = e2

# =====================================================================
# E3  change point detector robustness
# =====================================================================
print("\n=== E3: change-point stability under small perturbations ===")
def hausdorff(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0: return 1.0
    d1 = max(np.min(np.abs(a[:, None]-b[None, :]), axis=1))
    d2 = max(np.min(np.abs(b[:, None]-a[None, :]), axis=1))
    return float(max(d1, d2)/n)

idx = rng.choice(len(seqs), 60, replace=False)
pert_names = ["noise_0.005", "noise_0.01", "resample_1.05", "shift_1pct", "yscale_0.98"]
stats = {p: {"dcount": [], "hd": []} for p in pert_names}
for k in idx:
    s = seqs[k]
    c0 = np.array(change_points_detection(s), float)
    variants = {
        "noise_0.005": normalize_sequence(s + rng.normal(0, 0.005, len(s))),
        "noise_0.01":  normalize_sequence(s + rng.normal(0, 0.01, len(s))),
        "resample_1.05": normalize_sequence(resample(resample(s, int(len(s)*1.05)), len(s))),
        "shift_1pct":  normalize_sequence(np.concatenate([s[10:], np.repeat(s[-1], 10)])),
        "yscale_0.98": normalize_sequence(s*0.98 + 0.01),
    }
    for p, sv in variants.items():
        try:
            c1 = np.array(change_points_detection(sv), float)
        except Exception:
            continue
        stats[p]["dcount"].append(abs(len(c1)-len(c0)))
        stats[p]["hd"].append(hausdorff(c0, c1, len(s)))
e3 = {}
for p in pert_names:
    dc, hd = np.array(stats[p]["dcount"]), np.array(stats[p]["hd"])
    e3[p] = {"mean_abs_count_change": float(dc.mean()), "pct_count_changed": float((dc > 0).mean()*100),
             "median_hausdorff_pct": float(np.median(hd)*100), "p90_hausdorff_pct": float(np.percentile(hd, 90)*100)}
    print(f"  {p:15s} |Δ#cp| mean={dc.mean():5.2f}  changed={100*(dc>0).mean():5.1f}%  "
          f"median Hausdorff={100*np.median(hd):5.2f}% of length  p90={100*np.percentile(hd,90):5.2f}%")
OUT["E3_cp_stability"] = e3

# number of change points distribution
ncp = [len(change_points_detection(seqs[i])) for i in idx]
OUT["E3_ncp"] = {"mean": float(np.mean(ncp)), "min": int(np.min(ncp)), "max": int(np.max(ncp))}
print(f"  #change points over 60 sequences: mean={np.mean(ncp):.1f} min={np.min(ncp)} max={np.max(ncp)}")

# ruptures comparison
try:
    import ruptures as rpt
    hd_r, dc_r = [], []
    for k in idx[:25]:
        s = seqs[k]
        algo = rpt.Pelt(model="l2", min_size=10, jump=5).fit(s.reshape(-1, 1))
        c0 = np.array(algo.predict(pen=0.05), float)
        sv = normalize_sequence(s + rng.normal(0, 0.01, len(s)))
        algo2 = rpt.Pelt(model="l2", min_size=10, jump=5).fit(sv.reshape(-1, 1))
        c1 = np.array(algo2.predict(pen=0.05), float)
        dc_r.append(abs(len(c1)-len(c0))); hd_r.append(hausdorff(c0, c1, len(s)))
    OUT["E3_ruptures_pelt"] = {"mean_abs_count_change": float(np.mean(dc_r)),
                               "pct_count_changed": float(np.mean(np.array(dc_r) > 0)*100),
                               "median_hausdorff_pct": float(np.median(hd_r)*100)}
    print(f"  [ruptures PELT l2] noise_0.01: |Δ#cp| mean={np.mean(dc_r):.2f}  "
          f"changed={100*np.mean(np.array(dc_r)>0):.1f}%  median Hausdorff={100*np.median(hd_r):.2f}%")
except Exception as e:
    print("  ruptures comparison failed:", e)

# =====================================================================
# E4  merge penalty: sign, magnitude, and the magic number 3
# =====================================================================
print("\n=== E4: merge penalty analysis ===")
from scipy.spatial.distance import cdist
FW = np.array([0.089638,0.155970,0.149875,0.081570,0.166547,0.060416,
               0.010039,0.084600,0.011294,0.083740,0.022768,0.083542])
FW = FW/FW.sum()

def feats_of(seq):
    cps = change_points_detection(seq)
    d = np.diff(seq); da = np.max(np.abs(d))-np.min(np.abs(d))
    F = np.array([extract_segment_features(seq[cps[i]:cps[i+1]], len(seq), da)
                  for i in range(len(cps)-1)])
    L = np.array([cps[i+1]-cps[i] for i in range(len(cps)-1)], float)
    return F*FW, L

neg, tot, corrs, pens = 0, 0, [], []
pairs = [(int(a), int(b)) for a, b in rng.choice(len(seqs), (40, 2), replace=True)]
for a, b in pairs:
    try:
        X, l1 = feats_of(seqs[a]); Y, l2 = feats_of(seqs[b])
    except Exception:
        continue
    if len(X) < 3 or len(Y) < 3: continue
    D = cdist(X, Y); lam1, lam2 = D.mean(), D.std()
    for i in range(len(X)-1):
        for w in [2, 3]:
            if i+w > len(X): continue
            c = features_correlation(X[i:i+w])
            if np.isnan(c): continue
            p = (lam1 - 3*lam2*c)
            corrs.append(float(c)); pens.append(float(p)); tot += 1
            if p < 0: neg += 1
corrs, pens = np.array(corrs), np.array(pens)
OUT["E4_penalty"] = {"n": int(tot), "pct_negative_penalty": float(100*neg/max(tot,1)),
                     "corr_mean": float(corrs.mean()), "corr_min": float(corrs.min()),
                     "corr_frac_negative": float((corrs < 0).mean()),
                     "penalty_mean": float(pens.mean()), "penalty_std": float(pens.std())}
print(f"  n={tot} merge candidates; penalty<0 (merging REWARDED) in {100*neg/max(tot,1):.1f}% of cases")
print(f"  correlation term: mean={corrs.mean():.3f} min={corrs.min():.3f} frac<0={100*(corrs<0).mean():.1f}%")
print(f"  penalty coefficient: mean={pens.mean():.4f} std={pens.std():.4f}")

# how many merge-window correlations are undefined (constant rows -> NaN)
nan_c = 0; tot_c = 0
for a, b in pairs[:20]:
    try:
        X, _ = feats_of(seqs[a])
    except Exception:
        continue
    for i in range(len(X)-1):
        c = features_correlation(X[i:i+2]); tot_c += 1
        if np.isnan(c): nan_c += 1
OUT["E4_nan_corr_pct"] = float(100*nan_c/max(tot_c,1))
print(f"  correlation undefined (NaN, constant trend profile): {100*nan_c/max(tot_c,1):.1f}% of windows")

# =====================================================================
# E5  merging z-scored features is algebraically inconsistent
# =====================================================================
print("\n=== E5: merge on standardised features ===")
means = np.array([1.29917091e-04,6.88773468e-05,2.15043790e-03,2.47959340e-01,
                  4.91384676e-01,2.41590926e-01,1.07280928e-01,3.82233768e-01,
                  8.75325723e-02,3.36648956e-01,8.52701912e-02,1.08314512e-01])
stds  = np.array([8.64548709e-05,2.54086189e-03,1.45989651e-03,2.42832550e-01,
                  2.83370148e-01,2.43137777e-01,7.82924368e-02,3.93206966e-01,
                  1.25110193e-01,3.85479883e-01,1.28169222e-01,1.30012737e-01])
# for an additive feature f: z(f1)+z(f2) vs z(f1+f2); bias = mean/std
bias = means/stds
e5 = {FEATNAMES[i]: float(bias[i]) for i in [3, 5, 6]}
OUT["E5_merge_bias_in_sd"] = e5
print("  summing z-scores instead of z-scoring the sum introduces a constant offset of mean/std:")
for i in [3, 5, 6]:
    print(f"    {FEATNAMES[i]:15s} offset = {bias[i]:.3f} standard deviations per extra merged segment")
# k-fold merge
for k in [2, 3, 4]:
    print(f"    merging {k} segments -> length feature offset = {(k-1)*bias[6]:.2f} sd")
OUT["E5_length_offset_by_k"] = {str(k): float((k-1)*bias[6]) for k in [2,3,4,5]}

# =====================================================================
# E6  weights enter the L2 distance quadratically
# =====================================================================
print("\n=== E6: effective weighting ===")
w = FW
eff = w**2/np.sum(w**2)
OUT["E6_nominal_vs_effective"] = {nm: {"nominal": float(a), "effective": float(b)}
                                  for nm, a, b in zip(FEATNAMES, w, eff)}
for nm, a, b in zip(FEATNAMES, w, eff):
    print(f"  {nm:15s} nominal={a:.4f}  effective(w^2 normalised)={b:.4f}  x{b/a:.2f}")

# =====================================================================
# E7  tuning objective is a coarse plateau
# =====================================================================
print("\n=== E7: Optuna objective granularity ===")
with open("results/human_tuning_results.json") as f:
    tr = json.load(f)
vals = np.array([t["value"] for t in tr["all_trials"]])
u, cnt = np.unique(np.round(vals, 6), return_counts=True)
OUT["E7_tuning"] = {"n_trials": int(len(vals)), "n_distinct_values": int(len(u)),
                    "best": float(vals.max()), "n_at_best": int((vals == vals.max()).sum()),
                    "n_params": 12, "n_anchors": 17,
                    "baseline_dtw": tr["before_tuning"]["ap"]["dtw"],
                    "baseline_lcss": tr["before_tuning"]["ap"]["lcss"],
                    "ours_before": tr["before_tuning"]["ap"]["ours"],
                    "ours_after": tr["after_tuning"]["ap"]["ours"]}
print(f"  {len(vals)} trials -> only {len(u)} distinct objective values (step = 1/(3*17) = {1/51:.4f})")
print(f"  trials tied at the best value: {(vals == vals.max()).sum()}")
print(f"  ours {tr['before_tuning']['ap']['ours']} -> {tr['after_tuning']['ap']['ours']} (TRAIN), "
      f"dtw {tr['before_tuning']['ap']['dtw']}, lcss {tr['before_tuning']['ap']['lcss']}")

with open(os.path.join(HERE, "diag.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print("\nSaved diag.json")
