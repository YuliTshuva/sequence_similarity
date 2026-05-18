"""
Yuli Tshuva
Tune FEATURE_WEIGHTS for seq_distance so that its top-k ranking
agrees with DTW and LCSS rankings, using Optuna (Bayesian optimisation).

Install once:
    pip install optuna

Usage:
    python tune_weights.py
"""

import json
import os
import random
import numpy as np
import optuna
from scipy.stats import kendalltau
from os.path import join

from mine_data import load_sequences
from seq_sim_alg import seq_distance, ALPHA
from compare_baselines import dtw_distance, lcss_distance

# ── config ────────────────────────────────────────────────────────────────────

DATA_PATH      = join("data", "stock_sequences.npz")
META_PATH      = join("data", "stock_sequences_meta.json")
N_CANDIDATES   = 100        # number of candidates to rank against the anchor
INTERP_LEN     = 1000      # match main() in seq_sim_alg.py
TOP_K          = 20        # evaluate agreement on top-k ranking
N_TRIALS       = 100        # Optuna trials (each trial = one weight config)
SEED           = 42
ANCHOR_IDX     = None      # set to an int to fix the anchor, or None to pick randomly
RESULTS_PATH   = "results/tuned_weights.json"

# Feature names — must match order in extract_node_features
FEATURE_NAMES = [
    "curvature", "mean_diff", "mean_abs_diff", "mean_value",
    "amplitude", "length", "sharp_increasing", "light_increasing",
    "sharp_decreasing", "light_decreasing", "constant",
]
N_FEATURES = len(FEATURE_NAMES)

# ── data loading ──────────────────────────────────────────────────────────────

def load_tuning_sequences():
    random.seed(SEED)
    seqs, meta = load_sequences(DATA_PATH, META_PATH)
    seqs = [
        np.interp(np.linspace(0, len(s) - 1, INTERP_LEN), np.arange(len(s)), s)
        for s in seqs
    ]
    all_ids = list(range(len(seqs)))

    anchor_idx = ANCHOR_IDX if ANCHOR_IDX is not None else random.choice(all_ids)
    candidate_ids = random.sample([i for i in all_ids if i != anchor_idx], N_CANDIDATES)

    anchor_seq     = seqs[anchor_idx]
    candidate_seqs = [seqs[i] for i in candidate_ids]

    print(f"Anchor: index {anchor_idx}")
    print(f"Candidates: {candidate_ids}\n")
    return anchor_seq, candidate_seqs, candidate_ids


# ── baseline rankings (precomputed once, they don't change) ───────────────────

def precompute_baseline_rankings(anchor, candidates, candidate_ids):
    """
    Rank all candidates against the anchor by DTW and LCSS.
    Returns {"dtw": [sorted ids], "lcss": [sorted ids]}
    """
    dtw_scores, lcss_scores = {}, {}
    print("Pre-computing baseline rankings (done once)...")
    for cid, cseq in zip(candidate_ids, candidates):
        dtw_scores[cid]  = dtw_distance(anchor, cseq)
        lcss_scores[cid] = lcss_distance(anchor, cseq)
    rankings = {
        "dtw":  sorted(dtw_scores,  key=dtw_scores.__getitem__),
        "lcss": sorted(lcss_scores, key=lcss_scores.__getitem__),
    }
    print(f"  DTW  top-5: {rankings['dtw'][:5]}")
    print(f"  LCSS top-5: {rankings['lcss'][:5]}\n")
    return rankings


# ── objective ─────────────────────────────────────────────────────────────────

def top_k_kendall_tau(our_ranking, baseline_ranking, k):
    """
    Kendall's tau between our top-k and baseline top-k.
    Considers only the union of both top-k sets for a fair comparison.
    """
    union = list(dict.fromkeys(our_ranking[:k] + baseline_ranking[:k]))
    our_pos  = {cid: our_ranking.index(cid)  if cid in our_ranking  else len(our_ranking)  for cid in union}
    base_pos = {cid: baseline_ranking.index(cid) if cid in baseline_ranking else len(baseline_ranking) for cid in union}
    our_vec  = [our_pos[c]  for c in union]
    base_vec = [base_pos[c] for c in union]
    if len(set(our_vec)) < 2 or len(set(base_vec)) < 2:
        return 0.0
    tau, _ = kendalltau(our_vec, base_vec)
    return float(tau) if not np.isnan(tau) else 0.0


def objective(trial, anchor, candidates, candidate_ids, baseline_rankings):
    # Sample weights in log-space so all are positive
    raw_weights = np.array([
        trial.suggest_float(f"w_{name}", 1e-2, 20.0, log=True)
        for name in FEATURE_NAMES
    ])
    weights = raw_weights / raw_weights.sum()

    # Compute our distances from anchor to all candidates
    our_scores = {}
    for cid, cseq in zip(candidate_ids, candidates):
        dist, _ = seq_distance(anchor, cseq, alpha=ALPHA, feature_weights=weights)
        our_scores[cid] = dist

    our_ranking = sorted(our_scores, key=our_scores.__getitem__)

    # Average tau across DTW and LCSS
    tau_scores = [
        top_k_kendall_tau(our_ranking, baseline_rankings[b], k=TOP_K)
        for b in ["dtw", "lcss"]
    ]
    mean_tau = float(np.mean(tau_scores))
    print(f"  trial {trial.number:3d}: τ = {mean_tau:.4f}  "
          f"our top-5 = {our_ranking[:5]}  weights = {np.round(weights, 3)}")
    return mean_tau


# ── main ──────────────────────────────────────────────────────────────────────

def tune_weights():
    anchor, candidates, candidate_ids = load_tuning_sequences()
    baseline_rankings = precompute_baseline_rankings(anchor, candidates, candidate_ids)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="feature_weight_tuning",
    )

    study.optimize(
        lambda trial: objective(trial, anchor, candidates, candidate_ids, baseline_rankings),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    # ── results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    raw_best = np.array([best.params[f"w_{n}"] for n in FEATURE_NAMES])
    best_weights = raw_best / raw_best.sum()

    print("\n" + "=" * 55)
    print(f"Best mean Kendall's τ: {best.value:.4f}")
    print("\nBest FEATURE_WEIGHTS (normalised, copy into seq_sim_alg.py):")
    print("FEATURE_WEIGHTS = np.array([")
    for name, w in zip(FEATURE_NAMES, best_weights):
        print(f"    {w:.6f},  # {name}")
    print("])")

    os.makedirs("results", exist_ok=True)
    output = {
        "best_tau": best.value,
        "best_weights": {n: float(w) for n, w in zip(FEATURE_NAMES, best_weights)},
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "weights": {
                    n: float(t.params[f"w_{n}"] / sum(t.params[f"w_{n2}"] for n2 in FEATURE_NAMES))
                    for n in FEATURE_NAMES
                },
            }
            for t in study.trials
        ],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")

    return best_weights


if __name__ == "__main__":
    tune_weights()