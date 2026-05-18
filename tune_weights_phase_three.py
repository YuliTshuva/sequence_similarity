"""
Yuli Tshuva
Evaluate sequence similarity methods on a human-ranked dataset,
then tune FEATURE_WEIGHTS to better match human judgments.

Dataset format:
  data/dataset.csv         — anchor, candidate, human_rank (1=most similar, 9=most different)
  data/name_data/<name>.csv — one sequence per file

Usage:
    python tune_weights_human.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from os.path import join

from seq_sim_alg import seq_distance, ALPHA, FEATURE_WEIGHTS
from compare_baselines import dtw_distance, lcss_distance

# ── config ────────────────────────────────────────────────────────────────────

DATASET_PATH   = join("data", "dataset.csv")
SEQ_DIR        = join("data", "name_data")
RESULTS_PATH   = join("results", "human_tuning_results.json")
TOP_K          = 3      # top-k human ranks treated as positives
N_TRIALS       = 50
SEED           = 42

FEATURE_NAMES = [
    "curvature", "mean_diff", "mean_abs_diff", "mean_value",
    "amplitude", "length", "sharp_increasing", "light_increasing",
    "sharp_decreasing", "light_decreasing", "constant",
]

# ── data loading ──────────────────────────────────────────────────────────────

def load_sequence(name):
    """Load a single sequence from data/name_data/<name>.csv."""
    path = join(SEQ_DIR, f"{name}.csv")
    seq = pd.read_csv(path, header=None).values.flatten()
    # skip header if present
    if str(seq[0]).isalpha():
        seq = seq[1:].astype(float)
    # Normalize to [0, 1]
    mn, mx = seq.min(), seq.max()
    if mx - mn > 1e-6:
        seq = (seq - mn) / (mx - mn)
    return seq


def load_dataset():
    """
    Load dataset.csv and return a list of anchor groups:
    [
      {
        "anchor": anchor_name,
        "candidates": [
          {"name": name, "human_rank": rank, "seq": array},
          ...
        ],
        "anchor_seq": array
      },
      ...
    ]
    """
    df = pd.read_csv(DATASET_PATH, header=None, names=["anchor", "candidate", "human_rank"])

    # Cache sequences to avoid re-reading the same file
    seq_cache = {}
    def get_seq(name):
        if name not in seq_cache:
            seq_cache[name] = load_sequence(name)
        return seq_cache[name]

    groups = []
    for anchor_name, group in df.groupby("anchor", sort=False):
        anchor_seq = get_seq(anchor_name)
        candidates = []
        for _, row in group.iterrows():
            candidates.append({
                "name":       row["candidate"],
                "human_rank": int(row["human_rank"]),
                "seq":        get_seq(row["candidate"]),
            })
        groups.append({
            "anchor":     anchor_name,
            "anchor_seq": anchor_seq,
            "candidates": candidates,
        })

    print(f"Loaded {len(groups)} anchors, "
          f"{sum(len(g['candidates']) for g in groups)} total pairs.\n")
    return groups


# ── evaluation ────────────────────────────────────────────────────────────────



def average_precision_vs_human(method_scores, candidates, top_k=TOP_K):
    """
    Precision@top_k: what fraction of the human top-k did our method
    also place in its own top-k.
    method_scores: list of distances (lower = more similar).
    """
    # Indices our method ranks in top_k (smallest distances)
    our_top_k = set(
        sorted(range(len(method_scores)), key=lambda i: method_scores[i])[:top_k]
    )
    # Indices humans ranked in top_k
    human_top_k = set(
        i for i, c in enumerate(candidates) if c["human_rank"] <= top_k
    )
    return len(our_top_k & human_top_k) / top_k


def evaluate_all_methods(groups, feature_weights=None):
    """
    For each anchor group, compute distances with all methods and score vs humans.
    Returns a DataFrame with Precision@TOP_K per anchor per method.
    """
    if feature_weights is None:
        feature_weights = FEATURE_WEIGHTS

    methods = ["ours", "dtw", "lcss"]
    records = []

    for g in groups:
        anchor    = g["anchor_seq"]
        cands     = g["candidates"]

        scores = {m: [] for m in methods}
        for c in cands:
            cseq = c["seq"]
            dist_ours, _ = seq_distance(anchor, cseq, alpha=ALPHA,
                                        feature_weights=feature_weights)
            scores["ours"].append(dist_ours)
            scores["dtw"].append(dtw_distance(anchor, cseq))
            scores["lcss"].append(lcss_distance(anchor, cseq))

        for m in methods:
            records.append({
                "anchor": g["anchor"],
                "method": m,
                "ap": average_precision_vs_human(scores[m], cands),
            })

    return pd.DataFrame(records)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_evaluation(results_df, save_path=None):
    """
    Two subplots:
      Left  — Precision@TOP_K per anchor per method (grouped bars)
      Right — Mean Precision@TOP_K across all anchors (summary bars)
    """
    methods       = ["ours", "dtw", "lcss"]
    method_colors = {"ours": "#4C9BE8", "dtw": "#E8834C", "lcss": "#6DBE6D"}
    anchors       = results_df["anchor"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Evaluation vs Human Rankings", fontsize=13, fontweight="bold")

    # ── Left: Precision@TOP_K per anchor ────────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(anchors))
    w  = 0.25
    for bi, m in enumerate(methods):
        vals = [
            results_df[(results_df["anchor"] == a) & (results_df["method"] == m)]["ap"].values[0]
            for a in anchors
        ]
        ax.bar(x + (bi - 1) * w, vals, w, label=m, color=method_colors[m], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(anchors, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"Precision@{TOP_K}  (higher = more human-like)")
    ax.set_title(f"Per-Anchor Precision@{TOP_K} vs Human Ranking")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Right: mean Precision@TOP_K ─────────────────────────────────────────────
    ax2   = axes[1]
    x2    = np.arange(1)
    width = 0.2
    for bi, m in enumerate(methods):
        sub  = results_df[results_df["method"] == m]
        vals = [sub["ap"].mean()]
        bars = ax2.bar(x2 + (bi - 1) * width, vals, width,
                       label=m, color=method_colors[m], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"Mean Precision@{TOP_K}"], fontsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.set_title(f"Mean Precision@{TOP_K} Across All Anchors")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Evaluation plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── weight tuning ─────────────────────────────────────────────────────────────

def objective(trial, groups, init_weights):
    """Optuna objective: maximise mean AP@TOP_K vs human rankings."""
    raw = np.array([
        w + trial.suggest_float(f"w_{n}", -w * 0.5, w * 0.5)
        for w, n in zip(init_weights, FEATURE_NAMES)
    ])
    raw = np.clip(raw, 1e-6, None)
    weights = raw / raw.sum()

    aps = []
    for g in groups:
        anchor = g["anchor_seq"]
        cands  = g["candidates"]
        scores = []
        for c in cands:
            dist, _ = seq_distance(anchor, c["seq"], alpha=ALPHA,
                                   feature_weights=weights)
            scores.append(dist)
        aps.append(average_precision_vs_human(scores, cands))

    mean_ap = float(np.mean(aps))
    print(f"  trial {trial.number:3d}: mean AP = {mean_ap:.4f}  "
          f"weights = {np.round(weights, 3)}")
    return mean_ap


def tune_weights(groups, init_weights, n_trials=N_TRIALS):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="human_weight_tuning_ap",
    )
    study.optimize(
        lambda trial: objective(trial, groups, init_weights),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best    = study.best_trial
    raw     = np.array([
        init_weights[i] + best.params[f"w_{n}"]
        for i, n in enumerate(FEATURE_NAMES)
    ])
    raw     = np.clip(raw, 1e-6, None)
    best_w  = raw / raw.sum()

    print("\n" + "=" * 55)
    print(f"Best mean AP@{TOP_K}: {best.value:.4f}")
    print("\nBest FEATURE_WEIGHTS (copy into seq_sim_alg.py):")
    print("FEATURE_WEIGHTS = np.array([")
    for name, w in zip(FEATURE_NAMES, best_w):
        print(f"    {w:.6f},  # {name}")
    print("])")

    return best_w, study


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("Loading dataset...")
    groups = load_dataset()

    # ── 2. Evaluate current weights ───────────────────────────────────────────
    print("Evaluating current weights vs baselines...")
    results_before = evaluate_all_methods(groups)

    print("\n── Results BEFORE tuning ──")
    print(results_before.groupby("method")[["ap"]].mean().round(3))

    plot_evaluation(results_before,
                    save_path=join("results", "eval_before_tuning.png"))

    # ── 3. Tune weights ───────────────────────────────────────────────────────
    print("\nTuning weights against human rankings...")
    init_weights = FEATURE_WEIGHTS / FEATURE_WEIGHTS.sum()
    best_weights, study = tune_weights(groups, init_weights)

    # ── 4. Evaluate tuned weights ─────────────────────────────────────────────
    print("\nEvaluating tuned weights...")
    results_after = evaluate_all_methods(groups, feature_weights=best_weights)

    print("\n── Results AFTER tuning ──")
    print(results_after.groupby("method")[["ap"]].mean().round(3))

    plot_evaluation(results_after,
                    save_path=join("results", "eval_after_tuning.png"))

    # ── 5. Save results ───────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    output = {
        "best_ap":      study.best_value,
        "best_weights": {n: float(w) for n, w in zip(FEATURE_NAMES, best_weights)},
        "before_tuning": results_before.groupby("method")[["ap"]]
                         .mean().round(4).to_dict(),
        "after_tuning":  results_after.groupby("method")[["ap"]]
                         .mean().round(4).to_dict(),
        "all_trials": [
            {"number": t.number, "value": t.value}
            for t in study.trials
        ],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()