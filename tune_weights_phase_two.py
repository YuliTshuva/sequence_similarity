"""
Yuli Tshuva
Phase 2 weight tuning: optimize FEATURE_WEIGHTS so that our method
correctly identifies structurally similar sequences under transformations
that preserve human-perceived similarity but break DTW.

Three transformations are used to generate positive (similar) candidates:
  1. Scale perturbation  — one or more segments scaled vertically
  2. Amplitude shift     — one or more segments shifted vertically
  3. Speed variation     — one or more segments stretched/compressed in time

Negative (dissimilar) candidates are structurally different sequences
generated from scratch. The objective rewards our method for ranking
positives above negatives, and explicitly penalizes DTW failure cases
where DTW ranks a positive low but our method ranks it high.

Starts from the best weights found in phase one.

Usage:
    python tune_weights_phase_two.py --phase1_results results/tuned_weights.json
"""

import json
import os
import argparse
import numpy as np
import optuna
from scipy.stats import kendalltau

from seq_sim_alg import seq_distance, ALPHA, FEATURE_WEIGHTS
from compare_baselines import dtw_distance

# ── config ────────────────────────────────────────────────────────────────────

N_ANCHORS          = 5     # synthetic anchors to generate per trial evaluation
N_POSITIVES        = 6     # similar candidates per anchor (2 per transformation)
N_NEGATIVES        = 14    # dissimilar candidates per anchor
SEQ_LEN            = 200   # length of generated sequences
N_SEGMENTS         = 3     # number of hills/segments per sequence
N_TRIALS           = 50    # Optuna trials
SEED               = 42
RESULTS_PATH       = "results/tuned_weights_phase2.json"

FEATURE_NAMES = [
    "curvature", "mean_diff", "mean_abs_diff", "mean_value",
    "amplitude", "length", "sharp_increasing", "light_increasing",
    "sharp_decreasing", "light_decreasing", "constant",
]

# ── synthetic sequence generation ─────────────────────────────────────────────

def make_hill(length, peak_pos=0.5, peak_height=1.0):
    """Single smooth hill (sine arch) of given length, peak position and height."""
    x = np.linspace(0, np.pi, length)
    hill = np.sin(x) * peak_height
    # Shift peak position by warping x
    shift = int((peak_pos - 0.5) * length * 0.6)
    hill = np.roll(hill, shift)
    hill[:max(0, shift)] = 0
    hill[min(length, length + shift):] = 0
    return np.clip(hill, 0, None)


def make_sequence(n_segments=N_SEGMENTS, seq_len=SEQ_LEN, rng=None):
    """
    Generate a sequence of n_segments hills concatenated and normalized to [0,1].
    Each segment gets a random length (summing to seq_len), random peak height,
    and random peak position within the segment.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    # Random segment lengths summing to seq_len
    breaks = np.sort(rng.integers(20, seq_len - 20 * n_segments, size=n_segments - 1))
    lengths = np.diff([0] + list(breaks) + [seq_len])
    lengths = np.maximum(lengths, 20)
    lengths[-1] = seq_len - lengths[:-1].sum()

    parts = []
    for length in lengths:
        peak_pos    = rng.uniform(0.3, 0.7)
        peak_height = rng.uniform(0.3, 1.0)
        parts.append(make_hill(int(length), peak_pos, peak_height))

    seq = np.concatenate(parts)
    seq = seq[:seq_len]

    # Normalize to [0, 1]
    mn, mx = seq.min(), seq.max()
    if mx - mn > 1e-6:
        seq = (seq - mn) / (mx - mn)
    return seq


# ── transformations ───────────────────────────────────────────────────────────

def transform_scale(seq, n_segments, rng, strength=0.7, n_affected=2):
    """
    Scale perturbation: scale n_affected random segments amplitude by a random
    factor in [1-strength, 1+strength], then renormalize to [0,1].
    Preserves shape; breaks DTW because values change.
    """
    seg_len = len(seq) // n_segments
    seg_indices = rng.choice(n_segments, size=min(n_affected, n_segments), replace=False)
    out = seq.copy()
    for seg_idx in seg_indices:
        start, end = seg_idx * seg_len, (seg_idx + 1) * seg_len
        factor = rng.uniform(1 - strength, 1 + strength)
        out[start:end] *= factor
    mn, mx = out.min(), out.max()
    if mx - mn > 1e-6:
        out = (out - mn) / (mx - mn)
    return out


def transform_amplitude_shift(seq, n_segments, rng, strength=0.4, n_affected=2):
    """
    Amplitude shift: shift n_affected random segments vertically by a random
    offset, then clip and renormalize. Preserves shape but moves values up/down.
    """
    seg_len = len(seq) // n_segments
    seg_indices = rng.choice(n_segments, size=min(n_affected, n_segments), replace=False)
    out = seq.copy()
    for seg_idx in seg_indices:
        start, end = seg_idx * seg_len, (seg_idx + 1) * seg_len
        shift = rng.uniform(-strength, strength)
        out[start:end] += shift
    out = np.clip(out, 0, 1)
    mn, mx = out.min(), out.max()
    if mx - mn > 1e-6:
        out = (out - mn) / (mx - mn)
    return out


def transform_speed(seq, n_segments, rng, strength=0.6, n_affected=2):
    """
    Speed variation: stretch or compress n_affected random segments in time,
    keeping total length fixed by resampling the full sequence at the end.
    Preserves shape but changes local timing.
    """
    seg_len = len(seq) // n_segments
    seg_indices = set(
        rng.choice(n_segments - 1, size=min(n_affected, n_segments - 1), replace=False)
    )
    parts = []
    for seg_idx in range(n_segments):
        start = seg_idx * seg_len
        end   = (seg_idx + 1) * seg_len if seg_idx < n_segments - 1 else len(seq)
        seg   = seq[start:end]
        if seg_idx in seg_indices:
            factor  = rng.uniform(1 - strength, 1 + strength)
            new_len = max(10, int(len(seg) * factor))
            seg = np.interp(
                np.linspace(0, len(seg) - 1, new_len),
                np.arange(len(seg)), seg
            )
        parts.append(seg)
    combined = np.concatenate(parts)
    out = np.interp(
        np.linspace(0, len(combined) - 1, len(seq)),
        np.arange(len(combined)), combined
    )
    mn, mx = out.min(), out.max()
    if mx - mn > 1e-6:
        out = (out - mn) / (mx - mn)
    return out


TRANSFORMATIONS = [transform_scale, transform_amplitude_shift, transform_speed]
TRANSFORM_NAMES = ["scale", "amplitude_shift", "speed"]


# ── synthetic benchmark construction ─────────────────────────────────────────

def build_benchmark(rng):
    """
    For a single anchor, build:
      - positives: 2 variants per transformation (6 total)
      - negatives: N_NEGATIVES structurally different sequences

    Returns anchor, list of (seq, label, transform_name) candidates.
    label: 1 = similar (positive), 0 = dissimilar (negative)
    """
    anchor = make_sequence(rng=rng)
    candidates = []

    # Positives: apply each transformation twice with different random states
    for t_fn, t_name in zip(TRANSFORMATIONS, TRANSFORM_NAMES):
        for _ in range(N_POSITIVES // len(TRANSFORMATIONS)):
            cseq = t_fn(anchor, N_SEGMENTS, rng)
            candidates.append((cseq, 1, t_name))

    # Negatives: independently generated sequences (different structure)
    for _ in range(N_NEGATIVES):
        # Randomize n_segments too so structure genuinely differs
        n_seg = rng.integers(2, N_SEGMENTS + 3)
        cseq = make_sequence(n_segments=int(n_seg), rng=rng)
        candidates.append((cseq, 0, "negative"))

    rng.shuffle(candidates)
    return anchor, candidates


# ── scoring ───────────────────────────────────────────────────────────────────

def average_precision(our_scores, candidates):
    """
    Average precision @ N_POSITIVES: reward our method for ranking
    positives (label=1) above negatives (label=0).
    """
    sorted_cands = sorted(
        zip(our_scores, [c[1] for c in candidates]),
        key=lambda x: x[0]  # lower distance = more similar = better rank
    )
    n_pos    = sum(c[1] for c in candidates)
    hits     = 0
    ap       = 0.0
    for rank, (_, label) in enumerate(sorted_cands, start=1):
        if label == 1:
            hits += 1
            ap   += hits / rank
    return ap / n_pos if n_pos > 0 else 0.0


def dtw_failure_bonus(our_scores, candidates, anchor):
    """
    Extra reward for cases where DTW ranks a positive LOW (rank > N_POSITIVES)
    but our method ranks it HIGH (rank <= N_POSITIVES).
    This directly targets the cases DTW fails on.
    """
    dtw_scores = [dtw_distance(anchor, c[0]) for c in candidates]
    n = len(candidates)

    our_order = sorted(range(n), key=lambda i: our_scores[i])
    dtw_order = sorted(range(n), key=lambda i: dtw_scores[i])

    our_rank  = {i: r for r, i in enumerate(our_order)}
    dtw_rank  = {i: r for r, i in enumerate(dtw_order)}

    bonus = 0.0
    for i, (_, label, _) in enumerate(candidates):
        if label == 1:
            dtw_failed  = dtw_rank[i] >= N_POSITIVES   # DTW ranked it low
            we_succeed  = our_rank[i] < N_POSITIVES     # we ranked it high
            if dtw_failed and we_succeed:
                bonus += 1.0
    return bonus / max(N_POSITIVES, 1)


# ── objective ─────────────────────────────────────────────────────────────────

def objective(trial, phase1_weights, rng_seed):
    # Perturb around phase 1 weights (narrower search range than phase 1)
    raw_weights = np.array([
        w + trial.suggest_float(f"w_{name}", -w * 0.5, w * 0.5)
        for w, name in zip(phase1_weights, FEATURE_NAMES)
    ])
    raw_weights = np.clip(raw_weights, 1e-6, None)
    weights = raw_weights / raw_weights.sum()

    rng = np.random.default_rng(rng_seed)
    ap_scores    = []
    bonus_scores = []

    for _ in range(N_ANCHORS):
        anchor, candidates = build_benchmark(rng)
        seqs = [c[0] for c in candidates]

        our_scores = []
        for cseq in seqs:
            dist, _ = seq_distance(anchor, cseq, alpha=ALPHA, feature_weights=weights)
            our_scores.append(dist)

        ap_scores.append(average_precision(our_scores, candidates))
        bonus_scores.append(dtw_failure_bonus(our_scores, candidates, anchor))

    # Combined objective: AP rewards correct ranking; bonus rewards DTW failures
    mean_ap    = float(np.mean(ap_scores))
    mean_bonus = float(np.mean(bonus_scores))
    score      = mean_ap + 0.5 * mean_bonus

    print(f"  trial {trial.number:3d}: AP={mean_ap:.3f}  DTW-failure bonus={mean_bonus:.3f}  "
          f"score={score:.3f}  weights={np.round(weights, 3)}")
    return score


# ── benchmark visualisation ───────────────────────────────────────────────────

def plot_benchmark_examples(rng_seed=SEED, save_path=None):
    """
    3x3 grid:
      Row 0 — 3 anchor sequences
      Row 1 — one positive example per anchor (randomly chosen transformation)
      Row 2 — one negative example per anchor
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(rng_seed)
    fig, axes = plt.subplots(3, 3, figsize=(13, 7))

    row_labels = ["Anchor", "Positive (similar)", "Negative (dissimilar)"]
    row_colors = ["#4C9BE8", "#6DBE6D", "#E8834C"]

    for col in range(3):
        anchor, candidates = build_benchmark(rng)

        # Pick one positive and one negative at random
        positives = [(seq, t) for seq, label, t in candidates if label == 1]
        negatives = [(seq, t) for seq, label, t in candidates if label == 0]
        pos_seq, pos_transform = positives[rng.integers(len(positives))]
        neg_seq, _             = negatives[rng.integers(len(negatives))]

        rows = [anchor, pos_seq, neg_seq]

        for row, (seq, color, row_label) in enumerate(zip(rows, row_colors, row_labels)):
            ax = axes[row][col]
            ax.plot(seq, color=color, linewidth=1.8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(alpha=0.2)

            # Column title on top row
            if row == 0:
                ax.set_title(f"Anchor {col + 1}", fontsize=10, fontweight="bold")

            # Transformation label on positive row
            if row == 1:
                ax.set_xlabel(f"transform: {pos_transform}", fontsize=8, color="gray")

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight="bold", color=color)

    fig.suptitle(
        "Synthetic benchmark examples\n"
        "Positives share the anchor's structure under transformation; "
        "negatives are structurally different",
        fontsize=10, y=1.02
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Benchmark plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def tune_weights_phase2(phase1_results_path: str = "results/tuned_weights.json"):
    # Load phase 1 best weights as starting point
    with open(phase1_results_path) as f:
        phase1 = json.load(f)
    phase1_weights = np.array(list(phase1["best_weights"].values()))
    phase1_weights = phase1_weights / phase1_weights.sum()

    print("Phase 1 weights loaded:")
    for name, w in zip(FEATURE_NAMES, phase1_weights):
        print(f"  {name:<22} {w:.4f}")
    print()

    # Plot benchmark examples before tuning starts
    print("Plotting benchmark examples...")
    plot_benchmark_examples(rng_seed=SEED, save_path="results/benchmark_examples.png")

    rng_seed = SEED
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="feature_weight_tuning_phase2",
    )

    study.optimize(
        lambda trial: objective(trial, phase1_weights, rng_seed),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    # ── results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    raw_best = np.array([
        phase1_weights[i] + best.params[f"w_{n}"]
        for i, n in enumerate(FEATURE_NAMES)
    ])
    raw_best = np.clip(raw_best, 1e-6, None)
    best_weights = raw_best / raw_best.sum()

    print("\n" + "=" * 55)
    print(f"Best score: {best.value:.4f}")
    print("\nBest FEATURE_WEIGHTS (copy into seq_sim_alg.py):")
    print("FEATURE_WEIGHTS = np.array([")
    for name, w in zip(FEATURE_NAMES, best_weights):
        print(f"    {w:.6f},  # {name}")
    print("])")

    os.makedirs("results", exist_ok=True)
    output = {
        "best_score": best.value,
        "best_weights": {n: float(w) for n, w in zip(FEATURE_NAMES, best_weights)},
        "phase1_weights": {n: float(w) for n, w in zip(FEATURE_NAMES, phase1_weights)},
        "transformations": TRANSFORM_NAMES,
        "all_trials": [
            {
                "number": t.number,
                "value":  t.value,
            }
            for t in study.trials
        ],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")
    return best_weights


if __name__ == "__main__":
    tune_weights_phase2("results/tuned_weights.json")
