import json
import random
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from mine_data import load_sequences
from compare_baselines import all_distances
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from tqdm.auto import tqdm

# ── helpers ──────────────────────────────────────────────────────────────────

METHODS = ["euclidean", "dtw", "dtw_sakoe_chiba", "lcss", "pearson", "ours"]
BASELINES = [m for m in METHODS if m != "ours"]


def rank_candidates(anchor, candidates, ids):
    """
    For a single anchor, compute distances to every candidate with all methods
    and return a DataFrame with one row per candidate.
    """
    rows = []
    for cid, seq in zip(ids, candidates):
        d = all_distances(anchor, seq)
        d["candidate_id"] = cid
        rows.append(d)
    df = pd.DataFrame(rows).set_index("candidate_id")

    # Rank: rank 1 = most similar (smallest distance).
    # None values (e.g. Euclidean on unequal lengths) are ranked last.
    for m in METHODS:
        df[f"rank_{m}"] = df[m].rank(method="min", na_option="bottom")

    return df


def disagreement_score(df, baseline: str) -> float:
    """
    Kendall's tau between ours and a baseline ranking.
    Returns 1 - |tau| so that 1.0 = complete disagreement, 0.0 = perfect agreement.
    """
    ours = df["rank_ours"].values
    base = df[f"rank_{baseline}"].values
    mask = ~(np.isnan(ours) | np.isnan(base))
    if mask.sum() < 2:
        return 0.0
    tau, _ = kendalltau(ours[mask], base[mask])
    return float(1.0 - abs(tau))


def aggregate_disagreement(df) -> float:
    """Mean disagreement across all baselines."""
    return float(np.mean([disagreement_score(df, b) for b in BASELINES]))


# ── main loop ─────────────────────────────────────────────────────────────────

def run(
        data_path: str = "data/stock_sequences.npz",
        meta_path: str = "data/stock_sequences_meta.json",
        n_iterations: int = 50,
        n_candidates: int = 100,
        top_k: int = 10,
        disagreement_threshold: float = 0.4,
        seed: int = 42,
        results_path: str = "results/disagreements.json",
):
    """
    Parameters
    ----------
    n_iterations          : number of (anchor, candidates) samples to evaluate
    n_candidates          : candidates sampled per anchor
    top_k                 : number of top-disagreement iterations saved in detail
    disagreement_threshold: minimum mean disagreement score to flag an iteration
    seed                  : random seed for reproducibility
    results_path          : where to write the JSON output
    """
    random.seed(seed)
    np.random.seed(seed)

    seqs, meta = load_sequences(data_path, meta_path)
    # Increase resolution of all sequences to 1000 (if not already)
    seqs = [np.interp(np.linspace(0, len(s) - 1, 1000), np.arange(len(s)), s) for s in seqs]
    all_ids = list(range(len(seqs)))

    iteration_summaries = []

    for it in range(n_iterations):
        # Sample anchor + candidates without replacement
        anchor_id = random.choice(all_ids)
        pool = [i for i in all_ids if i != anchor_id]
        candidate_ids = random.sample(pool, n_candidates)

        anchor_seq = np.array(seqs[anchor_id], dtype=float)
        candidate_seqs = [np.array(seqs[i], dtype=float) for i in candidate_ids]

        df = rank_candidates(anchor_seq, candidate_seqs, candidate_ids)

        # Per-baseline disagreement
        per_baseline = {b: disagreement_score(df, b) for b in BASELINES}
        mean_disagreement = float(np.mean(list(per_baseline.values())))

        summary = {
            "iteration": it,
            "anchor_id": anchor_id,
            "anchor_meta": meta[anchor_id] if meta else None,
            "mean_disagreement": round(mean_disagreement, 4),
            "per_baseline_disagreement": {k: round(v, 4) for k, v in per_baseline.items()},
            "flagged": mean_disagreement >= disagreement_threshold,
        }

        # For flagged iterations, store top-5 rank flips
        if summary["flagged"]:
            summary["rank_flips"] = _top_rank_flips(df, n=5)

        iteration_summaries.append(summary)
        print(
            f"[{it + 1:3d}/{n_iterations}] anchor={anchor_id:4d} "
            f"disagreement={mean_disagreement:.3f}"
            + (" *** FLAGGED" if summary["flagged"] else "")
        )

    # Sort by disagreement descending and keep top_k details
    iteration_summaries.sort(key=lambda x: x["mean_disagreement"], reverse=True)
    flagged = [s for s in iteration_summaries if s["flagged"]]

    output = {
        "config": {
            "n_iterations": n_iterations,
            "n_candidates": n_candidates,
            "disagreement_threshold": disagreement_threshold,
            "seed": seed,
        },
        "summary": {
            "total_flagged": len(flagged),
            "mean_disagreement_overall": round(
                float(np.mean([s["mean_disagreement"] for s in iteration_summaries])), 4
            ),
        },
        "top_disagreements": iteration_summaries[:top_k],
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(flagged)}/{n_iterations} iterations flagged.")
    print(f"Results written to {results_path}")
    return output


def _top_rank_flips(df: pd.DataFrame, n: int = 5) -> list:
    """
    Find candidates where ours and baselines disagree most on rank.
    Returns the n candidates with the largest average rank difference.
    """
    flip_rows = []
    for cid, row in df.iterrows():
        our_rank = row["rank_ours"]
        baseline_ranks = {b: row[f"rank_{b}"] for b in BASELINES if not np.isnan(row[f"rank_{b}"])}
        if not baseline_ranks:
            continue
        avg_baseline_rank = float(np.mean(list(baseline_ranks.values())))
        flip_rows.append({
            "candidate_id": int(cid),
            "our_rank": int(our_rank),
            "avg_baseline_rank": round(avg_baseline_rank, 1),
            "rank_delta": round(avg_baseline_rank - our_rank, 1),  # positive = ours ranks it higher
            "per_baseline_rank": {k: int(v) for k, v in baseline_ranks.items()},
        })
    flip_rows.sort(key=lambda x: abs(x["rank_delta"]), reverse=True)
    return flip_rows[:n]


def plot_results(
        results_path: str = "results/disagreements.json",
        data_path: str = "data/stock_sequences.npz",
        meta_path: str = "data/stock_sequences_meta.json",
        n_iterations: int = 3,
        n_flips: int = 3,
        save_path: str = None,
):
    """
    For each of the top `n_iterations` disagreement cases, plot the anchor
    sequence alongside the top `n_flips` candidates that our method ranked
    low but baselines ranked high.

    Parameters
    ----------
    results_path  : path to the JSON written by run()
    data_path     : path to the sequences file
    meta_path     : path to the metadata file
    n_iterations  : how many anchor cases to plot (one row each)
    n_flips       : how many rank-flip candidates to show per anchor
    save_path     : if provided, saves figure here; otherwise shows interactively
    """
    with open(results_path) as f:
        data = json.load(f)

    # Load the data
    seqs, meta = load_sequences(data_path, meta_path)
    # Increase resolution of all sequences to 1000 (if not already)
    seqs = [np.interp(np.linspace(0, len(s) - 1, 1000), np.arange(len(s)), s) for s in seqs]

    iterations = data["top_disagreements"][:n_iterations]

    fig, axes = plt.subplots(
        n_iterations, n_flips + 1,
        figsize=(4 * (n_flips + 1), 3.2 * n_iterations),
    )
    # Ensure axes is always 2D
    if n_iterations == 1:
        axes = [axes]

    flip_colors = ["#E8834C", "#6DBE6D", "#B46DE8"]

    for row, it in enumerate(iterations):
        anchor_id = it["anchor_id"]
        anchor_seq = np.array(seqs[anchor_id], dtype=float)
        anchor_label = (
            it["anchor_meta"]["ticker"] if it.get("anchor_meta") else f"id={anchor_id}"
        )

        flips = it.get("rank_flips", [])[:n_flips]

        # ── anchor plot (first column) ────────────────────────────────────────
        ax = axes[row][0]
        ax.plot(anchor_seq, color="#4C9BE8", linewidth=2)
        ax.set_title(f"Anchor: {anchor_label}", fontweight="bold", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)

        # Row label
        ax.annotate(
            f"iter {it['iteration']}  |  disagreement={it['mean_disagreement']:.3f}",
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.35, 0.5), textcoords="axes fraction",
            fontsize=8, color="gray", rotation=90, va="center",
        )

        # ── one column per rank-flip candidate ────────────────────────────────
        for col, flip in enumerate(flips, start=1):
            cid = flip["candidate_id"]
            cseq = np.array(seqs[cid], dtype=float)
            color = flip_colors[(col - 1) % len(flip_colors)]

            ax = axes[row][col]

            # Plot anchor faintly for reference
            ax.plot(anchor_seq, color="#4C9BE8", linewidth=1.2, alpha=0.3, label="anchor")
            ax.plot(cseq, color=color, linewidth=2, label=f"candidate {cid}")

            our_rank = flip["our_rank"]
            base_rank = flip["avg_baseline_rank"]
            per_base = "  ".join(
                f"{k[:3]}={v}" for k, v in flip["per_baseline_rank"].items()
            )

            ax.set_title(
                f"Candidate {cid}\n"
                f"our rank: {our_rank}  |  avg baseline: {base_rank:.0f}",
                fontsize=9,
            )
            ax.set_xlabel(per_base, fontsize=7, color="gray")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Rank Flips: sequences our method ranked low, baselines ranked high\n"
        "(anchor shown in blue; faint blue in candidate plots = anchor for reference)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_top_similar(
        results_path: str = "results/disagreements.json",
        data_path: str = "data/stock_sequences.npz",
        meta_path: str = "data/stock_sequences_meta.json",
        n_iterations: int = 3,
        top_n: int = 5,
        save_path: str = None,
        n_candidates: int = 10,
):
    """
    For each of the top `n_iterations` disagreement cases, plot:
      - Row 0: the anchor sequence (full width)
      - Row 1: top `top_n` most similar by our method
      - Row 2: top `top_n` most similar by DTW
      - Row 3: top `top_n` most similar by LCSS

    Each candidate subplot overlays the anchor faintly for direct comparison.

    Parameters
    ----------
    results_path  : path to the JSON written by run()
    data_path     : path to the sequences file
    meta_path     : path to the metadata file
    n_iterations  : how many anchor cases to plot (one figure each)
    top_n         : how many top similar sequences to show per method
    save_path     : base path for saving (e.g. 'results/top_similar.png');
                    if multiple iterations, saves as top_similar_iter0.png etc.
                    If None, shows interactively.
    """
    with open(results_path) as f:
        data = json.load(f)

    seqs, meta = load_sequences(data_path, meta_path)
    seqs = [np.interp(np.linspace(0, len(s) - 1, 1000), np.arange(len(s)), s) for s in seqs]
    all_ids = list(range(len(seqs)))
    iterations = data["top_disagreements"][:n_iterations]

    methods = ["ours", "dtw", "lcss"]
    method_labels = {"ours": "Ours", "dtw": "DTW", "lcss": "LCSS"}
    method_colors = {"ours": "#E8834C", "dtw": "#6DBE6D", "lcss": "#B46DE8"}
    anchor_color = "#4C9BE8"

    for it in iterations:
        anchor_id = it["anchor_id"]
        anchor_seq = np.array(seqs[anchor_id], dtype=float)
        anchor_label = (
            it["anchor_meta"]["ticker"] if it.get("anchor_meta") else f"id={anchor_id}"
        )

        # ── compute distances from anchor to all other sequences ──────────────
        candidate_ids = [i for i in all_ids if i != anchor_id][:n_candidates]
        rows = []
        for cid in tqdm(candidate_ids, desc="Calculate distances"):
            d = all_distances(anchor_seq, np.array(seqs[cid], dtype=float))
            d["candidate_id"] = cid
            rows.append(d)

        # Sort by each method and grab top_n
        top_by_method = {}
        for m in methods:
            valid = [r for r in rows if r[m] is not None]
            sorted_rows = sorted(valid, key=lambda r: r[m])
            top_by_method[m] = sorted_rows[:top_n]

        # ── figure layout: 1 anchor row + 1 row per method ───────────────────
        n_rows = len(methods)
        fig = plt.figure(figsize=(5 * top_n, 3.2 * n_rows))
        fig.suptitle(
            f"Anchor: {anchor_label}  (iter {it['iteration']}, "
            f"disagreement={it['mean_disagreement']:.3f})",
            fontsize=30, fontweight="bold",
        )

        gs = gridspec.GridSpec(
            n_rows, top_n, figure=fig, hspace=0.55, wspace=0.3
        )

        # ── Row 0: anchor (spans all columns) ────────────────────────────────
        # ax_anchor = fig.add_subplot(gs[0, :])
        # ax_anchor.plot(anchor_seq, color=anchor_color, linewidth=2)
        # ax_anchor.set_title("Anchor sequence", fontweight="bold", fontsize=10)
        # ax_anchor.set_ylim(-0.05, 1.05)
        # ax_anchor.set_ylabel("Value")
        # ax_anchor.set_xlabel("Time step")
        # ax_anchor.grid(alpha=0.25)

        # ── Rows 1–3: top_n candidates per method ────────────────────────────
        for row_idx, m in enumerate(methods, start=0):
            color = method_colors[m]
            label = method_labels[m]

            for col_idx, r in enumerate(top_by_method[m]):
                cid = r["candidate_id"]
                cseq = np.array(seqs[cid], dtype=float)
                dist = r[m]

                ax = fig.add_subplot(gs[row_idx, col_idx])

                # Faint anchor for reference
                ax.plot(anchor_seq, color=anchor_color, linewidth=1,
                        alpha=0.3, label="anchor")
                ax.plot(cseq, color=color, linewidth=2,
                        label=f"id={cid}")

                ax.set_title(
                    f"#{col_idx + 1} by {label}\n"
                    f"id={cid}  dist={dist:.4f}",
                    fontsize=20,
                )
                ax.set_ylim(-0.05, 1.05)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7, loc="upper right")

                # Row label on leftmost subplot
                if col_idx == 0:
                    ax.set_ylabel(label, fontsize=15, fontweight="bold",
                                  color=color)

        plt.tight_layout()  # leave space for suptitle

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if n_iterations > 1:
                base, ext = save_path.rsplit(".", 1)
                path = f"{base}_iter{it['iteration']}.{ext}"
            else:
                path = save_path
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {path}")
        else:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":
    # run(n_iterations=1, n_candidates=20)
    # plot_results()
    plot_top_similar(n_candidates=100)
