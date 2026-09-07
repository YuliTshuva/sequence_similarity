"""
Yuli Tshuva
Compare candidate sequence sources side by side, raw and unsmoothed.

One row per dataset, several real series across the row, coloured by class. The
question the figure answers is whether curves from the same class visibly belong
together -- which is what stock windows never had, and what makes a similarity
judgement mean something.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complexity import turning_points, _prepare
from ucr_loader import load_many

CLASS_COLOURS = ["#6d5efc", "#12c2b6", "#ff5c93", "#ffb020", "#4b3ddb", "#0b6a63"]


def plot_sources(data, out_path, per_row=5, stock_npz=None, seed=1):
    rng = np.random.default_rng(seed)
    rows = list(data.items())

    if stock_npz and os.path.exists(stock_npz):
        # The raw crop, not the smoothed curve -- every UCR row below is raw, so
        # showing the smoothed stock version would flatter it by comparison.
        # Read the archive directly rather than importing mine_sequences, which
        # drags in yfinance and pandas for no reason here.
        import json
        archive = np.load(stock_npz, allow_pickle=False)
        n = len(json.loads(str(archive["metadata_json"])))
        raw = [archive[f"raw_{i}"] for i in range(n)]
        rows.insert(0, ("Stocks (raw)", (raw, [0] * len(raw))))

    fig, axes = plt.subplots(len(rows), per_row,
                             figsize=(2.7 * per_row, 1.9 * len(rows)))
    axes = np.atleast_2d(axes)

    for r, (name, (seqs, labels)) in enumerate(rows):
        classes = sorted(set(labels))
        # Show one series per class where possible, so within-row differences are
        # differences of kind rather than of luck.
        picks = []
        for cls in classes[:per_row]:
            members = [i for i, l in enumerate(labels) if l == cls]
            picks.append(int(rng.choice(members)))
        while len(picks) < per_row:
            picks.append(int(rng.integers(len(seqs))))

        for c, idx in enumerate(picks[:per_row]):
            ax = axes[r, c]
            curve = _prepare(seqs[idx], 160)
            colour = CLASS_COLOURS[classes.index(labels[idx]) % len(CLASS_COLOURS)] \
                if labels[idx] in classes else "#1a1a1a"
            ax.plot(curve, lw=1.4, color=colour)
            ax.set_title(f"class {labels[idx]:g} · {turning_points(curve)} turns"
                         if len(classes) > 1 else f"{turning_points(curve)} turns",
                         fontsize=8.5, color="#444")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(name, fontsize=10, weight="bold")

    fig.suptitle("Candidate sources, raw and unsmoothed — colour marks the class",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ucr-dir", required=True)
    parser.add_argument("--stock-npz", default="data/study_sequences.npz")
    parser.add_argument("--out", default="reports/figs/source_comparison.png")
    args = parser.parse_args()
    plot_sources(load_many(args.ucr_dir), args.out, stock_npz=args.stock_npz)


if __name__ == "__main__":
    main()
