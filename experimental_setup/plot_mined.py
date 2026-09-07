"""
Yuli Tshuva
Plot freshly mined sequences: raw closing prices on top, the smoothed curve that
will actually be shown to raters underneath.

Each column is one sequence, so a column read top to bottom shows exactly what
the smoothing removed and what it kept.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from complexity import _prepare
from mine_sequences import load

RAW_COLOUR = "#b0b0b0"
SMOOTH_COLOUR = "#1a1a1a"


def plot(smoothed, raw, meta, out_path, n_columns=5, seed=0):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(meta), size=min(n_columns, len(meta)), replace=False)

    fig, axes = plt.subplots(2, len(picks), figsize=(3.1 * len(picks), 5.0))
    axes = np.atleast_2d(axes)

    for col, i in enumerate(picks):
        m = meta[i]
        for row, (curve, colour) in enumerate(((raw[i], RAW_COLOUR),
                                               (smoothed[i], SMOOTH_COLOUR))):
            ax = axes[row, col]
            ax.plot(_prepare(curve, 128), lw=1.3, color=colour)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[0, col].set_title(f"{m['ticker']}  {m['crop_start_date'][:7]}\n"
                               f"{m['n_raw_days']} days", fontsize=9)
        axes[1, col].set_title(f"smooth={m['smooth_window']}, "
                               f"{m['turning_points']} turns", fontsize=9)

    axes[0, 0].set_ylabel("raw crop", fontsize=10, color=RAW_COLOUR)
    axes[1, 0].set_ylabel("smoothed", fontsize=10)
    fig.suptitle("Freshly mined sequences: raw prices above, "
                 "the curve raters will see below", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default="data/study_sequences.npz")
    parser.add_argument("--out", default="reports/figs/mined_raw_vs_smoothed.png")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    smoothed, raw, meta = load(args.npz)
    print(f"Loaded {len(meta)} sequences")
    plot(smoothed, raw, meta, args.out, args.columns, args.seed)


if __name__ == "__main__":
    main()
