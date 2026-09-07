"""
Yuli Tshuva
Show the same mined sequences at three smoothing levels.

Columns are different sequences, rows are smoothing windows. Reading a column
top to bottom shows what the smoothing knob buys: the jitter goes, the structure
stays, and the turning-point count (at the strict 0.02 tolerance) settles into
the target band.
"""

import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from complexity import turning_points, _prepare, TURNING_POINT_TOL
from mine_data import SECTORS, _random_window, _smooth
from mine_demo import fetch_raw

SMOOTH_WINDOWS = [1, 5, 12, 25]
RESAMPLE_LEN = 128
MIN_TURNS, MAX_TURNS = 2, 8


def collect(n_columns, seed, min_days=120):
    """Fetch raw price windows long enough to survive heavy smoothing."""
    random.seed(seed)
    found = []
    while len(found) < n_columns:
        sector = random.choice(list(SECTORS))
        ticker = random.choice(SECTORS[sector])
        start, end = _random_window(6, 12, "2010-01-01", "2025-06-01")
        arr = fetch_raw(ticker, start, end)
        if arr is not None and len(arr) >= min_days:
            found.append((ticker, start, arr))
            print(f"  {ticker:5s} {start}  {len(arr)} trading days")
    return found


def plot(found, out_path):
    fig, axes = plt.subplots(len(SMOOTH_WINDOWS), len(found),
                             figsize=(3.4 * len(found), 2.2 * len(SMOOTH_WINDOWS)))

    for row, window in enumerate(SMOOTH_WINDOWS):
        for col, (ticker, start, arr) in enumerate(found):
            smoothed = arr if window == 1 else _smooth(arr, window)
            curve = _prepare(smoothed, RESAMPLE_LEN)
            n_turns = turning_points(curve, TURNING_POINT_TOL)
            in_band = MIN_TURNS <= n_turns <= MAX_TURNS

            ax = axes[row, col]
            ax.plot(curve, lw=1.3, color="#1a1a1a" if in_band else "#b0b0b0")
            ax.set_title(f"{n_turns} turns" + ("" if in_band else "  (out of band)"),
                         fontsize=9, color="#1a1a1a" if in_band else "#b0b0b0")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.text(0.5, 1.35, f"{ticker}  {start[:7]}", transform=ax.transAxes,
                        ha="center", fontsize=10, weight="bold")

        label = "no smoothing" if window == 1 else f"smooth = {window}"
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f"Same sequences at four smoothing levels "
                 f"(turning points at tolerance {TURNING_POINT_TOL}, "
                 f"target band {MIN_TURNS}-{MAX_TURNS})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--out", default="reports/figs/smoothing_levels.png")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    plot(collect(args.columns, args.seed), args.out)


if __name__ == "__main__":
    main()
