"""
Yuli Tshuva
Mine a handful of sequences with no smoothing, keeping only those whose
turning-point count falls in the target band, and plot them.

A demonstration of the proposed filter, not a replacement for mine_data.py.
"""

import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from complexity import turning_points, _prepare, TURNING_POINT_TOL
from mine_data import SECTORS, _random_window

MIN_TURNING_POINTS = 2
MAX_TURNING_POINTS = 8
RESAMPLE_LEN = 128


def fetch_raw(ticker, start, end):
    """Closing prices with no smoothing applied."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False,
                           multi_level_index=False, auto_adjust=True)
        if data.empty:
            return None
        arr = data["Close"].dropna().to_numpy()
        return arr if len(arr) >= 10 else None
    except Exception:
        return None


def mine(n_sequences, min_tp, max_tp, tol=TURNING_POINT_TOL, min_months=3, max_months=12,
         earliest="2010-01-01", latest="2025-06-01", seed=0, max_attempts=200):
    """Draw random ticker/window pairs until `n_sequences` pass the filter."""
    random.seed(seed)
    kept, rejected, attempts = [], [], 0

    while len(kept) < n_sequences and attempts < max_attempts:
        attempts += 1
        sector = random.choice(list(SECTORS))
        ticker = random.choice(SECTORS[sector])
        start, end = _random_window(min_months, max_months, earliest, latest)

        arr = fetch_raw(ticker, start, end)
        if arr is None:
            continue

        curve = _prepare(arr, RESAMPLE_LEN)
        n_turns = turning_points(curve, tol)
        record = {"ticker": ticker, "sector": sector, "start": start, "end": end,
                  "n_days": len(arr), "turning_points": n_turns, "curve": curve}

        if min_tp <= n_turns <= max_tp:
            kept.append(record)
            print(f"  KEEP   {ticker:5s} {start}..{end}  {len(arr):3d}d  turns={n_turns}")
        else:
            rejected.append(record)
            print(f"  reject {ticker:5s} {start}..{end}  {len(arr):3d}d  turns={n_turns}")

    return kept, rejected, attempts


def plot(kept, out_path, min_tp, max_tp, tol):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, rec in zip(axes.ravel(), kept):
        ax.plot(rec["curve"], lw=1.3, color="#1a1a1a")
        ax.set_title(f"{rec['ticker']}  {rec['start'][:7]}..{rec['end'][:7]}\n"
                     f"{rec['n_days']} trading days, {rec['turning_points']} turns",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"Unsmoothed closing prices, filtered to "
                 f"{min_tp}-{max_tp} turning points "
                 f"(tolerance {tol})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--min-turns", type=int, default=MIN_TURNING_POINTS)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNING_POINTS)
    parser.add_argument("--out", default="reports/figs/mined_unsmoothed.png")
    parser.add_argument("--tol", type=float, default=TURNING_POINT_TOL)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kept, rejected, attempts = mine(args.n, args.min_turns, args.max_turns,
                                    tol=args.tol, seed=args.seed)
    print(f"\n{len(kept)} kept / {len(rejected)} rejected over {attempts} downloads")
    if rejected:
        turns = np.array([r["turning_points"] for r in rejected])
        print(f"rejected turning-point counts: min={turns.min()} "
              f"median={int(np.median(turns))} max={turns.max()}")
    if kept:
        plot(kept[:9], args.out, args.min_turns, args.max_turns, args.tol)


if __name__ == "__main__":
    main()
