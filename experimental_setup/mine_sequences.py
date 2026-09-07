"""
Yuli Tshuva
Mine stock sequences for the human-alignment study.

The problem this solves: forcing a long, busy price window into the target
complexity band requires heavy smoothing, and heavy smoothing destroys the very
structure that made the sequence worth showing -- a curve with six real events
becomes a smooth ramp. The structure was never too fine, the *window* was too
long.

So instead of smoothing harder, this downloads a generous window and searches
inside it for a crop that already sits in the band under light smoothing. The
search runs smoothing levels in ascending order and takes the first hit, so
every stored sequence carries the least smoothing that worked for it.

Provenance: arrays and metadata are written in one atomic save, with the
metadata duplicated inside the .npz, so the two cannot drift apart across
resumed runs. Each record stores its raw crop alongside the smoothed curve and
records the exact dates, so any sequence can be re-derived from a fresh
download.
"""

import argparse
import json
import os
import random

import numpy as np
import yfinance as yf

from complexity import turning_points, _prepare, TURNING_POINT_TOL
from mine_data import SECTORS, _random_window, _smooth

STORE_LEN = 1000            # stored resolution
MEASURE_LEN = 128           # length complexity is judged at
MIN_TURNS, MAX_TURNS = 2, 8
TARGET_RANGE = (3, 6)       # per-sequence target, so the set varies without drifting busy
SMOOTH_LEVELS = range(3, 10)     # tried in ascending order; lightest wins
MIN_CROP_DAYS, MAX_CROP_DAYS = 30, 110
FETCH_MONTHS = (9, 24)      # download wide, then crop
CROP_CANDIDATES = 80
EARLIEST, LATEST = "2010-01-01", "2025-06-01"


def fetch(ticker, start, end):
    """Unadjusted closes with their dates, or None if unusable."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False,
                           multi_level_index=False, auto_adjust=False)
        if data.empty:
            return None
        series = data["Close"].dropna()
        if len(series) < MIN_CROP_DAYS + 10:
            return None
        return series.to_numpy(dtype=float), series.index.strftime("%Y-%m-%d").tolist()
    except Exception:
        return None


def find_crop(values, rng, target, n_candidates=CROP_CANDIDATES):
    """
    Find a sub-window that lands in the turning-point band under light smoothing.

    Smoothing levels are tried outermost and ascending, so a crop that works at
    smooth=3 is always preferred over one needing smooth=9; among the crops that
    work at that level, the one closest to `target` turns wins. Returns
    (start, length, smooth_window, n_turns) or None.
    """
    n = len(values)
    max_len = min(MAX_CROP_DAYS, n)
    if max_len < MIN_CROP_DAYS:
        return None

    crops = []
    for _ in range(n_candidates):
        length = rng.randint(MIN_CROP_DAYS, max_len)
        crops.append((rng.randint(0, n - length), length))

    for smooth_window in SMOOTH_LEVELS:
        in_band = []
        for start, length in crops:
            if length <= smooth_window + 15:
                continue
            smoothed = _smooth(values[start:start + length], smooth_window)
            n_turns = turning_points(_prepare(smoothed, MEASURE_LEN), TURNING_POINT_TOL)
            if MIN_TURNS <= n_turns <= MAX_TURNS:
                in_band.append((start, length, smooth_window, n_turns))
        # Among the crops that work at this smoothing level, take the one nearest
        # the requested target. Returning the first hit instead skews the whole
        # dataset to 7-8 turns, because busy crops are simply more common.
        if in_band:
            return min(in_band, key=lambda c: abs(c[3] - target))
    return None


def _stretch(arr, length):
    """Resample to `length` points without rescaling the values."""
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(arr)), arr)


def save(records, npz_path, meta_path):
    """Write arrays and metadata as one unit, atomically."""
    meta = [r["meta"] for r in records]
    arrays = {}
    for i, r in enumerate(records):
        arrays[f"seq_{i}"] = r["smoothed"]
        arrays[f"raw_{i}"] = r["raw"]
    arrays["metadata_json"] = np.array(json.dumps(meta))

    np.savez(npz_path + ".tmp.npz", **arrays)
    os.replace(npz_path + ".tmp.npz", npz_path)
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_path + ".tmp", meta_path)


def load(npz_path):
    """Load records and the metadata carried inside the archive."""
    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    smoothed = [data[f"seq_{i}"] for i in range(len(meta))]
    raw = [data[f"raw_{i}"] for i in range(len(meta))]
    return smoothed, raw, meta


def mine(n_sequences, npz_path, meta_path, seed=0, checkpoint_every=25, max_attempts=None):
    rng = random.Random(seed)
    records, attempts, no_crop = [], 0, 0
    max_attempts = max_attempts or n_sequences * 8

    while len(records) < n_sequences and attempts < max_attempts:
        attempts += 1
        sector = rng.choice(list(SECTORS))
        ticker = rng.choice(SECTORS[sector])
        random.seed(rng.randrange(1 << 30))  # _random_window uses the global RNG
        start, end = _random_window(*FETCH_MONTHS, EARLIEST, LATEST)

        fetched = fetch(ticker, start, end)
        if fetched is None:
            continue
        values, dates = fetched

        found = find_crop(values, rng, rng.randint(*TARGET_RANGE))
        if found is None:
            no_crop += 1
            continue
        crop_start, crop_len, smooth_window, n_turns = found
        crop = values[crop_start:crop_start + crop_len]

        records.append({
            "smoothed": _stretch(_smooth(crop, smooth_window), STORE_LEN),
            "raw": _stretch(crop, STORE_LEN),
            "meta": {
                "index": len(records),
                "ticker": ticker,
                "sector": sector,
                "fetch_start": start,
                "fetch_end": end,
                "crop_start_date": dates[crop_start],
                "crop_end_date": dates[crop_start + crop_len - 1],
                "crop_offset": int(crop_start),
                "n_raw_days": int(crop_len),
                "smooth_window": int(smooth_window),
                "turning_points": int(n_turns),
            },
        })
        print(f"[{len(records):>4}/{n_sequences}]  {ticker:5s} {sector:12s} "
              f"{dates[crop_start]}..{dates[crop_start + crop_len - 1]}  "
              f"{crop_len:3d}d  smooth={smooth_window}  turns={n_turns}", flush=True)

        if len(records) % checkpoint_every == 0:
            save(records, npz_path, meta_path)

    save(records, npz_path, meta_path)
    print(f"\nKept {len(records)} over {attempts} downloads "
          f"({no_crop} had no crop in band)")
    print(f"  {npz_path}\n  {meta_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--npz", default="data/study_sequences.npz")
    parser.add_argument("--meta", default="data/study_sequences_meta.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    mine(args.n, args.npz, args.meta, args.seed)


if __name__ == "__main__":
    main()
