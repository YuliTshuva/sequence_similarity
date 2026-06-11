import os
import yfinance as yf
import numpy as np
import random
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from utils import change_points_detection

# ── Sector universe ────────────────────────────────────────────────────────────
SECTORS = {
    "technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "INTC", "AMD", "ORCL", "CRM", "ADBE"],
    "consumer": ["AMZN", "TSLA", "WMT", "TGT", "COST", "HD", "NKE", "MCD", "SBUX", "LOW"],
    "finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP", "USB", "SCHW"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "OXY", "PSX"],
    "industrials": ["CAT", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE", "FDX"],
    "utilities": ["NEE", "DUK", "SO", "AEP", "EXC", "SRE", "PEG", "ED", "XEL", "WEC"],
    "realestate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _random_window(
        min_months: int,
        max_months: int,
        earliest: str,
        latest: str,
) -> tuple[str, str]:
    """Return a (start, end) date-string pair for a random window."""
    earliest_dt = datetime.strptime(earliest, "%Y-%m-%d")
    latest_dt = datetime.strptime(latest, "%Y-%m-%d")
    duration = random.randint(min_months, max_months)

    max_start = latest_dt - relativedelta(months=duration)
    if max_start <= earliest_dt:
        raise ValueError("Date range too narrow for requested window length.")

    days_available = (max_start - earliest_dt).days
    start = earliest_dt + timedelta(days=random.randint(0, days_available))
    end = start + relativedelta(months=duration)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _smooth(arr: np.ndarray, window: int) -> np.ndarray | None:
    """Moving-average smooth; returns None if the result would be empty."""
    if len(arr) <= window:
        return None
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def _fetch_and_smooth(
        ticker: str,
        start: str,
        end: str,
        window: int,
) -> np.ndarray | None:
    """Download closing prices, smooth them, and return the array (or None on failure)."""
    try:
        data = yf.download(
            ticker, start=start, end=end,
            progress=False, multi_level_index=False,
        )
        if data.empty:
            return None
        arr = data["Close"].dropna().to_numpy()
        return _smooth(arr, window)
    except Exception:
        return None


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_stock_sequences(
        n_sequences: int,
        tickers_per_sector: int,  # companies sampled per sector per batch
        min_window_months: int,
        max_window_months: int,
        min_smooth_window: int,
        max_smooth_window: int,
        min_change_points: int,
        max_change_points: int,
        earliest_date: str,
        latest_date: str,
        save_path: str,
        metadata_path: str,
        seed: int,
) -> list[np.ndarray]:
    """
    Generate `n_sequences` smoothed stock trend sequences sampled across sectors.

    Each sequence is drawn by:
      1. Randomly picking a sector.
      2. Sampling `tickers_per_sector` tickers from that sector.
      3. Choosing a random date window and smoothing window per ticker.
      4. Downloading, smoothing, and storing the result.

    Saves:
      - `save_path`     (.npz)  — arrays under keys seq_0, seq_1, …
      - `metadata_path` (.json) — ticker / date / window info per sequence

    Returns a list of numpy arrays.
    """
    sequences: list[np.ndarray] = []
    metadata: list[dict] = []

    if os.path.exists(save_path) and os.path.exists(metadata_path):
        sequences, metadata = load_sequences(save_path, metadata_path)
        print(f"Resuming from checkpoint: {len(sequences)} sequences already saved.")
        random.seed(seed + len(sequences))
        np.random.seed(seed + len(sequences))
    else:
        random.seed(seed)
        np.random.seed(seed)

    sequences, metadata = sequences[:7400], metadata[:7400]

    sector_names = list(SECTORS.keys())

    while len(sequences) < n_sequences:
        # Pick a random sector and sample tickers from it
        sector = random.choice(sector_names)
        tickers = random.sample(SECTORS[sector], k=min(tickers_per_sector, len(SECTORS[sector])))

        for ticker in tickers:
            if len(sequences) >= n_sequences:
                break

            start, end = _random_window(min_window_months, max_window_months,
                                        earliest_date, latest_date)
            smooth_window = random.randint(min_smooth_window, max_smooth_window)

            arr = _fetch_and_smooth(ticker, start, end, smooth_window)
            if arr is None or len(arr) < 10:  # skip sequences that are too short
                continue

            # Normalize to [0, 1] so sequences are amplitude-agnostic
            lo, hi = arr.min(), arr.max()
            arr_norm = (arr - lo) / (hi - lo + 1e-8)

            # Interpolate to length 1000 for uniformity
            arr_norm = np.interp(np.linspace(0, len(arr_norm) - 1, 1000), np.arange(len(arr_norm)), arr_norm)

            # Find the change points in the sequence and skip if there are too few
            cps = change_points_detection(arr_norm)

            # If the sequence is too simple, skip it
            if len(cps) < min_change_points:
                continue

            # If the sequence is too complicated
            while len(cps) > max_change_points:
                # Find the last change point to keep
                last_change_point = cps[max_change_points - 1]
                # Trim to max change points
                arr_norm = arr_norm[:last_change_point]
                # Normalize again after trimming
                lo, hi = arr_norm.min(), arr_norm.max()
                arr_norm = (arr_norm - lo) / (hi - lo + 1e-8)
                # Interpolate to length 1000 for uniformity
                arr_norm = np.interp(np.linspace(0, len(arr_norm) - 1, 1000), np.arange(len(arr_norm)), arr_norm)
                # Recompute change points
                cps = change_points_detection(arr_norm)

            sequences.append(arr_norm)
            metadata.append({
                "index": len(sequences) - 1,
                "ticker": ticker,
                "sector": sector,
                "start": start,
                "end": end,
                "smooth_window": smooth_window,
                "length": len(arr_norm),
            })

            print(f"[{len(sequences):>4}/{n_sequences}]  {ticker:6s}  {sector:12s}  "
                  f"{start} → {end} len={len(arr_norm)}")

        # ── Persist ────────────────────────────────────────────────────────────────
        # Save every 100 sequences to avoid losing progress on long runs
        if len(sequences) % 100 == 0:
            np.savez(save_path, **{f"seq_{i}": s for i, s in enumerate(sequences)})
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(sequences)} sequences → {save_path}")
    print(f"Metadata                        → {metadata_path}")
    return sequences


# ── Loading utility ────────────────────────────────────────────────────────────

def load_sequences(
        save_path: str,
        metadata_path: str,
) -> tuple[list[np.ndarray], list[dict]]:
    """Reload sequences and metadata saved by generate_stock_sequences."""
    data = np.load(save_path, allow_pickle=False)
    sequences = [data[f"seq_{i}"] for i in range(len(data.files))]
    with open(metadata_path) as f:
        metadata = json.load(f)
    return sequences, metadata


def main():
    generate_stock_sequences(
        n_sequences=10000,
        save_path="data/ten_plus_cps_sequences.npz",
        metadata_path="data/ten_plus_cps_sequences_meta.json",
        min_window_months=6, max_window_months=6,
        min_smooth_window=5, max_smooth_window=5,
        min_change_points=10, max_change_points=600,
        tickers_per_sector=1,
        earliest_date="2010-01-01",
        latest_date="2026-01-01",
        seed=42
    )

    # # Later, reload without re-downloading:
    # seqs, meta = load_sequences("data/stock_sequences.npz", "data/stock_sequences_meta.json")


if __name__ == "__main__":
    main()
