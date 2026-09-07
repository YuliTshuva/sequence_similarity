"""
Yuli Tshuva
Load sequences from UCR archive zips (as distributed by timeseriesclassification.com).

Why this source: stock prices are close to a random walk, so their wiggles are
noise rather than events -- there is no structure for a rater to perceive, which
is why no amount of smoothing made them meaningful. UCR series are recordings of
actual things (a heartbeat, a hand gesture, a leaf outline), so two curves are
similar because they are the same kind of thing, and the class labels give a
second, independent anchor for what "similar" ought to mean.
"""

import io
import os
import zipfile

import numpy as np

DEFAULT_DATASETS = ("ECG5000", "GunPoint", "OSULeaf", "Trace", "Wafer", "Strawberry")


def _parse_ts_text(text):
    """
    Parse a UCR .txt split: one series per line, whitespace separated, with the
    class label in the first column.
    """
    rows, labels = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue
        labels.append(values[0])
        rows.append(np.array(values[1:], dtype=float))
    return rows, labels


def load_dataset(zip_path, split="TRAIN"):
    """
    Return (sequences, labels, name) for one dataset zip.

    Series carrying NaNs are dropped rather than imputed -- a few UCR datasets
    pad variable-length series, and an interpolated tail is not a real shape.
    """
    name = os.path.splitext(os.path.basename(zip_path))[0]
    with zipfile.ZipFile(zip_path) as z:
        candidates = [n for n in z.namelist()
                      if n.endswith(f"_{split}.txt") or n.endswith(f"_{split}.tsv")]
        if not candidates:
            raise FileNotFoundError(f"no {split} split in {zip_path}: {z.namelist()[:6]}")
        with z.open(candidates[0]) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace").read()

    rows, labels = _parse_ts_text(text)
    keep = [(r, l) for r, l in zip(rows, labels) if np.isfinite(r).all() and len(r) > 8]
    if not keep:
        return [], [], name
    return [r for r, _ in keep], [l for _, l in keep], name


def load_many(zip_dir, datasets=DEFAULT_DATASETS, split="TRAIN"):
    """Load several datasets, returning {name: (sequences, labels)}."""
    out = {}
    for ds in datasets:
        path = os.path.join(zip_dir, f"{ds}.zip")
        if not os.path.exists(path):
            print(f"  {ds}: missing at {path}")
            continue
        seqs, labels, name = load_dataset(path, split)
        if seqs:
            out[name] = (seqs, labels)
            lengths = {len(s) for s in seqs}
            print(f"  {name:12s} {len(seqs):5d} series, length {sorted(lengths)[:3]}"
                  f"{'...' if len(lengths) > 3 else ''}, "
                  f"{len(set(labels))} classes")
    return out
