"""
Yuli Tshuva
Build the trials file the labeling site serves.

Takes the sampler's output plus the sequence archive and emits the only thing
the browser ever sees: curves and their grouping into trials. Every trace of
which algorithm voted how is deliberately dropped here, so the collected labels
stay measure-agnostic and any distance -- including one written later -- can be
scored against them without running the study again.

    python build_trials.py --triplets ../triplets.json \
                           --sequences ../../data/study_sequences.npz \
                           --out trials.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

RENDER_LEN = 160    # points per curve as drawn; enough for shape, small over the wire
DECIMALS = 4


def to_curve(seq, length=RENDER_LEN):
    """Resample to the rendered length and scale to [0, 1]."""
    seq = np.asarray(seq, dtype=float)
    out = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(seq)), seq)
    span = out.max() - out.min()
    out = (out - out.min()) / span if span > 0 else out * 0.0
    return [round(float(v), DECIMALS) for v in out]


def load_sequences(path):
    """Load sequences from either the study archive or a plain seq_i archive."""
    data = np.load(path, allow_pickle=False)
    if "metadata_json" in data.files:
        n = len(json.loads(str(data["metadata_json"])))
    else:
        n = len([k for k in data.files if k.startswith("seq_")])
    return [data[f"seq_{i}"] for i in range(n)]


def build(triplets, sequences):
    curves, trials = {}, []

    def register(idx):
        key = f"s{idx}"
        if key not in curves:
            curves[key] = to_curve(sequences[idx])
        return key

    for i, t in enumerate(triplets):
        trials.append({
            "trial_id": t.get("trial_id", f"t{i:04d}"),
            "query": register(t["query"]),
            "candidate_a": register(t["candidate_a"]),
            "candidate_b": register(t["candidate_b"]),
        })
    return {"version": 1, "sequences": curves, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", required=True,
                        help="JSON list from sample_triplets.py")
    parser.add_argument("--sequences", required=True, help=".npz of sequences")
    parser.add_argument("--out", default="trials.json")
    args = parser.parse_args()

    triplets = json.load(open(args.triplets, encoding="utf-8"))
    sequences = load_sequences(args.sequences)
    payload = build(triplets, sequences)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    size = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out}: {len(payload['trials'])} trials, "
          f"{len(payload['sequences'])} curves, {size:.0f} KB")
    print("No algorithm votes are included -- the labels stay measure-agnostic.")


if __name__ == "__main__":
    main()
