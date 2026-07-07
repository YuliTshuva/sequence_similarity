from mine_data import load_sequences
from seq_sim_alg import seq_distance
import numpy as np
import os
from os.path import join
from compare_baselines import dtw_distance, lcss_d2
import pickle
import argparse

# ── config ────────────────────────────────────────────────────────────────────
INTERP_LEN = 1000
TOP_K = 11
SEED = 42
BUFFER = 100


def adversarial_stocks(dataset_prefix):
    data_path = join("data", f"{dataset_prefix}_cps_sequences.npz")
    meta_path = join("data", f"{dataset_prefix}_cps_sequences_meta.json")

    # Load 1000 sequences
    seqs, _ = load_sequences(save_path=data_path, metadata_path=meta_path)

    # Increase sequences to INTERP_LEN using linear interpolation
    seqs = [np.interp(np.linspace(0, len(s) - 1, INTERP_LEN), np.arange(len(s)), s) for s in seqs]

    # Set a dict for storing results, resuming from a previous run if available
    results_path = join("results", f"{dataset_prefix}_cps_adversarial_examples.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        start_anchor = max(results.keys()) + 1 if results else 0
    else:
        results = {}
        start_anchor = 0

    # For each anchor, rank all sequences by distance of DTW, LCSS, and our method
    for anchor in range(start_anchor, len(seqs)):
        # Calculate distances
        dists_dtw = []
        dists_ours = []

        for i, s in enumerate(seqs):
            dists_dtw.append(dtw_distance(seqs[anchor], s))
            dists_ours.append(seq_distance(seqs[anchor], s)[0])

            if i % 100 == 0:
                with open("output.log", "w") as f:
                    f.write(f"Anchor {anchor}, sequence {i} processed.")

        # Rank sequences by distance
        rank_dtw = np.argsort(dists_dtw)
        rank_ours = np.argsort(dists_ours)

        # Store results
        results[anchor] = {
            "rank_dtw": rank_dtw,
            "rank_ours": rank_ours
        }

        # Save results to a file
        with open(results_path, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_prefix", default="ten_plus", help="Dataset prefix (e.g. six / ten_plus)")
    args = parser.parse_args()
    adversarial_stocks(args.dataset_prefix)
