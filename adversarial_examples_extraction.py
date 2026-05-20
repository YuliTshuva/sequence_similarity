import numpy as np
from os.path import join
from mine_data import load_sequences
from seq_sim_alg import seq_distance
from compare_baselines import dtw_distance, lcss_distance
from tqdm.auto import tqdm
import pickle

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH = join("data", "stock_sequences.npz")
META_PATH = join("data", "stock_sequences_meta.json")
INTERP_LEN = 1000
TOP_K = 10
BUFFER = 90
SEED = 42


def main():
    # Load 1000 sequences
    seqs, _ = load_sequences()

    # Increase sequences to INTERP_LEN using linear interpolation
    high_res_seqs = [np.interp(np.linspace(0, len(s) - 1, INTERP_LEN), np.arange(len(s)), s) for s in seqs]

    # Set a dict for storing results
    results = {}

    # For each anchor, rank all sequences by distance of DTW, LCSS, and our method
    for anchor in range(0, len(seqs)):
        # Calculate distances
        dists_dtw = [dtw_distance(seqs[anchor], s) for i, s in tqdm(enumerate(seqs), desc="DTW", total=len(seqs) - 1) if
                     i != anchor]
        dists_lcss = [lcss_distance(seqs[anchor], s) for i, s in tqdm(enumerate(seqs), desc="LCSS", total=len(seqs) - 1)
                      if i != anchor]
        anchor_seq = high_res_seqs[anchor]
        dists_ours = [seq_distance(anchor_seq, s)[0] for i, s in
                      tqdm(enumerate(high_res_seqs), desc="Ours", total=len(high_res_seqs) - 1) if i != anchor]

        # Rank sequences by distance
        rank_dtw = np.argsort(dists_dtw)
        rank_lcss = np.argsort(dists_lcss)
        rank_ours = np.argsort(dists_ours)

        # Find all sequences that are in the top 100 of DTW and LCSS but in the bottom 100 of our method
        top_dtw = set(rank_dtw[:TOP_K])
        top_lcss = set(rank_lcss[:TOP_K])
        bottom_ours = set(rank_ours[TOP_K + BUFFER - 1:])

        # Find all sequences that are in the bottom 100 of DTW and LCSS but in the top 100 of our method
        bottom_dtw = set(rank_dtw[TOP_K + BUFFER - 1:])
        bottom_lcss = set(rank_lcss[TOP_K + BUFFER - 1:])
        top_ours = set(rank_ours[:TOP_K])

        # Find adversarial examples
        adversarial_examples_class_1 = top_dtw.intersection(bottom_ours)
        adversarial_examples_class_2 = top_lcss.intersection(bottom_ours)
        adversarial_examples_class_3 = bottom_dtw.intersection(top_ours)
        adversarial_examples_class_4 = bottom_lcss.intersection(top_ours)

        # Store results
        results[anchor] = {
            "rank_dtw": rank_dtw,
            "rank_lcss": rank_lcss,
            "rank_ours": rank_ours,
            "class_1": list(adversarial_examples_class_1),
            "class_2": list(adversarial_examples_class_2),
            "class_3": list(adversarial_examples_class_3),
            "class_4": list(adversarial_examples_class_4),
        }

        # Print results
        print(f"Anchor: {anchor}")
        print(f"Class 1: {adversarial_examples_class_1}")
        print(f"Class 2: {adversarial_examples_class_2}")
        print(f"Class 3: {adversarial_examples_class_3}")
        print(f"Class 4: {adversarial_examples_class_4}")
        print("-" * 50)

        # Save results to a file
        with open(join("results", f"adversarial_examples.pkl"), "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    main()
