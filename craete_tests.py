"""
Creating test files for sequence similarity algorithms.
"""

from os.path import join
import pickle
from mine_data import load_sequences
from utils import *

# Set test directory
TEST_DIR = join("data", "tests")


def main():
    # Read ranking of baselines
    with open(join("results", "adversarial_examples.pkl"), "rb") as f:
        results = pickle.load(f)

    print(f"Total anchors: {len(results)}")

    # Load sequences
    sequences, _ = load_sequences("data/stock_sequences.npz", "data/stock_sequences_meta.json")

    for anchor_index, res in results.items():
        if not -1 < anchor_index < 10:
            continue

        # Extract ranks for each method
        rank_dtw = list(res["rank_dtw"])
        rank_lcss = list(res["rank_lcss"])

        # Get top 10 examples from each method
        top_10 = list(set(rank_dtw[:10] + rank_lcss[:10]))[:15]

        plt.subplots(4, 4)

        plt.subplot(4, 4, 1)
        plt.plot(sequences[anchor_index], color="blue")
        plt.axis("off")
        plt.title(f"Anchor Index: {anchor_index}")

        for i, seq_idx in enumerate(top_10):
            seq = sequences[seq_idx]
            plt.subplot(4, 4, i + 2)
            plt.plot(seq, color="green")
            plt.axis("off")
            plt.title(f"Index: {seq_idx}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
