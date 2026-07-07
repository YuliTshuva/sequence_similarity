import os
from os.path import join
import matplotlib.pyplot as plt
import pickle
from mine_data import load_sequences
import numpy as np
from matplotlib import rcParams
import argparse

rcParams["font.family"] = "Times New Roman"

index_to_sample_dir = lambda index: join("experimental_setup", "labeling_data", f"sample_{index}")


def read_results(dataset_prefix, top_k):
    data_path = join("data", f"{dataset_prefix}_cps_sequences.npz")
    meta_path = join("data", f"{dataset_prefix}_cps_sequences_meta.json")

    seqs, _ = load_sequences(data_path, meta_path)
    seqs = [np.interp(np.linspace(0, len(s) - 1, 1000), np.arange(len(s)), s) for s in seqs]

    with open(join("results", f"{dataset_prefix}_cps_adversarial_examples.pkl"), "rb") as f:
        results = pickle.load(f)

    print(f"Loaded {len(results)} anchors.")

    # First pass: score each anchor by how strongly the two methods disagree
    scored = []
    for anchor_index, res in results.items():
        # Extract ranks for each method (most similar first)
        rank_ours = list(res["rank_ours"])
        rank_dtw = list(res["rank_dtw"])

        # Top 10 of each method, excluding the anchor itself
        top_ours = [s for s in rank_ours if s != anchor_index][:10]
        top_dtw = [s for s in rank_dtw if s != anchor_index][:10]

        # In our top 10, the candidate DTW ranks lowest
        dtw_worst = max(top_ours, key=lambda s: rank_dtw.index(s))
        # In DTW's top 10, the candidate our method ranks lowest
        ours_worst = max(top_dtw, key=lambda s: rank_ours.index(s))

        # Disagreement: how badly each method ranks the other's pick
        disagreement = rank_dtw.index(dtw_worst) + rank_ours.index(ours_worst)
        scored.append((disagreement, anchor_index, rank_ours, rank_dtw, dtw_worst, ours_worst))

    # Keep only the top_k anchors with the highest disagreement
    scored.sort(key=lambda t: t[0], reverse=True)
    scored = scored[:top_k]
    print(f"Keeping {len(scored)} anchors with the highest disagreement.")

    # Second pass: save and plot the selected triplets
    sample_index = 0
    for disagreement, anchor_index, rank_ours, rank_dtw, dtw_worst, ours_worst in scored:
        # Save the triplet for labeling
        os.makedirs(index_to_sample_dir(sample_index), exist_ok=True)
        np.save(join(index_to_sample_dir(sample_index), "anchor.npy"), seqs[anchor_index])
        np.save(join(index_to_sample_dir(sample_index), "candidate1.npy"), seqs[dtw_worst])
        np.save(join(index_to_sample_dir(sample_index), "candidate2.npy"), seqs[ours_worst])

        # Set title size
        title_size = 32
        # Plotting
        plt.subplots(1, 3, figsize=(25, 8))

        # Plot the anchor
        plt.subplot(1, 3, 1)
        plt.title(f"Anchor: {anchor_index}", fontsize=title_size)
        seq = seqs[anchor_index]
        plt.plot(range(len(seq)), seq, color="blue")
        plt.ylim(0, 1)
        plt.xticks([])
        plt.yticks([])

        # Plot the candidate in our top 10 that DTW ranks lowest
        plt.subplot(1, 3, 2)
        plt.title(f"Seq {dtw_worst} | GDTW Rank {rank_ours.index(dtw_worst)} | DTW Rank {rank_dtw.index(dtw_worst)}",
                  fontsize=title_size)
        seq = seqs[dtw_worst]
        plt.plot(range(len(seq)), seq, color="salmon")
        plt.ylim(0, 1)
        plt.xticks([])
        plt.yticks([])

        # Plot the candidate in DTW's top 10 that our method ranks lowest
        plt.subplot(1, 3, 3)
        plt.title(f"Seq {ours_worst} | GDTW Rank {rank_ours.index(ours_worst)} | DTW Rank {rank_dtw.index(ours_worst)}",
                  fontsize=title_size)
        seq = seqs[ours_worst]
        plt.plot(range(len(seq)), seq, color="turquoise")
        plt.ylim(0, 1)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        os.makedirs(join("results", f"{dataset_prefix}_triplets"), exist_ok=True)
        plt.savefig(join("results", f"{dataset_prefix}_triplets", f"sample_{anchor_index}.pdf"))
        plt.close()

        sample_index += 1

    print(f"Saved {sample_index} triplets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_prefix", default="six", help="Dataset prefix (e.g. six / ten_plus)")
    parser.add_argument("--top_k", type=int, default=100, help="Number of highest-disagreement anchors to save")
    args = parser.parse_args()
    read_results(args.dataset_prefix, args.top_k)
