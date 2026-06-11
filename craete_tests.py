"""
Creating test files for sequence similarity algorithms.
"""
import pickle
import random
import os
from os.path import join
from compare_baselines import euclidean_distance, pearson_distance, dtw_distance
from mine_data import load_sequences
import matplotlib.pyplot as plt

# Set test directory
TEST_DIR = join("data", "tests")
TAGS = {
    47: [6614, 1312, 3697],
    54: [4047, 6315, 5052],
    69: [4560, 2927, 4670],
    76: [7437, 1603, 1888],
    79: [961, 6378, 5224],
    86: [648, 5103, 3899],
    87: [5749, 1192, 2144],
    92: [4815, 1071, 6861],
    115: [60, 5165, 3238],
    138: [6589, 9492, 4954]
}
TOP_N = 8
N_NEGS = 6


def find_closest_by_euclidean(sequences, query_idx, top_n=TOP_N):
    query = sequences[query_idx]
    distances = []
    for i, seq in enumerate(sequences):
        if i == query_idx:
            continue
        dist = euclidean_distance(query, seq)
        distances.append((dist, i))
    distances.sort(key=lambda x: x[0])
    return distances[:top_n]


def plot_closest_sequences(sequences, query_idx, closest):
    query = sequences[query_idx]
    n = len(closest)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)

    axes[0, 0].plot(query, color='royalblue')
    axes[0, 0].set_title(f"Query (idx={query_idx})", fontsize=13)
    axes[0, 0].set_xlabel("Timestep", fontsize=11)
    axes[0, 0].set_ylabel("Value", fontsize=11)

    for plot_i, (dist, seq_idx) in enumerate(closest):
        ax = axes[(plot_i + 1) // 3, (plot_i + 1) % 3]
        ax.plot(sequences[seq_idx], color='hotpink')
        ax.set_title(f"#{plot_i + 1}  idx={seq_idx}\nd={dist:.4f}", fontsize=13)
        ax.set_xlabel("Timestep", fontsize=11)

    plt.suptitle(f"Top-{len(closest)} closest sequences to query idx={query_idx} (Euclidean)", fontsize=15)
    plt.tight_layout()
    plt.show()


def find_anchor_positives_matches(sequences):
    anchor = max(TAGS) + 20
    for _ in range(15):
        # Find its closest seqs by euclidean distance and plot them
        closest = find_closest_by_euclidean(sequences, query_idx=anchor, top_n=TOP_N)
        plot_closest_sequences(sequences, query_idx=anchor, closest=closest)
        anchor += 1


def create_tests(sequences):
    # Find the amount of sequences
    n = len(sequences)

    tests = []
    for anchor in TAGS:
        positives = TAGS[anchor]
        negs = random.sample(range(len(sequences)), k=N_NEGS)
        tests.append([anchor, positives, negs])

    # Make sure the test directory exists
    os.makedirs(TEST_DIR, exist_ok=True)

    # Save tests to a pickle file
    with open(join(TEST_DIR, "tests.pkl"), "wb") as f:
        pickle.dump(tests, f)


def main():
    # Load sequences
    sequences, _ = load_sequences("data/stock_sequences.npz", "data/stock_sequences_meta.json")

    # Create tests
    create_tests(sequences)


if __name__ == "__main__":
    main()
