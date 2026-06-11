"""
Creating test files for sequence similarity algorithms.
"""
import pickle
import random
import os
from os.path import join
from compare_baselines import euclidean_distance
import numpy as np
from mine_data import load_sequences
import matplotlib.pyplot as plt
from seq_sim_alg import seq_distance

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


def generate_relation_example():
    """
    Check the relational modeling of our method by the example Dafna gave in the lab.
    """
    # The sequences from the lab
    seq_a = [183, 220, 235, 247, 251, 255, 258, 260, 261, 263, 264, 265, 266, 267, 268, 268, 267, 264,
             258, 251, 241, 213, 177, 154, 136, 133, 131, 131, 131, 131, 130, 130, 130, 130, 136, 145,
             157, 177, 188, 198, 202, 205, 207, 204, 197, 188, 159, 133, 131, 131, 132, 133, 136, 140,
             146, 149, 155, 161, 165, 170, 171, 160, 143, 140, 140, 140, 140, 140, 140, 141]

    seq_b = [210, 235, 248, 261, 272, 285, 299, 308, 317, 324, 330, 334, 336, 338, 338, 335, 328, 314,
             222, 214, 206, 203, 198, 196, 195, 195, 195, 195, 195, 195, 199, 219, 257, 266, 271, 273,
             274, 274, 269, 233, 213, 210, 208, 205, 203, 200, 199, 196, 192]

    seq_c = [52, 63, 77, 93, 112, 127, 141, 152, 163, 172, 174, 172, 170, 166, 155, 128, 108, 85,
             68, 63, 61, 60, 60, 60, 60, 60, 62, 63, 66, 68, 74, 81, 88, 108, 140, 198,
             225, 262, 279, 289, 293, 294, 294, 294, 293, 290, 287, 281, 278, 269, 259, 245, 227, 201,
             167, 141, 84, 74, 66, 62, 60, 54, 51, 49, 47, 47]

    # Normalize them to [0, 1]
    seq_a = (np.array(seq_a) - min(seq_a)) / (max(seq_a) - min(seq_a))
    seq_b = (np.array(seq_b) - min(seq_b)) / (max(seq_b) - min(seq_b))
    seq_c = (np.array(seq_c) - min(seq_c)) / (max(seq_c) - min(seq_c))

    # Interpolate them to length 1000
    seq_a = np.interp(np.linspace(0, len(seq_a) - 1, 1000), np.arange(len(seq_a)), seq_a)
    seq_b = np.interp(np.linspace(0, len(seq_b) - 1, 1000), np.arange(len(seq_b)), seq_b)
    seq_c = np.interp(np.linspace(0, len(seq_c) - 1, 1000), np.arange(len(seq_c)), seq_c)

    # Compute their pairwise distances
    distance_ab, path_ab = seq_distance(seq_a, seq_b)
    distance_ac, path_ac = seq_distance(seq_a, seq_c)
    distance_bc, path_bc = seq_distance(seq_b, seq_c)

    # Plot them side by side
    suptitle_size, title_size, label_size = 30, 20, 15
    plt.subplots(1, 3, figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(seq_a, color='royalblue')
    plt.title(f"Sequence A\nd(B, C)={distance_bc:.3f}", fontsize=title_size)
    plt.xlabel("Timestep", fontsize=label_size)
    plt.ylabel("Value", fontsize=label_size)
    plt.subplot(1, 3, 2)
    plt.plot(seq_b, color='hotpink')
    plt.title(f"Sequence B\nd(A, B)={distance_ab:.3f}", fontsize=title_size)
    plt.xlabel("Timestep", fontsize=label_size)
    plt.subplot(1, 3, 3)
    plt.plot(seq_c, color='seagreen')
    plt.title(f"Sequence C\nd(A,C)={distance_ac:.3f}", fontsize=title_size)
    plt.xlabel("Timestep", fontsize=label_size)
    plt.suptitle("Relation Rolling Example", fontsize=suptitle_size)
    plt.tight_layout()
    plt.show()


def main():
    generate_relation_example()


if __name__ == '__main__':
    main()