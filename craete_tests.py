"""
Creating test files for sequence similarity algorithms.
"""

from os.path import join
import pickle
from mine_data import load_sequences
from utils import *

# Set test directory
TEST_DIR = join("data", "tests")


def expand_sequence(seq):
    if seq[0] < 0.5:
        segment = np.arange(seq[0] + 0.3, seq[0], -0.03)
    else:
        segment = np.arange(seq[0] - 0.3, seq[0], 0.03)

    seq1 = np.concatenate([segment, seq])

    if seq[-1] < 0.5:
        segment1 = np.arange(seq[-1], seq[-1] + 0.3, 0.03)
        segment2 = np.arange(seq[-1] + 0.3, seq[-1] + 0.5, 0.02)
        segment3 = np.arange(seq[-1] + 0.5, seq[-1] + 0.1, -0.035)
    else:
        segment1 = np.arange(seq[-1], seq[-1] - 0.3, -0.03)
        segment2 = np.arange(seq[-1] - 0.3, seq[-1] - 0.5, -0.02)
        segment3 = np.arange(seq[-1] - 0.5, seq[-1] - 0.1, 0.035)

    seq2 = np.concatenate([seq1, segment1])
    seq3 = np.concatenate([seq2, segment2])
    seq4 = np.concatenate([seq3, segment3])

    return seq1, seq2, seq3, seq4


def expand_partition(seq):
    # Detect change points
    change_ps = change_points_detection(seq)

    # Find the longest segment
    longest_segment_length = 0
    longest_segment_index = 0
    for i in range(len(change_ps) - 1):
        segment_length = change_ps[i + 1] - change_ps[i]
        if segment_length > longest_segment_length:
            longest_segment_length = segment_length
            longest_segment_index = i

    # Find the slope of the longest segment
    start, end = change_ps[longest_segment_index], change_ps[longest_segment_index + 1]
    slope = (seq[end] - seq[start]) / (end - start)

    # Insert a plateau in the middle of the longest segment
    plateau_length = longest_segment_length // 3
    plateau_start = start + longest_segment_length // 2

    seq1 = np.array(list(seq[:plateau_start + 1]) +
                    list(np.arange(seq[plateau_start], seq[plateau_start] + slope * plateau_length, slope))
                    + list(seq[plateau_start + 1:] + slope * plateau_length))

    return seq1


def main():
    seqs, _ = load_sequences()
    seqs = [normalize_sequence(seq) for seq in seqs]

    n_samples = 10

    for i in range(n_samples):
        index = np.random.randint(0, len(seqs))
        index2 = np.random.randint(0, len(seqs))
        seq = seqs[index]
        seq1, seq2, seq3, seq4 = expand_sequence(seq)

        # Make the sequence of lenght 1000
        seq = np.interp(np.linspace(0, len(seq) - 1, 1000), np.arange(len(seq)), seq)
        seq1 = np.interp(np.linspace(0, len(seq1) - 1, 1000), np.arange(len(seq1)), seq1)
        seq2 = np.interp(np.linspace(0, len(seq2) - 1, 1000), np.arange(len(seq2)), seq2)
        seq3 = np.interp(np.linspace(0, len(seq3) - 1, 1000), np.arange(len(seq3)), seq3)
        seq4 = np.interp(np.linspace(0, len(seq4) - 1, 1000), np.arange(len(seq4)), seq4)

        seq5 = expand_partition(seq)
        seq5 = np.interp(np.linspace(0, len(seq5) - 1, 1000), np.arange(len(seq5)), seq5)
        seq6 = expand_partition(seq5)
        seq6 = np.interp(np.linspace(0, len(seq6) - 1, 1000), np.arange(len(seq6)), seq6)
        seq7 = expand_partition(seq6)
        seq7 = np.interp(np.linspace(0, len(seq7) - 1, 1000), np.arange(len(seq7)), seq7)
        seq8 = seqs[index2]
        seq8 = np.interp(np.linspace(0, len(seq8) - 1, 1000), np.arange(len(seq8)), seq8)

        # Save the sequences
        with open(join(TEST_DIR, f"seq_{index}.pkl"), "wb") as f:
            pickle.dump([seq, seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8], f)


if __name__ == "__main__":
    main()
