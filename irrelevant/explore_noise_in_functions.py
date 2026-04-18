"""
Yuli Tshuva
Explore the noise appearing in our dataset and its effect on sequence similarity.
"""
import os

import matplotlib.pyplot as plt

# Imports
from utils import *
from seq_sim_alg import *

# Constants
DATA_DIR = "data"
rcParams['font.family'] = 'Times New Roman'


def load_a_seq_and_plot(seq_name):
    # Load a sequence
    seq_path = join(DATA_DIR, seq_name)
    seq = load_data(seq_path)

    # Iterate over the sequence with a window of size 25 and calculate the variance in each window
    window_size = 5
    variances = []
    for i in range(len(seq) - window_size + 1):
        window = seq[i:i + window_size]
        variances.append(np.var(window))

    plt.figure(figsize=(10, 5))

    # First axis (sequence)
    ax1 = plt.gca()
    ax1.plot(seq, color='royalblue', label='Sequence')
    ax1.set_xlabel("Timestep", fontsize=15)
    ax1.set_ylabel("Value", fontsize=15, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Second axis (variance)
    ax2 = ax1.twinx()
    ax2.plot(range(window_size // 2, window_size // 2 + len(variances)), variances, color='hotpink',
             label=f'Variance (window size: {window_size})')
    ax2.set_ylabel("Variance", fontsize=15, color='hotpink')
    ax2.tick_params(axis='y', labelcolor='hotpink')

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=12)

    plt.title("Example Sequence with Noise", fontsize=20)
    plt.tight_layout()
    plt.show()


def main():
    # Get all the paths in the data directory
    paths = os.listdir(DATA_DIR)
    for i in range(9, 10):
        # Load a sequence and plot it
        load_a_seq_and_plot(paths[i])


if __name__ == "__main__":
    main()
