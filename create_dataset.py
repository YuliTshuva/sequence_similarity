"""
Yuli Tshuva
Creating a dataset for labeling.
"""

# Imports
import numpy as np
import os
from os.path import join
from generate_sequences import get_stock_trend, shrink_and_stretch_sequence, permute_sequence, add_noise_to_sequence, plot_two_sequences
from utils import increase_sequence_resolution
import shutil

# Constants
LABELING_DIR = join("experimental_setup", "labeling_data")


def main():
    # Get the 11 sequences
    sequences = get_stock_trend()
    # Set the new length for all sequences
    new_length = 1000
    # Increase the resolution of all sequences
    sequences = [increase_sequence_resolution(seq, new_length) for seq in sequences]

    # Iterate through the sequences and create a folder for each one
    for idx, seq in enumerate(sequences):
        # Create a path for the sequence
        seq_path = join(LABELING_DIR, f"sample_{idx}")
        # Create the folder
        if os.path.exists(seq_path):
            shutil.rmtree(seq_path)
        os.makedirs(seq_path, exist_ok=False)

        # Save the sequence as anchor.npy
        np.save(join(seq_path, "anchor.npy"), seq)

        ## Create 9 candidate sequences
        # Add some noise to the original sequence
        noisy_seq1 = add_noise_to_sequence(seq, noise_level=0.1)
        noisy_seq2 = add_noise_to_sequence(seq, noise_level=0.25)
        # Permute the original sequence
        permuted_seq1 = permute_sequence(seq, permutation_level=0.34)
        permuted_seq2 = permute_sequence(seq, permutation_level=0.67)
        # Shrink and stretch the original sequence
        stretched_seq1 = shrink_and_stretch_sequence(seq, change_level=0.34)
        stretched_seq2 = shrink_and_stretch_sequence(seq, change_level=0.67)
        # Sample two other sequences
        other_seq1 = sequences[(idx + 1) % len(sequences)]
        other_seq2 = sequences[(idx + 2) % len(sequences)]
        # Generate a random sequence
        random_seq = np.random.rand(new_length)

        # Save the candidate sequences
        np.save(join(seq_path, "candidate1.npy"), noisy_seq1)
        np.save(join(seq_path, "candidate2.npy"), noisy_seq2)
        np.save(join(seq_path, "candidate3.npy"), permuted_seq1)
        np.save(join(seq_path, "candidate4.npy"), permuted_seq2)
        np.save(join(seq_path, "candidate5.npy"), stretched_seq1)
        np.save(join(seq_path, "candidate6.npy"), stretched_seq2)
        np.save(join(seq_path, "candidate7.npy"), other_seq1)
        np.save(join(seq_path, "candidate8.npy"), other_seq2)
        np.save(join(seq_path, "candidate9.npy"), random_seq)


if __name__ == "__main__":
    main()
