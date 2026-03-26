"""
Yuli Tshuva
Explore sequence similarity.
"""

# Imports
from seq_sim_alg import *


def main():
    # Load a random sequence from the database
    seq1 = load_data("data/Atkinson_cycle_44.csv")
    seq2 = seq1.copy()
    # Plot the sequence
    plot_two_sequences(seq1, seq2, suptitle="Random Sequence from Database")
    # Compute distance
    distance, sigma = seq_distance(seq1, seq2)
    print(distance)
    print(sigma)



if __name__ == "__main__":
    main()
