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

    # Modify the second sequence such that the 50 last values become a up trended segment
    seq2[-50:] = np.linspace(seq2[-50], seq2[-1] + 30, 50)

    # Compute distance
    distance, sigma = seq_distance(seq1, seq2)
    print(distance)
    print(sigma)

    # Create a heatmap of the mapping matrix sigma
    plt.figure(figsize=(8, 6))
    plt.imshow(sigma.cpu().detach().numpy(), cmap='cool', aspect='auto')
    plt.colorbar(label='Mapping Strength')
    plt.title('Learned Mapping Matrix (Sigma)')
    plt.xlabel('Sequence 2 Nodes')
    plt.ylabel('Sequence 1 Nodes')
    plt.show()


if __name__ == "__main__":
    main()
