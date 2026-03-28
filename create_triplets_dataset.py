"""
Yuli Tshuva
Creating a dataset of triplets (anchor, positive, negative) for tuning our model's parameters.
"""

# Imports
import os
from seq_sim_alg import seq_distance
import matplotlib.pyplot as plt
from utils import *

# Constants
DATA_DIR = "data"
DATASET_DIR = "datasets"
DATASET_NAME = "triplets_random.csv"
N_TRIPLETS = 100


def create_triplets_randomly():
    """
    Creates triplets by randomly sampling sequences.
    """
    # Set a list to hold the triplets
    triplets = []
    # Get the dataset in paths format
    paths = [join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    # We will create N_TRIPLETS triplets for tuning
    for _ in range(N_TRIPLETS):
        # Randomly sample three different indices
        idxs = np.random.choice(len(paths), size=3, replace=False)
        anchor_idx, positive_idx, negative_idx = idxs
        # Append the triplet to our list
        triplets.append((paths[anchor_idx], paths[positive_idx], paths[negative_idx]))

    # Save the triplets to a csv
    with open(join(DATASET_DIR, DATASET_NAME), "w") as f:
        f.write("anchor,first_sample,second_sample,first_sample_label\n")
        for anchor, positive, negative in triplets:
            f.write(f"{anchor},{positive},{negative},{np.nan}\n")


def label_dataset():
    # Load the dataset
    df = pd.read_csv(join(DATASET_DIR, DATASET_NAME))

    try:
        # Iterate through the rows and label the first sample manually after plotting
        for idx, row in df.iterrows():
            # Get the paths for anchor, positive, and negative
            anchor_path = row["anchor"]
            positive_path = row["first_sample"]
            negative_path = row["second_sample"]
            label = row.get("first_sample_label", None)
            if pd.notna(label):
                continue  # Skip already labeled rows

            # Load the sequences
            anchor_seq = load_data(anchor_path)
            positive_seq = load_data(positive_path)
            negative_seq = load_data(negative_path)

            # Plot the sequences to visually inspect them
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(anchor_seq, color="turquoise")
            plt.title("Anchor")
            plt.subplot(1, 3, 2)
            plt.plot(positive_seq, color="dodgerblue")
            plt.title("First Sample")
            plt.subplot(1, 3, 3)
            plt.plot(negative_seq, color="hotpink")
            plt.title("Second Sample")
            plt.tight_layout()
            plt.show()

            # Manually label the first sample based on visual inspection
            label = input(f"[{idx}] Which sample is more similar to the anchor? (1/2): ")
            if label == "1":
                df.at[idx, "first_sample_label"] = 1
            elif label == "2":
                df.at[idx, "first_sample_label"] = 0
            else:
                print("Invalid input. Skipping this triplet.")
                continue
    except Exception as e:
        pass
    finally:
        # Save the updated dataset with labels
        df.to_csv(join(DATASET_DIR, DATASET_NAME), index=False)


def plot_losses(f_loss, s_loss, scheduler_steps, title=""):
    plt.figure(figsize=(10, 5))
    plt.plot(f_loss, color='hotpink', label='Feature Distance')
    plt.plot(s_loss, color='darksalmon', label='Structure Distance')
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Value", fontsize=15)
    plt.xticks([0] + scheduler_steps)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def explore_distance_assignment():
    # Get the dataset in paths format
    paths = [join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    # We will create N_TRIPLETS triplets for tuning
    for _ in range(1):
        # Randomly sample three different indices
        idxs = np.random.choice(len(paths), size=3, replace=False)

        # Load the sequences
        anchor_seq = load_data(paths[idxs[0]])
        positive_seq = load_data(paths[idxs[1]])
        negative_seq = load_data(paths[idxs[2]])

        # Calculate distances
        alpha = 0.1
        features = [1.0] * 6 + [5] * 3
        pos_distance, pos_mapping, f_loss1, s_loss1, sch_steps1 = seq_distance(anchor_seq, positive_seq, alpha=alpha,
                                                                               feature_weights=features, save_loss=True)
        neg_distance, neg_mapping, f_loss2, s_loss2, sch_steps2 = seq_distance(anchor_seq, negative_seq, alpha=alpha,
                                                                               feature_weights=features, save_loss=True)

        # Plot the loss curves for the positive pair
        plot_losses(f_loss1, s_loss1, sch_steps1, title="Feature Distance Loss (Anchor to First Sample)")
        plot_losses(f_loss2, s_loss2, sch_steps2, title="Feature Distance Loss (Anchor to First Sample)")

        # Annotate the choice
        annotate_mapping(anchor_seq, positive_seq, pos_mapping, title="Anchor to First Sample")
        annotate_mapping(anchor_seq, negative_seq, neg_mapping, title="Anchor to Second Sample")
        print(f"Positive distance: {pos_distance:.4f}, Negative distance: {neg_distance:.4f}")


def main():
    explore_distance_assignment()


if __name__ == "__main__":
    main()
