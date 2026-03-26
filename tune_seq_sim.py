"""
Yuli Tshuva
"""

# Imports
from seq_sim_alg import *
import optuna

# Constants
DATA_DIR = "data"
DATASET_DIR = "datasets"
DATASET_NAME = "triplets_random.csv"


def tune_parameters():
    # Load the dataset
    df = pd.read_csv(join(DATASET_DIR, DATASET_NAME))
    # Filter out rows without labels
    df = df[df["first_sample_label"].notna()]
    # Get the anchor, positive, negative paths and labels
    anchors = df["anchor"].tolist()
    positives = df["first_sample"].tolist()
    negatives = df["second_sample"].tolist()

    # Define a range of parameters to tune using Optuna
    def objective(trial):
        # Sample parameters
        alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
        feature_weights = [trial.suggest_float(f"feature_{i}_weight", 0.1, 10.0, step=0.1) for i in range(9)]

        # Iterate over the triplets and compute the loss
        total_success = 0
        for anchor_path, positive_path, negative_path in tqdm(zip(anchors, positives, negatives)):
            anchor_seq = load_data(join(anchor_path.split("\\")[0], anchor_path.split("\\")[1]))
            positive_seq = load_data(join(positive_path.split("\\")[0], positive_path.split("\\")[1]))
            negative_seq = load_data(join(negative_path.split("\\")[0], negative_path.split("\\")[1]))

            # Compute distances
            pos_distance, _ = seq_distance(anchor_seq, positive_seq, alpha=alpha, feature_weights=feature_weights)
            neg_distance, _ = seq_distance(anchor_seq, negative_seq, alpha=alpha, feature_weights=feature_weights)

            # Check if the positive is closer than the negative
            if pos_distance < neg_distance:
                total_success += 1

        return total_success

    # Optimize using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Print the best parameters
    print("Best parameters:", study.best_params)
    print("Best score (number of successful triplets):", study.best_value, "/", len(anchors))


def main():
    tune_parameters()


if __name__ == "__main__":
    main()
