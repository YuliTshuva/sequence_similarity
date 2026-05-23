import os
from os.path import join
import matplotlib.pyplot as plt
import pickle
from mine_data import load_sequences
import numpy as np

index_to_sample_dir = lambda index: join("experimental_setup", "labeling_data", f"sample_{index}")

seqs, _ = load_sequences()
seqs = [np.interp(np.linspace(0, len(s) - 1, 1000), np.arange(len(s)), s) for s in seqs]


def read_results():
    results_files = [join("results", f) for f in os.listdir("results") if f.startswith("baby_adversarial_examples")]
    results = {}
    for file in results_files:
        with open(file, "rb") as f:
            file_results = pickle.load(f)
            results.update(file_results)

    # Set list to store anchors to use for labeling
    anchors_to_use = []
    # Set a counter for storage
    sample_index = 0
    for anchor_index, res in results.items():
        # Stop after 50 samples
        if sample_index == 50:
            break

        # Extract ranks for each method
        rank_ours = list(res["rank_ours"])

        # Extract classes
        class_1 = res["class_1"]
        class_2 = res["class_2"]
        class_3 = res["class_3"]
        class_4 = res["class_4"]

        # Rank the classes by our method's rank to get the top 2 examples from each class
        class_1 = sorted(class_1, key=lambda x: rank_ours.index(x))
        class_2 = sorted(class_2, key=lambda x: rank_ours.index(x))
        class_3 = sorted(class_3, key=lambda x: rank_ours.index(x))
        class_4 = sorted(class_4, key=lambda x: rank_ours.index(x))

        # Check if there's at least 2 examples in each class
        if len(class_1) >= 2 and len(class_2) >= 4 and len(class_3) >= 2 and len(class_4) >= 4:
            anchors_to_use.append(anchor_index)

            os.makedirs(index_to_sample_dir(sample_index), exist_ok=True)

            seq = seqs[anchor_index]
            np.save(join(index_to_sample_dir(sample_index), "anchor.npy"), seq)

            for i, pos in enumerate(class_1[:2]):
                seq = seqs[pos]
                np.save(join(index_to_sample_dir(sample_index), f"candidate{i + 1}.npy"), seq)

            i, count = 0, 0
            while count < 2:
                pos = class_2[i]
                i += 1
                if pos in class_1[:2]:
                    class_2.remove(pos)
                    continue
                seq = seqs[pos]
                np.save(join(index_to_sample_dir(sample_index), f"candidate{count + 3}.npy"), seq)
                count += 1

            for i, neg in enumerate(class_3[:2]):
                seq = seqs[neg]
                np.save(join(index_to_sample_dir(sample_index), f"candidate{i + 5}.npy"), seq)

            i, count = 0, 0
            while count < 2:
                pos = class_4[i]
                i += 1
                if pos in class_3[:2]:
                    class_4.remove(pos)
                    continue
                seq = seqs[pos]
                np.save(join(index_to_sample_dir(sample_index), f"candidate{count + 7}.npy"), seq)
                count += 1

            if sample_index == 10:
                # Plotting
                plt.subplots(4, 3, figsize=(30, 20))

                # Plot the anchor
                plt.subplot(4, 3, 2)
                plt.title(f"Anchor: {anchor_index}")
                seq = seqs[anchor_index]
                plt.plot(range(len(seq)), seq, color="blue")
                plt.ylim(0, 1)

                # Plot positives
                for i, pos in enumerate(class_1[:2] + class_2[:2]):
                    plt.subplot(4, 3, i + 4)
                    plt.title(f"Positive {i + 1}: {pos}")
                    seq = seqs[pos]
                    plt.plot(range(len(seq)), seq, color="salmon")
                    plt.ylim(0, 1)

                # Plot negatives
                for i, neg in enumerate(class_3[:2] + class_4[:2]):
                    plt.subplot(4, 3, i + 8)
                    plt.title(f"Negative {i + 1}: {neg}")
                    seq = seqs[neg]
                    plt.plot(range(len(seq)), seq, color="turquoise")
                    plt.ylim(0, 1)

                plt.tight_layout()
                plt.show()

            sample_index += 1

    print(f'Anchors to use ({len(anchors_to_use)}):', anchors_to_use)


if __name__ == "__main__":
    read_results()
