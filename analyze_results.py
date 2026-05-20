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
    results_files = [join("results", f) for f in os.listdir("results") if f.startswith("adversarial_examples")]
    results = {}
    for file in results_files:
        with open(file, "rb") as f:
            file_results = pickle.load(f)
            results.update(file_results)

    anchors_to_use = []
    sample_index = 0
    count = 0
    for anchor_index, res in results.items():
        examples_in_top = set(res["top_dtw"]).intersection(set(res['adversarial_examples']))
        examples_in_bottom = set(res["bottom_dtw"]).intersection(set(res['adversarial_examples']))
        if len(examples_in_top) >= 6 and len(examples_in_bottom) >= 3:
            count += 1
        if len(examples_in_top) >= 6 and len(examples_in_bottom) >= 3:
            anchors_to_use.append(anchor_index)

            os.makedirs(index_to_sample_dir(sample_index), exist_ok=True)

            seq = seqs[anchor_index]
            np.save(join(index_to_sample_dir(sample_index), "anchor.npy"), seq)

            for i, pos in enumerate(list(examples_in_bottom)[:3]):
                seq = seqs[pos]
                np.save(join(index_to_sample_dir(sample_index), f"candidate_{i + 1}.npy"), seq)

            for i, neg in enumerate(list(examples_in_top)[:6]):
                seq = seqs[neg]
                np.save(join(index_to_sample_dir(sample_index), f"candidate_{i + 4}.npy"), seq)

            sample_index += 1

            if anchor_index == 16 and False:
                # Plotting
                plt.subplots(4, 3, figsize=(15, 20))

                # Plot the anchor
                plt.subplot(4, 3, 2)
                plt.title(f"Anchor: {anchor_index}")
                seq = seqs[anchor_index]
                plt.plot(range(len(seq)), seq, color="blue")
                plt.ylim(0, 1)

                # Plot positives
                for i, pos in enumerate(list(examples_in_bottom)[:3]):
                    plt.subplot(4, 3, i + 4)
                    plt.title(f"Positive {i + 1}: {pos}")
                    seq = seqs[pos]
                    plt.plot(range(len(seq)), seq, color="turquoise")
                    plt.ylim(0, 1)

                # Plot negatives
                for i, neg in enumerate(list(examples_in_top)[:6]):
                    plt.subplot(4, 3, i + 7)
                    plt.title(f"Negative {i + 1}: {neg}")
                    seq = seqs[neg]
                    plt.plot(range(len(seq)), seq, color="salmon")
                    plt.ylim(0, 1)

                plt.tight_layout()
                plt.show()

    print(f'Anchors to use ({len(anchors_to_use)}):', anchors_to_use)

    print("count:", count)


if __name__ == "__main__":
    read_results()
