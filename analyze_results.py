import os
from os.path import join
import matplotlib.pyplot as plt
import pickle
from mine_data import load_sequences
import numpy as np
from matplotlib import rcParams
from adversarial_examples_extraction import DATA_PATH, META_PATH

rcParams["font.family"] = "Times New Roman"

index_to_sample_dir = lambda index: join("experimental_setup", "labeling_data", f"sample_{index}")

seqs, _ = load_sequences(DATA_PATH, META_PATH)
seqs = [np.interp(np.linspace(0, len(s) - 1, 1000), np.arange(len(s)), s) for s in seqs]


def read_results():
    with open(join("results", "adversarial_examples.pkl"), "rb") as f:
        results = pickle.load(f)

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
        rank_dtw = list(res["rank_dtw"])
        rank_lcss = list(res["rank_lcss"])

        # Extract classes
        class_1 = res["class_1"]
        class_2 = res["class_2"]
        class_3 = res["class_3"]
        class_4 = res["class_4"]

        # Rank the classes by our method's rank to get the top 2 examples from each class
        class_1 = sorted(class_1, key=lambda x: -rank_ours.index(x))
        class_2 = sorted(class_2, key=lambda x: -rank_ours.index(x))
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

            if anchor_index == 5:
                pass

            i, count = 0, 0
            while count < 2:
                pos = class_4[i]
                if pos in class_3[:2]:
                    class_4.remove(pos)
                    continue
                i += 1
                seq = seqs[pos]
                np.save(join(index_to_sample_dir(sample_index), f"candidate{count + 7}.npy"), seq)
                count += 1

            if sample_index <= 5:
                # Set title size
                title_size = 32
                # Plotting
                plt.subplots(3, 3, figsize=(25, 17))

                # Plot the anchor
                plt.subplot(3, 3, 1)
                plt.title(f"Anchor: {anchor_index}", fontsize=title_size)
                seq = seqs[anchor_index]
                plt.plot(range(len(seq)), seq, color="blue")
                plt.ylim(0, 1)
                plt.xticks([])
                plt.yticks([])

                # Plot positives
                for i, pos in enumerate(class_1[:2] + class_2[:2]):
                    plt.subplot(3, 3, i + 2)
                    if i < 2:
                        plt.title(f"Seq {pos} | GDTW Rank {rank_ours.index(pos)} | DTW Rank {rank_dtw.index(pos)}",
                                  fontsize=title_size)
                    else:
                        plt.title(f"Seq {pos} | GDTW Rank {rank_ours.index(pos)} | LCSS Rank {rank_lcss.index(pos)}",
                                  fontsize=title_size)

                    seq = seqs[pos]
                    plt.plot(range(len(seq)), seq, color="salmon")
                    plt.ylim(0, 1)
                    plt.xticks([])
                    plt.yticks([])

                # Plot negatives
                for i, neg in enumerate(class_3[:2] + class_4[:2]):
                    plt.subplot(3, 3, i + 6)
                    if i < 2:
                        plt.title(f"Seq {neg} | GDTW Rank {rank_ours.index(neg)} | DTW Rank {rank_dtw.index(neg)}",
                                  fontsize=title_size)
                    else:
                        plt.title(f"Seq {neg} | GDTW Rank {rank_ours.index(neg)} | LCSS Rank {rank_lcss.index(neg)}",
                                  fontsize=title_size)
                    seq = seqs[neg]
                    plt.plot(range(len(seq)), seq, color="turquoise")
                    plt.ylim(0, 1)
                    plt.xticks([])
                    plt.yticks([])

                plt.tight_layout()
                plt.show()

            sample_index += 1

    print(f'Anchors to use ({len(anchors_to_use)}):', anchors_to_use)


if __name__ == "__main__":
    read_results()
