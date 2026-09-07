"""
Yuli Tshuva
Draw sampled triplets so the disagreement can be eyeballed.

One row per trial: the query, then the two candidates, with each baseline's vote
marked under the candidate it chose. These are the images a rater would judge.
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sample_triplets import resample, znorm, DEFAULT_RESAMPLE_LEN

QUERY_COLOUR = "#1a1a1a"
CANDIDATE_COLOUR = "#b0b0b0"
CHOSEN_COLOUR = "#c2453a"


def plot_triplets(triplets, sequences, out_path, per_class=1, resample_len=DEFAULT_RESAMPLE_LEN):
    """Draw `per_class` trials from each disagreement class."""
    picked, seen = [], {}
    for trial in triplets:
        if seen.get(trial["class"], 0) < per_class:
            picked.append(trial)
            seen[trial["class"]] = seen.get(trial["class"], 0) + 1

    fig, axes = plt.subplots(len(picked), 3, figsize=(11, 2.5 * len(picked)))
    axes = np.atleast_2d(axes)

    for row, trial in enumerate(picked):
        names = list(trial["votes"])
        for col, (role, key) in enumerate((("query", "query"),
                                           ("candidate A", "candidate_a"),
                                           ("candidate B", "candidate_b"))):
            ax = axes[row, col]
            curve = znorm(resample(sequences[trial[key]], resample_len))

            side = role[-1].lower()  # 'a' or 'b' for candidates, 'y' for the query
            voters = [n for n in names if trial["votes"][n] == side]
            is_query = key == "query"
            ax.plot(curve, lw=1.6,
                    color=QUERY_COLOUR if is_query else
                          (CHOSEN_COLOUR if voters else CANDIDATE_COLOUR))

            if is_query:
                title = f"query  (seq {trial[key]})"
            else:
                votes = ", ".join(f"{n} {trial['margins'][n]:+.2f}" for n in voters)
                title = f"{role}  (seq {trial[key]})\n{votes if voters else 'no votes'}"
            ax.set_title(title, fontsize=9,
                         color=QUERY_COLOUR if is_query or voters else CANDIDATE_COLOUR)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row, 0].set_ylabel(trial["class"].replace("_", " "), fontsize=9)

    fig.suptitle("Sampled trials: which candidate is more similar to the query?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", default="experimental_setup/triplets.json")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--out", default="reports/figs/sampled_triplets.png")
    parser.add_argument("--per-class", type=int, default=1)
    args = parser.parse_args()

    triplets = json.load(open(args.triplets))
    data = np.load(args.sequences, allow_pickle=False)
    sequences = [data[f"seq_{i}"] for i in range(len(data.files))]
    plot_triplets(triplets, sequences, args.out, args.per_class)


if __name__ == "__main__":
    main()
