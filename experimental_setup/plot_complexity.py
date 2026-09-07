"""
Yuli Tshuva
Show what each complexity measure actually looks like on the mined sequences.

One row per measure; across the row, real curves drawn from the 5th, 25th, 50th,
75th and 95th percentile of that measure. Reading a row left to right shows what
"too simple" and "too busy" mean for that measure, so the keep-band can be chosen
by eye rather than guessed.
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from complexity import measure_all, _prepare

PERCENTILES = [5, 25, 50, 75, 95]


def plot_complexity(sequences, out_path, seed=0, n_sample=1500):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sequences), size=min(n_sample, len(sequences)), replace=False)
    sample = [sequences[i] for i in idx]
    scores = measure_all(sample)

    names = list(scores)
    fig, axes = plt.subplots(len(names), len(PERCENTILES),
                             figsize=(3 * len(PERCENTILES), 2.4 * len(names)))

    for row, name in enumerate(names):
        values = scores[name]
        for col, pct in enumerate(PERCENTILES):
            target = np.percentile(values, pct)
            pick = int(np.abs(values - target).argmin())
            ax = axes[row, col]
            ax.plot(_prepare(sample[pick]), lw=1.5, color="#1a1a1a")
            shown = (f"{values[pick]:.0f}" if name == "turning_points"
                     else f"{values[pick]:.2f}")
            ax.set_title(f"p{pct}   {name} = {shown}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(name, fontsize=10)

    fig.suptitle("What each complexity measure looks like on the mined sequences",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}\n")

    for name in names:
        v = scores[name]
        qs = np.percentile(v, PERCENTILES)
        print(f"{name:>15}  " + "  ".join(f"p{p}={q:.2f}" for p, q in zip(PERCENTILES, qs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--out", default="reports/figs/sequence_complexity.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.sequences, allow_pickle=False)
    sequences = [data[f"seq_{i}"] for i in range(len(data.files))]
    print(f"Loaded {len(sequences)} sequences from {args.sequences}")
    plot_complexity(sequences, args.out, args.seed)


if __name__ == "__main__":
    main()
