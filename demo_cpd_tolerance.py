"""
Yuli Tshuva
Demo: the tolerance rule (smallest K whose leftover cost is within `TOL` of the
one-line fit) on ten stock sequences not used anywhere else in the report.

Writes reports/figs/cpd_opt_tolerance_demo.png
"""

import json
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils import normalize_sequence
from cpd_opt import change_points_by_tolerance

rcParams['font.family'] = 'Times New Roman'

DATA = join("data", "six_cps_sequences.npz")
META = join("data", "six_cps_sequences_meta.json")
OUT = join("reports", "figs", "cpd_opt_tolerance_demo.png")

TOL = 0.009
USED = {0, 885, 3050, 1871, 4200, 7300}
DP_C = "#1f4e9c"


def pick(meta, n=10, seed=7):
    """Ten unseen sequences, all different tickers, spread over sectors."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(meta))
    chosen, tickers, sectors = [], set(), {}
    for i in order:
        i = int(i)
        if i in USED:
            continue
        m = meta[i]
        if m["ticker"] in tickers:
            continue
        if sectors.get(m["sector"], 0) >= 2:
            continue
        chosen.append(i)
        tickers.add(m["ticker"])
        sectors[m["sector"]] = sectors.get(m["sector"], 0) + 1
        if len(chosen) == n:
            break
    return chosen


def main():
    d = np.load(DATA)
    meta = json.load(open(META))
    idxs = pick(meta)

    fig, axes = plt.subplots(5, 2, figsize=(13, 15), sharex=True)
    for ax, idx in zip(axes.ravel(), idxs):
        y = normalize_sequence(d["seq_%d" % idx])
        b, k = change_points_by_tolerance(y, tol=TOL, return_k=True)
        inner = [int(v) for v in b[1:-1]]

        ax.plot(y, color="0.45", lw=1.4, zorder=1)
        ax.scatter(inner, y[inner], s=55, color=DP_C, zorder=3,
                   edgecolor="white", linewidth=0.8,
                   label="change points (K=%d)" % k)

        m = meta[idx]
        ax.set_title("%s (%s)  %s to %s" % (m["ticker"], m["sector"], m["start"], m["end"]),
                     fontsize=12)
        ax.legend(fontsize=10, loc="best", framealpha=0.85)
        ax.set_ylabel("normalized value", fontsize=11)
        print("%-6s idx %-5d K=%d  cuts=%s" % (m["ticker"], idx, k, inner))

    for ax in axes[-1]:
        ax.set_xlabel("time index", fontsize=11)
    fig.suptitle("Tolerance rule on ten unseen sequences "
                 "($c_K/c_0 \\leq %.3g$)" % TOL, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
