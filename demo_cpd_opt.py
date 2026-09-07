"""
Yuli Tshuva
Demo: heuristic change point detection vs. the exact dynamic program in cpd_opt.

Figure 1  reports/figs/cpd_opt_examples.png   -- six stocks, both segmentations
                                                 with their piecewise-linear fits.
Figure 2  reports/figs/cpd_opt_cost_vs_k.png  -- cost as a function of K, with the
                                                 heuristic's operating point marked.
Figure 3  reports/figs/cpd_opt_beta_sweep.png -- effect of the single penalty knob.
"""

import json
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils import change_points_detection, normalize_sequence
from cpd_opt import optimal_change_points, LinearCost

rcParams['font.family'] = 'Times New Roman'

DATA = join("data", "six_cps_sequences.npz")
META = join("data", "six_cps_sequences_meta.json")
FIGS = join("reports", "figs")

EXAMPLES = [0, 885, 3050, 1871, 4200, 7300]
BETA_REL = 0.01

HEUR_C = "#c0392b"
DP_C = "#1f4e9c"


def load():
    d = np.load(DATA)
    meta = json.load(open(META))
    return d, meta


def piecewise_linear_fit(y, bounds):
    """Least-squares line per segment, returned as a full-length curve."""
    out = np.full(len(y), np.nan)
    for a, b in zip(bounds[:-1], bounds[1:]):
        b = b + 1 if b == len(y) - 1 else b
        seg = y[a:b]
        if len(seg) < 2:
            out[a:b] = seg
            continue
        x = np.arange(len(seg), dtype=float)
        coef = np.polyfit(x, seg, 1)
        out[a:b] = np.polyval(coef, x)
    return out


def sse(y, bounds):
    fit = piecewise_linear_fit(y, bounds)
    return float(np.nansum((y - fit) ** 2))


def fig_examples(d, meta):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    rows = []
    for ax, idx in zip(axes.ravel(), EXAMPLES):
        y = normalize_sequence(d["seq_%d" % idx])
        old = change_points_detection(y)
        new = optimal_change_points(y, cost="linear", beta_rel=BETA_REL)
        new = [int(v) for v in new]

        ax.plot(y, color="0.45", lw=1.4, zorder=1)
        inner = new[1:-1]
        ax.scatter(inner, y[inner], s=55, color=DP_C, zorder=3,
                   edgecolor="white", linewidth=0.8,
                   label="change points (K=%d)" % len(inner))

        m = meta[idx]
        ax.set_title("%s (%s)  %s to %s" % (m["ticker"], m["sector"], m["start"], m["end"]),
                     fontsize=12)
        ax.legend(fontsize=10, loc="best", framealpha=0.85)
        ax.set_ylabel("normalized value", fontsize=11)

        # Same-K comparison: what the DP achieves with the heuristic's budget.
        k_old = len(old) - 2
        matched = [int(v) for v in optimal_change_points(y, cost="linear", n_bkps=k_old)]
        rows.append((m["ticker"], k_old, sse(y, old), sse(y, matched), len(new) - 2, sse(y, new)))

    for ax in axes[-1]:
        ax.set_xlabel("time index", fontsize=11)
    fig.suptitle("Change points from the optimal segmentation "
                 "($\\beta_{rel}$ = %.3g)" % BETA_REL,
                 fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(join(FIGS, "cpd_opt_examples.png"), dpi=160)
    plt.close(fig)
    return rows


def fig_cost_vs_k(d, meta):
    ks = list(range(1, 13))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, idx in zip(axes, EXAMPLES[:3]):
        y = normalize_sequence(d["seq_%d" % idx])
        costs = []
        for k in ks:
            b = [int(v) for v in optimal_change_points(y, cost="linear", n_bkps=k)]
            costs.append(sse(y, b))
        old = change_points_detection(y)
        ax.plot(ks, costs, "o-", color=DP_C, ms=4, label="optimal DP")
        ax.plot([len(old) - 2], [sse(y, old)], "X", color=HEUR_C, ms=11,
                label="heuristic", zorder=5)
        ax.set_title(meta[idx]["ticker"], fontsize=13)
        ax.set_xlabel("number of change points K", fontsize=11)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=12)
    axes[0].set_ylabel("piecewise-linear SSE", fontsize=11)
    fig.suptitle("For any budget K the DP attains the minimum achievable cost", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(join(FIGS, "cpd_opt_cost_vs_k.png"), dpi=160)
    plt.close(fig)


def fig_beta_sweep(d, meta):
    betas = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    idx = EXAMPLES[2]
    y = normalize_sequence(d["seq_%d" % idx])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, br in zip(axes.ravel(), betas):
        b = [int(v) for v in optimal_change_points(y, cost="linear", beta_rel=br)]
        ax.plot(y, color="0.75", lw=1.1)
        ax.plot(piecewise_linear_fit(y, b), color=DP_C, lw=1.4)
        for c in b[1:-1]:
            ax.axvline(c, color=DP_C, alpha=0.35, lw=1.0)
        ax.set_title("$\\beta_{rel}$ = %.3g   (K = %d)" % (br, len(b) - 2), fontsize=12)
    fig.suptitle("%s: one interpretable knob replaces the whole heuristic cascade"
                 % meta[idx]["ticker"], fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(join(FIGS, "cpd_opt_beta_sweep.png"), dpi=160)
    plt.close(fig)


def main():
    d, meta = load()
    rows = fig_examples(d, meta)
    fig_cost_vs_k(d, meta)
    fig_beta_sweep(d, meta)

    print("%-8s %4s %12s %12s %8s | %4s %12s" %
          ("ticker", "K", "SSE heur", "SSE DP@K", "gain", "K_dp", "SSE DP"))
    for tk, k, s_old, s_new, k_dp, s_dp in rows:
        gain = 100 * (1 - s_new / s_old) if s_old > 0 else 0.0
        print("%-8s %4d %12.5f %12.5f %7.1f%% | %4d %12.5f" %
              (tk, k, s_old, s_new, gain, k_dp, s_dp))


if __name__ == "__main__":
    main()
