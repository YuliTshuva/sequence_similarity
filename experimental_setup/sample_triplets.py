"""
Yuli Tshuva
Sample triplets (query, candidate_a, candidate_b) for the human-alignment study.

A trial teaches us nothing when every baseline agrees which candidate is closer to
the query -- the rater agrees too, and no measure is distinguished. The informative
trials are the ones where the baselines split.

With three baselines each casting a binary vote there are eight vote patterns, but
the a/b labelling is arbitrary, so folding that symmetry away leaves four classes:
one unanimous, and three in which a single baseline dissents against the other two.
This module samples triplets so the three disagreement classes are equally
represented, and so that within a class the triplets differ both in how decisively
the baselines vote and in which sequences they use.

The proposed method takes no part in the selection.
"""

# Imports
import argparse
import json
import numpy as np

# Constants
DEFAULT_N_TRIPLETS = 100
DEFAULT_POOL_SIZE = 150
DEFAULT_RESAMPLE_LEN = 128
DEFAULT_SHAPE_DESC_LEN = 16
DEFAULT_N_CANDIDATES = 300_000
MIN_RELATIVE_MARGIN = 0.01  # a vote decided by less than this is a tie, not an opinion
PLAUSIBLE_RANK_FRAC = 0.5   # both candidates must sit in the query's closer half
REUSE_PENALTY = 0.05        # discourages building every triplet from the same curves
EPS = 1e-12


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def resample(seq, length):
    """Linearly resample a 1-D sequence to a fixed length."""
    seq = np.asarray(seq, dtype=float)
    x_old = np.linspace(0.0, 1.0, len(seq))
    x_new = np.linspace(0.0, 1.0, length)
    return np.interp(x_new, x_old, seq)


def znorm(seq):
    """Z-normalise, leaving a flat sequence at zero."""
    sd = seq.std()
    return (seq - seq.mean()) / sd if sd > EPS else seq - seq.mean()


def prepare_pool(sequences, indices, length, normalise=True):
    """Resample (and optionally z-normalise) the sequences at `indices`."""
    pool = np.stack([resample(sequences[i], length) for i in indices])
    if normalise:
        pool = np.stack([znorm(s) for s in pool])
    return pool


# ----------------------------------------------------------------------------
# Point descriptors -- one per baseline
#
# Each baseline is DTW over a different description of a point, which is what
# makes it stand for a different perceptual hypothesis:
#   DTW      -- the value itself       ("humans match levels")
#   DDTW     -- the local derivative   ("humans match trend")
#   ShapeDTW -- the local subsequence  ("humans match local morphology")
# ----------------------------------------------------------------------------
def desc_value(pool):
    """Raw value. Shape (n, length, 1)."""
    return pool[:, :, None]


def desc_derivative(pool):
    """Keogh & Pazzani (2001) smoothed derivative estimate. Shape (n, length, 1)."""
    padded = np.pad(pool, ((0, 0), (1, 1)), mode="edge")
    left = pool - padded[:, :-2]
    centred = (padded[:, 2:] - padded[:, :-2]) / 2.0
    return ((left + centred) / 2.0)[:, :, None]


def desc_shape(pool, desc_len=DEFAULT_SHAPE_DESC_LEN):
    """
    Zhao & Itti (2018) raw-subsequence descriptor: the z-normalised window centred
    on each point. Shape (n, length, desc_len).
    """
    half = desc_len // 2
    padded = np.pad(pool, ((0, 0), (half, desc_len - half)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, desc_len, axis=1)
    windows = windows[:, : pool.shape[1], :]
    mean = windows.mean(axis=2, keepdims=True)
    sd = windows.std(axis=2, keepdims=True)
    return (windows - mean) / np.maximum(sd, EPS)


# ----------------------------------------------------------------------------
# Batched DTW
# ----------------------------------------------------------------------------
def _dtw_batch(a, b, band=None):
    """
    DTW distance for a batch of aligned pairs, computed one anti-diagonal at a
    time so the whole batch advances together under numpy.

    Cells on an anti-diagonal d = i + j depend only on diagonals d-1 and d-2, so
    they can all be updated at once -- unlike a row-wise pass, where cell (i, j)
    waits on (i, j-1).

    a, b : (n_pairs, length, n_features)
    band : optional Sakoe-Chiba radius
    Returns (n_pairs,) of accumulated squared-euclidean cost.
    """
    n_pairs, length, _ = a.shape
    idx = np.arange(length)
    prev2 = np.full((n_pairs, length), np.inf)  # diagonal d-2
    prev1 = np.full((n_pairs, length), np.inf)  # diagonal d-1

    for d in range(2 * length - 1):
        col = d - idx
        valid = (col >= 0) & (col < length)
        if band is not None:
            valid &= np.abs(idx - col) <= band
        cur = np.full((n_pairs, length), np.inf)

        if valid.any():
            rows, cols = idx[valid], col[valid]
            diff = a[:, rows, :] - b[:, cols, :]
            cost = np.einsum("nkf,nkf->nk", diff, diff)
            if d == 0:
                cur[:, rows] = cost
            else:
                back = np.maximum(rows - 1, 0)
                has_prev_row = rows > 0
                insertion = prev1[:, rows]                                 # (i, j-1)
                deletion = np.where(has_prev_row, prev1[:, back], np.inf)  # (i-1, j)
                match = np.where(has_prev_row, prev2[:, back], np.inf)     # (i-1, j-1)
                cur[:, rows] = cost + np.minimum(np.minimum(insertion, deletion), match)

        prev2, prev1 = prev1, cur

    return prev1[:, length - 1]


def pairwise_dtw(desc, band=None, chunk=2000, verbose=False):
    """Full symmetric DTW distance matrix over a pool of descriptor arrays."""
    n = len(desc)
    rows, cols = np.triu_indices(n, k=1)
    out = np.zeros((n, n))
    for start in range(0, len(rows), chunk):
        sl = slice(start, start + chunk)
        out[rows[sl], cols[sl]] = _dtw_batch(desc[rows[sl]], desc[cols[sl]], band)
        if verbose:
            print(f"    {min(start + chunk, len(rows))}/{len(rows)} pairs", flush=True)
    return out + out.T


# ----------------------------------------------------------------------------
# Triplet sampling
# ----------------------------------------------------------------------------
def _relative_margins(dist_mats, query, cand_a, cand_b):
    """
    Signed margin per measure, in [-1, 1]: positive means candidate a is judged
    closer. Normalising by the sum makes margins comparable across measures whose
    raw distances live on different scales.
    """
    margins = np.empty((len(dist_mats), len(query)))
    for m, dist in enumerate(dist_mats):
        d_a, d_b = dist[query, cand_a], dist[query, cand_b]
        margins[m] = (d_b - d_a) / (d_a + d_b + EPS)
    return margins.T


def _plausible_mask(dist_mats, query, cand_a, cand_b, rank_frac):
    """
    Keep only candidates among the query's closer neighbours under the average of
    the baselines. A candidate that no measure considers remotely similar makes
    for an easy trial, whatever its vote pattern says.
    """
    n = dist_mats[0].shape[0]
    ranks = np.mean([d.argsort(axis=1).argsort(axis=1) for d in dist_mats], axis=0)
    cutoff = rank_frac * n
    return (ranks[query, cand_a] <= cutoff) & (ranks[query, cand_b] <= cutoff)


def _greedy_diverse(margins, triplets, quota, rng):
    """
    Pick `quota` triplets from one disagreement class, greedily taking the one
    furthest in margin space from everything already picked, less a penalty for
    reusing sequences that already appear in the selection.

    This spreads the class over decisive and marginal disagreements alike, and
    keeps the selection from collapsing onto a handful of curves.
    """
    if len(triplets) <= quota:
        return list(range(len(triplets)))

    chosen = [int(rng.integers(len(triplets)))]
    usage = np.zeros(int(triplets.max()) + 1)
    for seq_idx in triplets[chosen[0]]:
        usage[seq_idx] += 1
    nearest = np.linalg.norm(margins - margins[chosen[0]], axis=1)

    while len(chosen) < quota:
        score = nearest - REUSE_PENALTY * usage[triplets].sum(axis=1)
        score[chosen] = -np.inf
        pick = int(score.argmax())
        chosen.append(pick)
        for seq_idx in triplets[pick]:
            usage[seq_idx] += 1
        nearest = np.minimum(nearest, np.linalg.norm(margins - margins[pick], axis=1))

    return chosen


def sample_triplets(sequences,
                    n_triplets=DEFAULT_N_TRIPLETS,
                    pool_size=DEFAULT_POOL_SIZE,
                    n_candidates=DEFAULT_N_CANDIDATES,
                    resample_len=DEFAULT_RESAMPLE_LEN,
                    shape_desc_len=DEFAULT_SHAPE_DESC_LEN,
                    band=None,
                    min_margin=MIN_RELATIVE_MARGIN,
                    rank_frac=PLAUSIBLE_RANK_FRAC,
                    unanimous_quota=0,
                    seed=0,
                    verbose=True):
    """
    Sample triplets spread evenly over the baselines' disagreement patterns.

    Returns a list of dicts, each carrying the indices (into `sequences`) of the
    query and the two candidates, every baseline's vote and margin, and the
    disagreement class the triplet was drawn from.
    """
    rng = np.random.default_rng(seed)

    pool_size = min(pool_size, len(sequences))
    pool_idx = rng.choice(len(sequences), size=pool_size, replace=False)
    pool = prepare_pool(sequences, pool_idx, resample_len)

    measures = {
        "dtw": desc_value(pool),
        "shapedtw": desc_shape(pool, shape_desc_len),
        "ddtw": desc_derivative(pool),
    }
    names = list(measures)

    dist_mats = []
    for name in names:
        if verbose:
            print(f"  {name}: {pool_size * (pool_size - 1) // 2} pairs", flush=True)
        dist_mats.append(pairwise_dtw(measures[name], band, verbose=verbose))

    # Draw candidate triplets of three distinct pool members.
    draws = rng.integers(0, pool_size, size=(n_candidates, 3))
    distinct = ((draws[:, 0] != draws[:, 1]) & (draws[:, 0] != draws[:, 2])
                & (draws[:, 1] != draws[:, 2]))
    draws = draws[distinct]
    query, cand_a, cand_b = draws[:, 0], draws[:, 1], draws[:, 2]

    margins = _relative_margins(dist_mats, query, cand_a, cand_b)

    # Every baseline must hold a real opinion, and both candidates must be
    # plausible answers to the query.
    keep = (np.abs(margins) >= min_margin).all(axis=1)
    keep &= _plausible_mask(dist_mats, query, cand_a, cand_b, rank_frac)
    margins, query, cand_a, cand_b = margins[keep], query[keep], cand_a[keep], cand_b[keep]
    if verbose:
        print(f"  {len(margins)} candidate triplets survived filtering", flush=True)

    # Fold away the arbitrary a/b labelling by making the first baseline always
    # vote for a. What remains is the pattern of the other two.
    flip = margins[:, 0] < 0
    margins[flip] *= -1
    cand_a, cand_b = np.where(flip, cand_b, cand_a), np.where(flip, cand_a, cand_b)

    votes = margins > 0
    triplets = np.stack([query, cand_a, cand_b], axis=1)

    # Four classes: all agree, or exactly one baseline dissents.
    classes = {
        "unanimous": np.flatnonzero(votes[:, 1] & votes[:, 2]),
        f"{names[2]}_dissents": np.flatnonzero(votes[:, 1] & ~votes[:, 2]),
        f"{names[1]}_dissents": np.flatnonzero(~votes[:, 1] & votes[:, 2]),
        f"{names[0]}_dissents": np.flatnonzero(~votes[:, 1] & ~votes[:, 2]),
    }

    disagreement = [k for k in classes if k != "unanimous"]
    n_disagree = n_triplets - unanimous_quota
    quotas = {k: n_disagree // len(disagreement) for k in disagreement}
    for k in disagreement[: n_disagree % len(disagreement)]:
        quotas[k] += 1
    if unanimous_quota:
        quotas["unanimous"] = unanimous_quota

    selected = []
    for key, quota in quotas.items():
        members = classes[key]
        if verbose:
            print(f"  {key}: {len(members)} available, taking {quota}", flush=True)
        if len(members) < quota:
            print(f"  WARNING: only {len(members)} triplets in class '{key}' "
                  f"(wanted {quota}); raise n_candidates or pool_size.", flush=True)
        for i in _greedy_diverse(margins[members], triplets[members], quota, rng):
            row = members[i]
            selected.append({
                "class": key,
                "query": int(pool_idx[query[row]]),
                "candidate_a": int(pool_idx[cand_a[row]]),
                "candidate_b": int(pool_idx[cand_b[row]]),
                "votes": {n: ("a" if margins[row, m] > 0 else "b")
                          for m, n in enumerate(names)},
                "margins": {n: float(margins[row, m]) for m, n in enumerate(names)},
            })

    rng.shuffle(selected)
    for order, triplet in enumerate(selected):
        triplet["trial"] = order
    return selected


def load_sequences(path):
    """
    Load the smoothed sequences from an archive.

    Handles both layouts: the older one where every array is a `seq_i`, and the
    study archive, which also carries a `raw_i` per sequence and an embedded
    metadata blob.
    """
    data = np.load(path, allow_pickle=False)
    if "metadata_json" in data.files:
        import json as _json
        n = len(_json.loads(str(data["metadata_json"])))
    else:
        n = len([k for k in data.files if k.startswith("seq_")])
    return [data[f"seq_{i}"] for i in range(n)]


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, help=".npz written by mine_data")
    parser.add_argument("--out", default="experimental_setup/triplets.json")
    parser.add_argument("--n-triplets", type=int, default=DEFAULT_N_TRIPLETS)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    parser.add_argument("--resample-len", type=int, default=DEFAULT_RESAMPLE_LEN)
    parser.add_argument("--band", type=int, default=None, help="Sakoe-Chiba radius")
    parser.add_argument("--min-margin", type=float, default=MIN_RELATIVE_MARGIN,
                        help="a baseline voting by less than this is a tie, not an opinion")
    parser.add_argument("--unanimous-quota", type=int, default=0,
                        help="easy trials to include as attention checks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sequences = load_sequences(args.sequences)
    print(f"Loaded {len(sequences)} sequences from {args.sequences}")

    triplets = sample_triplets(
        sequences,
        n_triplets=args.n_triplets,
        pool_size=args.pool_size,
        n_candidates=args.n_candidates,
        resample_len=args.resample_len,
        band=args.band,
        min_margin=args.min_margin,
        unanimous_quota=args.unanimous_quota,
        seed=args.seed,
    )

    with open(args.out, "w") as f:
        json.dump(triplets, f, indent=2)
    print(f"Wrote {len(triplets)} triplets to {args.out}")


if __name__ == "__main__":
    main()
