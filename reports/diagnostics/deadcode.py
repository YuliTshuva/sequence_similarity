import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # project root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")
import numpy as np, utils
from utils import change_points_detection, normalize_sequence

d = np.load("data/six_cps_sequences.npz")
seqs = [normalize_sequence(d[f"seq_{i}"].astype(float)) for i in range(300)]

base = [list(change_points_detection(s)) for s in seqs]
utils.find_local_extrema = lambda der_f, f, mp: []          # neutralise the extrema branch
alt = [list(change_points_detection(s)) for s in seqs]
same = sum(1 for a, b in zip(base, alt) if a == b)
print(f"identical segmentations with find_local_extrema disabled: {same}/{len(seqs)}")

# how much of the segmentation survives merge_nearby_points' height-merge tail?
n_dropped = []
for s in seqs[:100]:
    cps = change_points_detection(s)
    n_dropped.append(len(cps))
print(f"change points per sequence: mean={np.mean(n_dropped):.2f}  "
      f"(dataset was filtered to >= 6 by mine_data)")
