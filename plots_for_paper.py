import numpy as np
import matplotlib.pyplot as plt
from compare_baselines import dtw_distance, lcss_distance


# ── helpers ────────────────────────────────────────────────────────────────────

def normalize(s):
    return (s - s.min()) / (s.max() - s.min())


# ── sequences ──────────────────────────────────────────────────────────────────

N = 120
t = np.linspace(0, 1, N)

# A = np.sin(4 * np.pi * t)  # 1 cycle, amplitude 1
# B = 2 * np.sin(4 * np.pi * t)  # 1 cycle, amplitude 2
# C = np.sin(8 * np.pi * t)  # 2 cycles, amplitude 1
#
# # Make the first 10 samples of A and C a linear line from -2 to 0
# A[:10] = np.linspace(-2, 0, 10)
# C[:10] = np.linspace(-2, 0, 10)
# # Normalize to [0, 1]
# A, B, C = normalize(A), normalize(B), normalize(C)

# Normalize to [0, 1]
A, B, C = normalize(A), normalize(B), normalize(C)

# ── verify metrics ─────────────────────────────────────────────────────────────

dtw_AB, dtw_AC = dtw_distance(A, B), dtw_distance(A, C)
lcss_AB, lcss_AC = lcss_distance(A, B), lcss_distance(A, C)

print(f"DTW   A-B={dtw_AB:.2f}  A-C={dtw_AC:.2f}  -> metric prefers A-{'C' if dtw_AC < dtw_AB else 'B'}")
print(f"LCSS  A-B={lcss_AB:.2f}  A-C={lcss_AC:.2f}  -> metric prefers A-{'C' if lcss_AC < lcss_AB else 'B'}")

# ── plot ───────────────────────────────────────────────────────────────────────
plt.subplots(1, 3, figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(t, A, color='royalblue', label='Sequence A')
plt.title("Sequences A", fontsize=20)
plt.xlabel("Timestep", fontsize=15)
plt.ylabel("Value", fontsize=15)
plt.subplot(1, 3, 2)
plt.plot(t, B, color='hotpink', label='Sequence B')
plt.title("Sequence B", fontsize=20)
plt.xlabel("Timestep", fontsize=15)
plt.subplot(1, 3, 3)
plt.plot(t, C, color='turquoise', label='Sequence C')
plt.title("Sequence C", fontsize=20)
plt.xlabel("Timestep", fontsize=15)

plt.tight_layout()
plt.show()
