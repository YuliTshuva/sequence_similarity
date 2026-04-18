"""
Yuli Tshuva
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import yfinance as yf
from sequence_similarity.seq_sim_alg import plot_two_sequences
from sequence_similarity.utils import change_points_detection, mark_nodes_limits, load_data

# Constants
R = 1.0
n = 1.0
rcParams["font.family"] = "Times New Roman"


def isothermal(V1, V2, T, n_points):
    V = np.linspace(V1, V2, n_points)
    P = n * R * T / V
    T_arr = np.full_like(V, T)
    S = n * R * np.log(V / V1)
    return V, P, T_arr, S


def adiabatic(V1, V2, T1, gamma, n_points):
    V = np.linspace(V1, V2, n_points)
    T = T1 * (V1 / V) ** (gamma - 1)
    P = n * R * T / V
    S = np.full_like(V, 0.0)
    return V, P, T, S


def isobaric(V1, V2, P_const, n_points):
    V = np.linspace(V1, V2, n_points)
    T = P_const * V / (n * R)
    P = np.full_like(V, P_const)
    S = np.log(V / V1)
    return V, P, T, S


def isochoric(V, T1, T2, n_points):
    T = np.linspace(T1, T2, n_points)
    V_arr = np.full_like(T, V)
    P = n * R * T / V
    S = np.log(T / T1)
    return V_arr, P, T, S


def concat_cycle(parts):
    """Concatenates V, P, T, S arrays from individual cycle segments."""
    V = np.concatenate([p[0] for p in parts])
    P = np.concatenate([p[1] for p in parts])
    T = np.concatenate([p[2] for p in parts])
    S = np.concatenate([p[3] for p in parts])
    return V, P, T, S


def carnot_cycle(n_points=100, Th=400, Tc=300, V_start=1.0, V_ratio=2.0, gamma=1.4):
    VA = V_start
    VB = VA * V_ratio
    # 1-2 Isothermal Expansion
    AB = isothermal(VA, VB, Th, n_points)
    # 2-3 Adiabatic Expansion
    VC = VB * (Th / Tc) ** (1 / (gamma - 1))
    BC = adiabatic(VB, VC, Th, gamma, n_points)
    # 3-4 Isothermal Compression
    VD = VC / V_ratio
    CD = isothermal(VC, VD, Tc, n_points)
    # 4-1 Adiabatic Compression
    DA = adiabatic(VD, VA, Tc, gamma, n_points)
    return concat_cycle([AB, BC, CD, DA])


def otto_cycle(n_points=100, T_min=300, T_max=600, V_max=1.0, r=2.0, gamma=1.4):
    VA = V_max
    VB = VA / r
    AB = adiabatic(VA, VB, T_min, gamma, n_points)
    BC = isochoric(VB, AB[2][-1], T_max, n_points)
    CD = adiabatic(VB, VA, T_max, gamma, n_points)
    DA = isochoric(VA, CD[2][-1], T_min, n_points)
    return concat_cycle([AB, BC, CD, DA])


def diesel_cycle(n_points=100, T_min=300, V_max=1.0, r=2.0, V_cutoff=0.8, gamma=1.4):
    VA = V_max
    VB = VA / r
    AB = adiabatic(VA, VB, T_min, gamma, n_points)
    BC = isobaric(VB, V_cutoff, AB[1][-1], n_points)
    CD = adiabatic(V_cutoff, VA, BC[2][-1], gamma, n_points)
    DA = isochoric(VA, CD[2][-1], T_min, n_points)
    return concat_cycle([AB, BC, CD, DA])


def stirling_cycle(n_points=100, Th=400, Tc=300, V_min=1.0, V_max=2.0):
    AB = isothermal(V_min, V_max, Th, n_points)
    BC = isochoric(V_max, Th, Tc, n_points)
    CD = isothermal(V_max, V_min, Tc, n_points)
    DA = isochoric(V_min, Tc, Th, n_points)
    return concat_cycle([AB, BC, CD, DA])


def atkinson_cycle(n_points=100, T_min=300, T_max=600, V_min=0.5, V_max=1.0, V_expand=1.5, gamma=1.4):
    AB = adiabatic(V_max, V_min, T_min, gamma, n_points)
    BC = isochoric(V_min, AB[2][-1], T_max, n_points)
    CD = adiabatic(V_min, V_expand, T_max, gamma, n_points)
    DA = isobaric(V_expand, V_max, CD[1][-1], n_points)
    return concat_cycle([AB, BC, CD, DA])


def brayton_cycle(n_points=100, T_min=300, V_start=1.0, V_comp=0.6, V_exp=1.2, gamma=1.4):
    AB = adiabatic(V_start, V_comp, T_min, gamma, n_points)
    BC = isobaric(V_comp, V_exp, AB[1][-1], n_points)
    CD = adiabatic(V_exp, V_start, BC[2][-1], gamma, n_points)
    DA = isobaric(V_start, V_start, CD[1][-1], n_points)
    return concat_cycle([AB, BC, CD, DA])


def reversed_brayton_cycle(n_points=100, T_min=300, V_start=1.0, V_comp=0.6, V_exp=1.2, gamma=1.4):
    AB = adiabatic(V_start, V_comp, T_min, gamma, n_points)
    BC = isobaric(V_comp, V_exp, AB[1][-1], n_points)
    CD = adiabatic(V_exp, V_start, BC[2][-1], gamma, n_points)
    DA = isobaric(V_start, V_start, CD[1][-1], n_points)
    return concat_cycle([AB, BC, CD, DA])


def lenoir_cycle(n_points=100, T_min=300, T_max=600, V_min=1.0, V_max=2.0, gamma=1.4):
    AB = isochoric(V_min, T_min, T_max, n_points)
    BC = adiabatic(V_min, V_max, T_max, gamma, n_points)
    CD = isobaric(V_max, V_min, BC[1][-1], n_points)
    return concat_cycle([AB, BC, CD])


def second_ericsson_cycle(n_points=100, Th=400, Tc=300, V_min=1.0, V_max=2.0):
    AB = isothermal(V_min, V_max, Th, n_points)
    BC = isobaric(V_max, V_max, AB[1][-1], n_points)
    CD = isothermal(V_max, V_min, Tc, n_points)
    DA = isobaric(V_min, V_min, CD[1][-1], n_points)
    return concat_cycle([AB, BC, CD, DA])


def rankine_cycle(n_points=100, V_min=1.0, V_max=2.0, T_low=300, T_high=500):
    AB = isochoric(V_min, T_low, T_high, n_points)
    BC = isobaric(V_min, V_max, AB[1][-1], n_points)
    CD = isochoric(V_max, T_high, T_low, n_points)
    DA = isobaric(V_max, V_min, CD[1][-1], n_points)
    return concat_cycle([AB, BC, CD, DA])


def plot_cycle(V, P, T, S):
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))

    axs[0].plot(V, color="turquoise")
    axs[0].set_title("Volume")

    axs[1].plot(P, color="salmon")
    axs[1].set_title("Pressure")

    axs[2].plot(S, color="dodgerblue")
    axs[2].set_title("Entropy")

    axs[3].plot(T, color="hotpink")
    axs[3].set_title("Temperature")

    plt.tight_layout()
    plt.show()


def generate_engines_data():
    # --- Brayton & Reversed Brayton Cycles ---
    for high_pressure in range(10000, 110000, 20000):
        for low_pressure in range(10000, 110000, 20000):
            for heat_ratio in range(1, 11, 2):
                q = brayton_cycle(V_comp=0.6, V_exp=1.2 * heat_ratio)
                plot_cycle(q[0], q[1], q[2], q[3])
                return
                q_rev = reversed_brayton_cycle(V_comp=0.6, V_exp=1.2 * heat_ratio)

    # --- Carnot Engine ---
    for hot in range(100, 1100, 200):
        for cold in range(100, 1100, 200):
            q = carnot_cycle(Th=hot, Tc=cold)

    # --- Diesel Engine Cycle ---
    for hot in range(500, 2000, 300):
        for cold in range(100, 500, 100):
            for compression in range(1, 10, 2):
                q = diesel_cycle(T_min=cold, r=compression)

    # --- Otto Cycle ---
    for heat in range(1, 30, 5):
        for compression in range(1, 20, 4):
            q = otto_cycle(T_max=300 + (heat * 10), r=compression)

    # --- Lenoir Cycle ---
    for hot in range(1000, 10000, 2000):
        for cold in range(100, 500, 100):
            for heat_ratio in range(1, 10, 2):
                q = lenoir_cycle(T_max=hot, T_min=cold, V_max=1.0 + heat_ratio)

    # --- Rankine Cycle ---
    for high_pressure in range(100000, 1100000, 200000):
        for low_pressure in range(100000, 1100000, 200000):
            for heat_ratio in range(1, 11, 2):
                q = rankine_cycle(V_max=1.0 + heat_ratio)

    # --- Second Ericsson Cycle ---
    for hot in range(100, 1100, 200):
        for cold in range(100, 1100, 200):
            q = second_ericsson_cycle(Th=hot, Tc=cold)

    # --- Stirling Cycle ---
    for hot in range(100, 1100, 200):
        for cold in range(100, 1100, 200):
            q = stirling_cycle(Th=hot, Tc=cold)

    # --- Atkinson Cycle ---
    for cold in range(100, 500, 100):
        for hot in range(500, 2000, 300):
            for compression in range(1, 10, 2):
                for ratio in range(1, 10, 2):
                    q = atkinson_cycle(T_min=cold, T_max=hot, V_min=1.0 / compression, V_expand=ratio)


def plot_stock_trend():
    plt.subplots(4, 3, figsize=(25, 20))
    title_size = 26
    label_size = 20

    ticker = "TSLA"
    stock_data = yf.download(ticker, start=f"2024-07-01", end=f"2024-11-01",
                             progress=False, multi_level_index=False)
    a1 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 6
    a_smooth = np.convolve(a1, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 1)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="dodgerblue")
    plt.title(f"TSLA: smooth window {window_size}", fontsize=title_size)
    plt.ylabel("Price ($)", fontsize=label_size)

    ticker = "TSLA"
    stock_data = yf.download(ticker, start=f"2024-09-01", end=f"2025-04-15",
                             progress=False, multi_level_index=False)
    a2 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 7
    a_smooth = np.convolve(a2, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 2)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="hotpink")
    plt.title(f"TSLA: smooth window {window_size}", fontsize=title_size)

    ticker = "AAPL"
    stock_data = yf.download(ticker, start=f"2024-11-01", end=f"2025-03-15",
                             progress=False, multi_level_index=False)
    a3 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 8
    a_smooth = np.convolve(a3, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 3)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="deepskyblue")
    plt.title(f"AAPL: smooth window {window_size}", fontsize=title_size)

    ticker = "AAPL"
    stock_data = yf.download(ticker, start=f"2024-09-01", end=f"2025-05-30",
                             progress=False, multi_level_index=False)
    a4 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 15
    a_smooth = np.convolve(a4, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 4)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="salmon")
    plt.title(f"AAPL: smooth window {window_size}", fontsize=title_size)
    plt.ylabel("Price ($)", fontsize=label_size)

    ticker = "WMT"
    stock_data = yf.download(ticker, start=f"2023-06-01", end=f"2023-12-15",
                             progress=False, multi_level_index=False)
    a5 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 12
    a_smooth = np.convolve(a5, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 5)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="royalblue")
    plt.title(f"WMT: smooth window {window_size}", fontsize=title_size)

    ticker = "MSFT"
    stock_data = yf.download(ticker, start=f"2024-02-22", end=f"2024-09-19",
                             progress=False, multi_level_index=False)
    a6 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 8
    a_smooth = np.convolve(a6, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 6)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="turquoise")
    plt.title(f"MSFT: smooth window {window_size}", fontsize=title_size)

    ticker = "MSFT"
    stock_data = yf.download(ticker, start=f"2024-09-19", end=f"2025-01-19",
                             progress=False, multi_level_index=False)
    a7 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 8
    a_smooth = np.convolve(a7, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 7)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="lightblue")
    plt.title(f"MSFT: smooth window {window_size}", fontsize=title_size)
    plt.ylabel("Price ($)", fontsize=label_size)

    ticker = "MSFT"
    stock_data = yf.download(ticker, start=f"2025-05-27", end=f"2025-12-12",
                             progress=False, multi_level_index=False)
    a8 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 13
    a_smooth = np.convolve(a8, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 8)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="purple")
    plt.title(f"MSFT: smooth window {window_size}", fontsize=title_size)

    ticker = "AMZN"
    stock_data = yf.download(ticker, start=f"2025-05-27", end=f"2026-03-12",
                             progress=False, multi_level_index=False)
    a9 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 25
    a_smooth = np.convolve(a9, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 9)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="cyan")
    plt.title(f"AMZN: smooth window {window_size}", fontsize=title_size)

    ticker = "AMZN"
    stock_data = yf.download(ticker, start=f"2024-02-27", end=f"2024-08-12",
                             progress=False, multi_level_index=False)
    a10 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 12
    a_smooth = np.convolve(a10, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 10)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="violet")
    plt.title(f"AMZN: smooth window {window_size}", fontsize=title_size)
    plt.ylabel("Price ($)", fontsize=label_size)
    plt.xlabel("Date (yyyy-mm-dd)", fontsize=label_size)

    ticker = "AMZN"
    stock_data = yf.download(ticker, start=f"2024-10-27", end=f"2025-03-18",
                             progress=False, multi_level_index=False)
    a11 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 10
    a_smooth = np.convolve(a11, np.ones(window_size) / window_size, mode="valid")
    plt.subplot(4, 3, 11)
    plt.plot(stock_data.index[window_size - 1:], a_smooth, color="magenta")
    plt.title(f"AMZN: smooth window {window_size}", fontsize=title_size)
    plt.xlabel("Date (yyyy-mm-dd)", fontsize=label_size)

    plt.subplot(4, 3, 12)
    plt.xlabel("Date (yyyy-mm-dd)", fontsize=label_size)

    plt.suptitle("Smoothed Stock Price Trends", fontsize=45)

    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    plt.show()


def get_stock_trend():
    ticker = "TSLA"
    stock_data = yf.download(ticker, start=f"2024-07-01", end=f"2024-11-01",
                             progress=False, multi_level_index=False)
    a1 = stock_data["Close"].to_numpy()
    window_size = 6
    a_smooth1 = np.convolve(a1, np.ones(window_size) / window_size, mode="valid")

    ticker = "TSLA"
    stock_data = yf.download(ticker, start=f"2024-09-01", end=f"2025-04-15",
                             progress=False, multi_level_index=False)
    a2 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 7
    a_smooth2 = np.convolve(a2, np.ones(window_size) / window_size, mode="valid")

    ticker = "AAPL"
    stock_data = yf.download(ticker, start=f"2024-11-01", end=f"2025-03-15",
                             progress=False, multi_level_index=False)
    a3 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 8
    a_smooth3 = np.convolve(a3, np.ones(window_size) / window_size, mode="valid")

    ticker = "AAPL"
    stock_data = yf.download(ticker, start=f"2024-09-01", end=f"2025-05-30",
                             progress=False, multi_level_index=False)
    a4 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 15
    a_smooth4 = np.convolve(a4, np.ones(window_size) / window_size, mode="valid")

    ticker = "WMT"
    stock_data = yf.download(ticker, start=f"2023-06-01", end=f"2023-12-15",
                             progress=False, multi_level_index=False)
    a5 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 12
    a_smooth5 = np.convolve(a5, np.ones(window_size) / window_size, mode="valid")

    ticker = "MSFT"
    stock_data = yf.download(ticker, start=f"2024-02-22", end=f"2024-09-19",
                             progress=False, multi_level_index=False)
    a6 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 8
    a_smooth6 = np.convolve(a6, np.ones(window_size) / window_size, mode="valid")

    ticker = "MSFT"
    stock_data = yf.download(ticker, start=f"2024-09-19", end=f"2025-01-19",
                             progress=False, multi_level_index=False)
    a7 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 8
    a_smooth7 = np.convolve(a7, np.ones(window_size) / window_size, mode="valid")

    ticker = "MSFT"
    stock_data = yf.download(ticker, start=f"2025-05-27", end=f"2025-12-12",
                             progress=False, multi_level_index=False)
    a8 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 13
    a_smooth8 = np.convolve(a8, np.ones(window_size) / window_size, mode="valid")

    ticker = "AMZN"
    stock_data = yf.download(ticker, start=f"2025-05-27", end=f"2026-03-12",
                             progress=False, multi_level_index=False)
    a9 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 25
    a_smooth9 = np.convolve(a9, np.ones(window_size) / window_size, mode="valid")

    ticker = "AMZN"
    stock_data = yf.download(ticker, start=f"2024-02-27", end=f"2024-08-12",
                             progress=False, multi_level_index=False)
    a10 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 12
    a_smooth10 = np.convolve(a10, np.ones(window_size) / window_size, mode="valid")

    ticker = "AMZN"
    stock_data = yf.download(ticker, start=f"2024-10-27", end=f"2025-03-18",
                             progress=False, multi_level_index=False)
    a11 = stock_data["Close"].to_numpy()
    # Smooth a aggressively to get a clear trend
    window_size = 10
    a_smooth11 = np.convolve(a11, np.ones(window_size) / window_size, mode="valid")

    return a_smooth1, a_smooth2, a_smooth3, a_smooth4, a_smooth5, a_smooth6, a_smooth7, a_smooth8, a_smooth9, a_smooth10, a_smooth11


def add_noise_to_sequence(seq, noise_level=0.05):
    """Adds random noise to a sequence."""
    # Normalize sequences to be in [0, 1]
    if np.max(seq) - np.min(seq) > 0:
        seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
    else:
        seq = np.zeros_like(seq) + 0.5

    seq_1_change_points = change_points_detection(seq)

    # Create nodes based on change points
    nodes_1 = mark_nodes_limits(len(seq), seq_1_change_points)

    # Randomly pick a node
    random_node_index = np.random.randint(0, len(nodes_1) - 1)

    # Get the end of the node chosen and start of the following node
    node_end = nodes_1[random_node_index][1]
    next_node_start = nodes_1[random_node_index + 1][0]

    # Create a linear interpolation of size len(seq)*noise_level between the value of the end of the current node and the start of the next node
    linear_interp = np.linspace(seq[node_end], seq[next_node_start], int(len(seq) * noise_level))
    # Add some random noise to the linear interpolation
    linear_interp += np.random.normal(0, 0.03, size=linear_interp.shape)

    # Insert the linear interpolation into the sequence at the end of the current node
    noisy_seq = np.insert(seq, next_node_start, linear_interp)

    return noisy_seq


def permute_sequence(seq, permutation_level=0.05):
    """Adds random noise to a sequence."""
    # Normalize sequences to be in [0, 1]
    if np.max(seq) - np.min(seq) > 0:
        seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
    else:
        seq = np.zeros_like(seq) + 0.5

    seq_1_change_points = change_points_detection(seq)

    # Create nodes based on change points
    nodes_1 = mark_nodes_limits(len(seq), seq_1_change_points)

    # Get a permutation of a sample of the nodes (the number of nodes to permute is determined by the permutation level)
    nodes_to_permute = np.random.choice(len(nodes_1), size=int(len(nodes_1) * permutation_level), replace=False)
    permutation = np.random.permutation(nodes_to_permute)
    # Permute the nodes in the sequence according to the permutation
    permuted_nodes = []
    for i in range(len(nodes_1)):
        if i in nodes_to_permute:
            permuted_nodes.append(nodes_1[permutation[np.where(nodes_to_permute == i)[0][0]]])
        else:
            permuted_nodes.append(nodes_1[i])
    # Create the permuted sequence by concatenating the nodes in the permuted order
    permuted_seq = np.concatenate([seq[node[0]:node[1] + 1] for node in permuted_nodes])

    return permuted_seq


def shrink_and_stretch_sequence(seq, change_level_level=0.05, factor=2.0):
    """Adds random noise to a sequence."""
    # Normalize sequences to be in [0, 1]
    if np.max(seq) - np.min(seq) > 0:
        seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
    else:
        seq = np.zeros_like(seq) + 0.5

    seq_1_change_points = change_points_detection(seq)

    # Create nodes based on change points
    nodes_1 = mark_nodes_limits(len(seq), seq_1_change_points)

    # Get a permutation of a sample of the nodes (the number of nodes to permute is determined by the permutation level)
    nodes_to_change = np.random.choice(len(nodes_1), size=int(len(nodes_1) * change_level_level), replace=False)

    # Stretch/shrink the segments in the sequence
    changed_sequence = []
    for i in range(len(nodes_1)):
        if i in nodes_to_change:
            # Stretch/shrink the segment by the factor
            segment = seq[nodes_1[i][0]:nodes_1[i][1] + 1]
            new_length = int(len(segment) * factor ** np.random.uniform(-1, 1))
            changed_segment = np.interp(np.linspace(0, len(segment) - 1, new_length), np.arange(len(segment)),
                                        segment)
            changed_sequence.append(changed_segment)
        else:
            changed_sequence.append(seq[nodes_1[i][0]:nodes_1[i][1] + 1])

    # Create the change sequence by concatenating the segments in the changed order
    changed_seq = np.concatenate(changed_sequence)

    return changed_seq


def main():
    seq1 = load_data("data/Atkinson_cycle_44.csv")
    noisy_seq = add_noise_to_sequence(seq1, noise_level=0.1)
    permuted_sequence = permute_sequence(seq1, permutation_level=0.5)
    stretched_sequence = shrink_and_stretch_sequence(seq1, change_level_level=0.4, factor=4.5)
    # plot_two_sequences(seq1, noisy_seq, suptitle="Adding Noise to a sequence")
    plot_two_sequences(seq1, stretched_sequence, suptitle="Stretching/Shrinking a sequence")


if __name__ == "__main__":
    main()
