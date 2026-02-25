"""
Yuli Tshuva
Analyze wolfram data.
"""

# Imports
from utils import *
import matplotlib.pyplot as plt
from os.path import join
import time
from matplotlib import rcParams

# Constants
rcParams['font.family'] = 'Times New Roman'
DATA_DIR = "data"
PLOTS_DIR = join("plots", "change_points")
CONVOLVE_KERNEL_SIZE = 10
CHANGE_THRESHOLD = 3


def annotate_change_points_example():
    # Load a sample data
    file_name = "Atkinson_cycle_12.csv"
    file_path = join(DATA_DIR, file_name)

    # Read data
    f = load_data(file_path)

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Set penalties to examine
    pens = [5, 15, 50, 100, 300]
    models = ["l1", "l2", "rbf"]

    # Loop over models and penalties
    for i in range(3):
        for j in range(5):
            # Find change points
            f_change_points = change_points(f, pen=pens[j], model=models[i])

            # Plot results
            axes[i, j].plot(f, label='Signal', color='royalblue')
            axes[i, j].vlines(f_change_points, ymin=min(f), ymax=max(f),
                              colors='hotpink', linestyles='dashed', label='Change Points')

    # Set labels
    for j in range(5):
        axes[0, j].set_title(f"Penalty: {pens[j]}", fontsize=15)
    for i in range(3):
        axes[i, 0].set_ylabel(f"Model: {models[i]}", fontsize=15)

    # Set suptitle
    plt.suptitle(f"Change Point Detection on {file_name}", fontsize=36)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(join(PLOTS_DIR, "change_point_detection_grid.png"))
    plt.show()


def annotate_sax_example():
    # Load a sample data
    file_name = "Atkinson_cycle_12.csv"
    file_path = join(DATA_DIR, file_name)

    # Read data
    f = load_data(file_path)

    # SAX transform
    n_bins = 7

    start = time.time()
    f_sax, bins, sax = sax_transform(f, n_bins=n_bins)
    print("SAX transform time:", time.time() - start)

    # Plot the data with SAX annotations
    plt.figure(figsize=(15, 6))
    plt.plot(f, color='turquoise', label='Original Signal')
    for i in range(1, len(f_sax), 3):
        plt.axvline(x=i, color='lightgray', linestyle='--', linewidth=0.5)
        plt.text(i - 0.5, f[i - 1], f_sax[i - 1], fontsize=12, color='darkblue', ha='center', va='bottom')
    plt.title(f"SAX Annotation on {file_name}", fontsize=20)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Value", fontsize=15)
    plt.legend()
    plt.savefig(join(PLOTS_DIR, "sax_annotation.png"))
    plt.show()


def plato_research():
    # Load a sample data
    file_name = "Atkinson_cycle_12.csv"
    file_path = join(DATA_DIR, file_name)

    # Read data
    f = load_data(file_path)

    # Get plateau indices
    plateau_indices = feature_points(f)

    # Plot the data
    plt.figure(figsize=(15, 6))
    plt.plot(f, color='royalblue', label='Original Signal')
    plt.scatter(plateau_indices, f[plateau_indices], color='red', label='Plateaus')

    plt.title(f"Plateaus in {file_name}", fontsize=20)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Value", fontsize=15)
    plt.legend()
    plt.savefig(join(PLOTS_DIR, "plateaus_exploration.png"))
    plt.show()


def wide_search(search="plateaus"):
    # Load a sample data
    file_names = [f"Atkinson_cycle_{i + 12}.csv" for i in range(9)]

    # Set a 3x3 grid for plots
    fig, ax = plt.subplots(3, 3, figsize=(20, 15))

    for i, file_name in enumerate(file_names):
        # Construct file path
        file_path = join(DATA_DIR, file_name)

        # Read data
        f = load_data(file_path)

        # Get plateau indices
        if search == "plateaus":
            plateau_indices = feature_points(f)
        else:
            raise ValueError("Invalid search type")

        # Plot the data
        ax[i // 3, i % 3].plot(f, color='royalblue')
        ax[i // 3, i % 3].scatter(plateau_indices, f[plateau_indices], color='red')

        # Set title
        ax[i // 3, i % 3].set_title(file_name, fontsize=20)

        if i // 3 == 2:
            ax[i // 3, i % 3].set_xlabel("Time", fontsize=15)
        if i % 3 == 0:
            ax[i // 3, i % 3].set_ylabel("Value", fontsize=15)

    # Set suptitle
    plt.suptitle(f"Plateaus feature points in Atkinson Cycles", fontsize=36)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    plt.savefig(join(PLOTS_DIR, "plateaus_wide_search.png"))
    plt.show()


def plot_derivatives():
    # Load a sample data
    file_names = [f"Atkinson_cycle_{i + 12}.csv" for i in range(9)]

    # Set a 3x3 grid for plots
    fig, ax = plt.subplots(3, 3, figsize=(20, 15))

    for i, file_name in enumerate(file_names):
        # Construct file path
        file_path = join(DATA_DIR, file_name)

        # Read data
        f = load_data(file_path).copy()

        # Smooth the data
        kernel = np.array(list(range(1, CONVOLVE_KERNEL_SIZE//2 + 1)) + list(range(CONVOLVE_KERNEL_SIZE//2 - 1, 0, -1)))
        kernel *= kernel
        # Normalize the kernel
        kernel = kernel / kernel.sum()
        convolved_f = np.convolve(f, kernel, mode='same')
        f[CONVOLVE_KERNEL_SIZE:-CONVOLVE_KERNEL_SIZE] = convolved_f[CONVOLVE_KERNEL_SIZE:-CONVOLVE_KERNEL_SIZE]

        # Calculate first derivative of f
        der_f = np.diff(f, n=1)

        # 1, 2, 3, 4, 5, 4, 3, 2, 1
        # 1, 1, 1, 1, -1, -1, -1, -1

        # Calculate the derivative amplitude
        der_amp = np.max(der_f) - np.min(der_f)

        # Track change points in the derivative
        threshold = CHANGE_THRESHOLD * der_amp / 100

        # Apply sign_func over der_f
        signs = [sign_func(x, threshold) for x in der_f]

        # Filter the edges
        signs = signs[CONVOLVE_KERNEL_SIZE:-CONVOLVE_KERNEL_SIZE]
        signs = np.array(signs)

        # Extract feature points out of signs
        signs_fps = feature_points(signs)

        # Plot the data
        ax[i // 3, i % 3].scatter(range(len(signs)), signs, color='royalblue')
        ax[i // 3, i % 3].scatter(signs_fps, signs[signs_fps], color='red')

        # Set title
        ax[i // 3, i % 3].set_title(file_name, fontsize=20)

        if i // 3 == 2:
            ax[i // 3, i % 3].set_xlabel("Time", fontsize=15)
        if i % 3 == 0:
            ax[i // 3, i % 3].set_ylabel("Value", fontsize=15)

    # Set suptitle
    plt.suptitle(f"Categorized points by derivatives values", fontsize=36)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Save the figure
    plt.savefig(join(PLOTS_DIR, "categorized_derivatives.png"))
    plt.show()

    # Set a 3x3 grid for plots
    fig, ax = plt.subplots(3, 3, figsize=(20, 15))

    for i, file_name in enumerate(file_names):
        # Construct file path
        file_path = join(DATA_DIR, file_name)

        # Read data
        f = load_data(file_path).copy()

        # Smooth the data
        kernel = np.array(
            list(range(1, CONVOLVE_KERNEL_SIZE // 2 + 1)) + list(range(CONVOLVE_KERNEL_SIZE // 2 - 1, 0, -1)))
        kernel *= kernel
        # Normalize the kernel
        kernel = kernel / kernel.sum()
        convolved_f = np.convolve(f, kernel, mode='same')
        f[CONVOLVE_KERNEL_SIZE:-CONVOLVE_KERNEL_SIZE] = convolved_f[CONVOLVE_KERNEL_SIZE:-CONVOLVE_KERNEL_SIZE]

        # Calculate first derivative of f
        der_f = np.diff(f, n=1)

        # Calculate the derivative amplitude
        der_amp = np.max(der_f) - np.min(der_f)

        # Track change points in the derivative
        threshold = CHANGE_THRESHOLD * der_amp / 100

        # Apply sign_func over der_f
        signs = [sign_func(x, threshold) for x in der_f]

        # Filter the edges
        signs = signs[CONVOLVE_KERNEL_SIZE:-CONVOLVE_KERNEL_SIZE]
        signs = np.array(signs)

        # Extract feature points out of signs
        signs_fps = feature_points(signs)
        # Convert signs_fps to numpy array for indexing
        signs_fps = np.array(signs_fps) + CONVOLVE_KERNEL_SIZE

        # Reload data
        f = load_data(file_path).copy()

        # Plot the data
        ax[i // 3, i % 3].plot(range(len(f)), f, color='royalblue')
        ax[i // 3, i % 3].scatter(signs_fps, f[signs_fps], color='hotpink', s=50)

        # Set title
        ax[i // 3, i % 3].set_title(file_name, fontsize=20)

        if i // 3 == 2:
            ax[i // 3, i % 3].set_xlabel("Time", fontsize=15)
        if i % 3 == 0:
            ax[i // 3, i % 3].set_ylabel("Value", fontsize=15)

    # Set suptitle
    plt.suptitle(f"Categorized points by derivatives values", fontsize=36)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Save the figure
    plt.savefig(join(PLOTS_DIR, "categorized_derivatives.png"))
    plt.show()


def generate_a_saw_and_plot_it():
    # Set a 1x2 grid for plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))

    # Generate a sawtooth wave
    n_points = 1000
    f = sawtooth_k_cycles(n_points, k=5)

    # Calculate first derivative of f
    der_f = np.diff(f, n=1)

    # Calculate the derivative amplitude
    der_amp = np.max(der_f) - np.min(der_f)

    # Track change points in the derivative
    threshold = CHANGE_THRESHOLD * der_amp / 100

    # Apply sign_func over der_f
    signs = [sign_func(x, threshold) for x in der_f]

    # Filter the edges
    signs = signs[CONVOLVE_KERNEL_SIZE:-CONVOLVE_KERNEL_SIZE]

    # Plot the data
    ax[0].plot(f, color='royalblue')
    ax[1].scatter(range(len(signs)), signs, color='royalblue')

    # Set title
    ax[0].set_title("Sawtooth Function", fontsize=20)
    ax[1].set_title("Categorized Derivatives", fontsize=20)

    ax[0].set_xlabel("Time", fontsize=15)
    ax[1].set_xlabel("Time", fontsize=15)
    ax[0].set_ylabel("Value", fontsize=15)

    # Set suptitle
    plt.suptitle(f"Categorized points by derivatives values", fontsize=36)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Save the figure
    plt.savefig(join(PLOTS_DIR, "Sawtooth.png"))
    plt.show()


if __name__ == "__main__":
    plot_derivatives()
