"""
First experimental trail.
The main goal is to generate some random data and visualize it.
I want to test the modularity of mirroring and transitioning of sequences.
"""

# Imports
from utils import *
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib import rcParams
from os.path import join
from numpy.fft import fft as dft

# Set font to Times New Roman
rcParams['font.family'] = 'Times New Roman'

# Setup
SAVE_DIR = join("plots", "beta_example")
colors = ["blue", "orange", "green", "red", "purple", "brown"]
a, b = 10, 50
n = 500

# Plot results
x = np.linspace(0, 1, n)
pdf = beta.pdf(x, a, b)

# Mirrored PDF
mirrored_pdf = pdf[::-1]
# Shitted PDF
shifts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
shifted_pdf = {shift: beta.pdf(x - shift, a, b) for shift in shifts}
# Scales and
scales = [2, 3]

# Calculate DFTs
pdf_dft = dft(pdf)
mirrored_pdf_dft = dft(mirrored_pdf)
shifted_pdf_dft = [dft(spdf) for spdf in shifted_pdf.values()]


def plot_mirrored_pdfs():
    # Mirrored PDF Visualization
    plt.plot(x, pdf, label="pdf", color="turquoise")
    plt.plot(x, mirrored_pdf, label="mirrored pdf", color="coral")
    plt.title("Beta distribution - Mirror Effect", fontsize=19)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.savefig(join(SAVE_DIR, "beta_mirror.png"))
    plt.show()


def plot_shifted_pdfs():
    # Shifted PDFs Visualization
    plt.plot(x, pdf, label="pdf", color="turquoise")
    for i, (shift, spdf) in enumerate(shifted_pdf.items()):
        plt.plot(x, spdf, label=f"shifted pdf (shift={shift})", color=colors[i])
    plt.title("Beta distribution - Shift Effect", fontsize=19)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.savefig(join(SAVE_DIR, "beta_shift.png"))
    plt.show()


def plot_dft_of_pdfs():
    # Setup
    suptitle_fontsize = 30
    title_fontsize = 17
    label_fontsize = 14

    # Create a figure with subplots
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    # DFT of PDFs Visualization
    ax[0, 0].plot(x, np.abs(pdf_dft), label="pdf DFT", color="turquoise")
    ax[0, 0].set_title("DFT of Beta distribution", fontsize=title_fontsize)
    ax[0, 0].set_ylabel("Magnitude", fontsize=label_fontsize)

    # DFT of Mirrored PDF Visualization
    ax[0, 1].plot(x, np.abs(mirrored_pdf_dft), label="mirrored pdf DFT", color="coral")
    ax[0, 1].set_title("DFT of Mirrored Beta distribution", fontsize=title_fontsize)

    # DFT of Shifted PDFs Visualization
    for i, spdf_dft in enumerate(shifted_pdf_dft, start=2):
        ax[i // 3, i % 3].plot(x, np.abs(spdf_dft), label=f"shifted pdf DFT (shift={shifts[i - 2]})",
                               color=colors[i - 2])
        ax[i // 3, i % 3].set_title(f"DFT of Shifted Beta distribution (shift={shifts[i - 2]})",
                                    fontsize=title_fontsize)
        if i % 3 == 0:
            ax[i // 3, i % 3].set_ylabel("Magnitude", fontsize=label_fontsize)

    # Set x-labels for all subplots
    for i in range(3):
        ax[2, i].set_xlabel("Frequency", fontsize=label_fontsize)

    # Plot all in one ax for comparison
    ax[2, 2].plot(x, np.abs(pdf_dft), label="pdf DFT", color="turquoise")
    ax[2, 2].plot(x, np.abs(mirrored_pdf_dft), label="mirrored pdf DFT", color="coral")
    for i, spdf_dft in enumerate(shifted_pdf_dft):
        ax[2, 2].plot(x, np.abs(spdf_dft), label=f"shifted pdf DFT (shift={shifts[i]})",
                      color=colors[i])
    ax[2, 2].set_title("Comparison of DFTs", fontsize=title_fontsize)
    ax[2, 2].set_xlabel("Frequency", fontsize=label_fontsize)
    ax[2, 2].set_ylabel("Magnitude", fontsize=label_fontsize)
    ax[2, 2].legend(fontsize=10)

    plt.suptitle("DFT of Beta distribution - Mirror and Shift Effect", fontsize=suptitle_fontsize)
    plt.savefig(join(SAVE_DIR, "beta_dft.png"))
    plt.show()


def plot_dft_of_scale():
    # Configurations
    title_size = 17
    label_size = 14
    legend_size = 10

    # Define figure
    fig, ax = plt.subplots(2, 2 + 4, figsize=(30, 10))

    # Plot the scaled pdfs
    ax[0, 0].plot(x, pdf, label="pdf", color="turquoise")
    for i, scale in enumerate(scales):
        ax[0, 0].plot(x, pdf * scale, label=f"scale={scale}", color=colors[i])
        ax[0, 0].plot(x, pdf / scale, label=f"scale={1 / scale:.2f}", color=colors[i + 2])
    ax[0, 0].legend(fontsize=legend_size)

    # Set title
    ax[0, 0].set_title("Scaled Beta distribution", fontsize=title_size)

    # Plot the dft of scaled pdfs
    ax[0, 1].plot(x, np.abs(pdf_dft), label="pdf DFT", color="turquoise")
    for i, scale in enumerate(scales):
        scaled_pdf = pdf * scale
        scaled_pdf_dft = dft(scaled_pdf)
        ax[0, 2 + i].plot(x, np.abs(scaled_pdf_dft), label=f"scale={scale}", color=colors[i])
        ax[0, 2 + i].set_title(f"DFT of scaled pdf for (scale={scale})")
        scaled_pdf_inv = pdf / scale
        scaled_pdf_inv_dft = dft(scaled_pdf_inv)
        ax[0, 2 + 2 + i].plot(x, np.abs(scaled_pdf_inv_dft), label=f"scale={1 / scale:.2f}", color=colors[i + 2])
        ax[0, 2 + 2 + i].set_title(f"DFT of scaled pdf for (scale={1 / scale:.2f})")
        ax[1, 2 + 2 + i].legend(fontsize=legend_size)
        ax[1, 2 + i].legend(fontsize=legend_size)
    ax[0, 1].legend(fontsize=legend_size)

    # Set title
    ax[0, 1].set_title("DFT of Beta distribution", fontsize=title_size)

    # Plot the stretched pdfs
    ax[1, 0].plot(x, pdf, label="pdf", color="turquoise")
    for i, scale in enumerate(scales):
        stretched_x = np.clip(x / scale, 0, 1)
        stretched_pdf = beta.pdf(stretched_x, a, b)
        ax[1, 0].plot(x, stretched_pdf, label=f"stretch={scale}", color=colors[i])
        compressed_x = np.clip(x * scale, 0, 1)
        compressed_pdf = beta.pdf(compressed_x, a, b)
        ax[1, 0].plot(x, compressed_pdf, label=f"stretch={1 / scale:.2f}", color=colors[i + 2])
    ax[1, 0].legend(fontsize=legend_size)

    # Set title
    ax[1, 0].set_title("Stretched Beta distribution", fontsize=title_size)

    # Plot the dft of stretched pdfs
    ax[1, 1].plot(x, np.abs(pdf_dft), label="pdf DFT", color="turquoise")
    for i, scale in enumerate(scales):
        stretched_x = np.clip(x / scale, 0, 1)
        stretched_pdf = beta.pdf(stretched_x, a, b)
        stretched_pdf_dft = dft(stretched_pdf)
        ax[1, 2 + i].plot(x, np.abs(stretched_pdf_dft), label=f"stretch={scale}", color=colors[i])
        ax[1, 2 + i].set_title(f"DFT of stretched pdf for (stretch={scale})")
        compressed_x = np.clip(x * scale, 0, 1)
        compressed_pdf = beta.pdf(compressed_x, a, b)
        compressed_pdf_dft = dft(compressed_pdf)
        ax[1, 2 + 2 + i].plot(x, np.abs(compressed_pdf_dft), label=f"stretch={1 / scale:.2f}", color=colors[i + 2])
        ax[1, 2 + 2 + i].set_title(f"DFT of stretched pdf for (stretch={1 / scale:.2f})")
        ax[1, 2 + 2 + i].legend(fontsize=legend_size)
        ax[1, 2 + i].legend(fontsize=legend_size)

    ax[1, 1].legend(fontsize=legend_size)

    # Set title
    ax[1, 1].set_title("DFT of Stretched Beta distribution", fontsize=title_size)

    # Set labels
    ax[1, 0].set_xlabel("x", fontsize=label_size)
    ax[1, 1].set_xlabel("Frequency", fontsize=label_size)
    ax[0, 0].set_ylabel("Density", fontsize=label_size)
    ax[0, 1].set_ylabel("Magnitude", fontsize=label_size)
    ax[1, 0].set_ylabel("Density", fontsize=label_size)
    ax[1, 1].set_ylabel("Magnitude", fontsize=label_size)

    plt.suptitle("Effects of Scaling and Stretching on Beta Distribution and its DFT", fontsize=25)
    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, "beta_scale_stretch_dft.png"))
    plt.show()


def plot_dft_of_noise():
    # Configurations
    title_size = 17
    label_size = 14
    legend_size = 10

    # Define figure
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Generate random noise
    noise = np.random.normal(0, 0.4, n)
    noise_dft = dft(noise)

    # Plot the pdf of the beta distribution
    ax[0, 0].plot(x, pdf, label="pdf", color="turquoise")
    ax[0, 0].set_title("Beta distribution", fontsize=title_size)
    ax[0, 0].set_ylabel("Density", fontsize=label_size)
    ax[0, 0].set_xlabel("x", fontsize=label_size)
    ax[0, 0].legend(fontsize=legend_size)

    # Plot the DFT of the beta distribution
    ax[1, 0].plot(x, np.abs(pdf_dft), label="pdf DFT", color="turquoise")
    ax[1, 0].set_title("DFT of Beta distribution", fontsize=title_size)
    ax[1, 0].set_ylabel("Magnitude", fontsize=label_size)
    ax[1, 0].set_xlabel("Frequency", fontsize=label_size)
    ax[1, 0].legend(fontsize=legend_size)

    # Plot the noise
    ax[0, 1].plot(x, noise, label="noise", color="red")
    ax[0, 1].set_title("Normal distributed Noise", fontsize=title_size)
    ax[0, 1].set_ylabel("Amplitude", fontsize=label_size)
    ax[0, 1].set_xlabel("Frequency", fontsize=label_size)
    ax[0, 1].legend(fontsize=legend_size)

    # Plot the DFT of the noise
    ax[1, 1].plot(x, np.abs(noise_dft), label="noise DFT", color="red")
    ax[1, 1].set_title("DFT of Normal distributed Noise", fontsize=title_size)
    ax[1, 1].set_ylabel("Magnitude", fontsize=label_size)
    ax[1, 1].set_ylim(0, np.max(np.abs(pdf_dft)))
    ax[1, 1].set_xlabel("Frequency", fontsize=label_size)
    ax[1, 1].legend(fontsize=legend_size)

    # Plot combined signal (pdf + noise)
    combined_signal = pdf + noise
    combined_signal_dft = dft(combined_signal)
    ax[0, 2].plot(x, combined_signal, label="combined signal", color="purple")
    ax[0, 2].set_title("Combined Signal (Beta PDF + Noise)", fontsize=title_size)
    ax[0, 2].set_ylabel("Amplitude", fontsize=label_size)
    ax[0, 2].set_xlabel("x", fontsize=label_size)
    ax[0, 2].legend(fontsize=legend_size)

    # Plot the DFT of the combined signal
    ax[1, 2].plot(x, np.abs(combined_signal_dft), label="combined signal DFT", color="purple")
    ax[1, 2].set_title("DFT of Combined Signal", fontsize=title_size)
    ax[1, 2].set_ylabel("Magnitude", fontsize=label_size)
    ax[1, 2].set_xlabel("Frequency", fontsize=label_size)
    ax[1, 2].legend(fontsize=legend_size)

    plt.suptitle("Effects of Noise on Beta Distribution and its DFT", fontsize=25)
    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, "beta_noise_dft.png"))
    plt.show()


def plot_dft_of_interruption():
    # Configurations
    title_size = 17
    label_size = 14
    legend_size = 10

    # Define figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Create interrupted pdf
    interrupted_pdf = pdf.copy()
    start, length = 330, 70
    interrupted_pdf[start:start + length] = np.array(list(range(length // 2)) + list(range(length // 2, 0, -1))) / 20
    interrupted_pdf_dft = dft(interrupted_pdf)

    # Plot the original pdf
    ax[0, 0].plot(x, pdf, label="pdf", color="turquoise")
    ax[0, 0].set_title("Beta distribution", fontsize=title_size)
    ax[0, 0].set_ylabel("Density", fontsize=label_size)
    ax[0, 0].set_xlabel("x", fontsize=label_size)
    ax[0, 0].legend(fontsize=legend_size)

    # Plot the DFT of the original pdf
    ax[1, 0].plot(x, np.abs(pdf_dft), label="pdf DFT", color="turquoise")
    ax[1, 0].set_title("DFT of Beta distribution", fontsize=title_size)
    ax[1, 0].set_ylabel("Magnitude", fontsize=label_size)
    ax[1, 0].set_xlabel("Frequency", fontsize=label_size)
    ax[1, 0].legend(fontsize=legend_size)

    # Plot the interrupted pdf
    ax[0, 1].plot(x, interrupted_pdf, label="interrupted pdf", color="orange")
    ax[0, 1].set_title("Interrupted Beta distribution", fontsize=title_size)
    ax[0, 1].set_ylabel("Density", fontsize=label_size)
    ax[0, 1].set_xlabel("x", fontsize=label_size)
    ax[0, 1].legend(fontsize=legend_size)

    # Plot the DFT of the interrupted pdf
    ax[1, 1].plot(x, np.abs(interrupted_pdf_dft), label="interrupted pdf DFT", color="orange")
    ax[1, 1].set_title("DFT of Interrupted Beta distribution", fontsize=title_size)
    ax[1, 1].set_ylabel("Magnitude", fontsize=label_size)
    ax[1, 1].set_xlabel("Frequency", fontsize=label_size)
    ax[1, 1].legend(fontsize=legend_size)

    # Plot the original pdf for comparison
    plt.suptitle("Effects of Interruption on Beta Distribution and its DFT", fontsize=25)
    plt.tight_layout()
    plt.savefig(join(SAVE_DIR, "beta_interruption_dft.png"))
    plt.show()


def main():
    mirrored_pdf = shifted_pdf[0.1]
    mirrored_pdf_dft = dft(mirrored_pdf)
    plt.subplots(3, 2, figsize=(15, 10))
    plt.subplot(3, 2, 1)
    plt.plot(x, pdf, label="pdf", color="turquoise")
    plt.subplot(3, 2, 2)
    plt.plot(x, mirrored_pdf, label="mirrored pdf", color="coral")
    plt.subplot(3, 2, 3)
    plt.plot(x, np.abs(pdf_dft), label="pdf DFT", color="turquoise")
    plt.subplot(3, 2, 4)
    plt.plot(x, np.abs(mirrored_pdf_dft), label="mirrored pdf DFT", color="coral")
    plt.subplot(3, 2, 5)

    # Get the phase of the original pdf DFT
    phase = np.angle(pdf_dft)
    plt.plot(x, phase,
             label="pdf DFT", color="turquoise")
    plt.subplot(3, 2, 6)
    phase2 = np.angle(mirrored_pdf_dft)
    plt.plot(x, phase2, label="mirrored pdf DFT", color="coral")
    plt.show()

    plt.plot(x, phase - phase2, label="phase difference", color="purple")
    plt.show()


if __name__ == "__main__":
    main()
