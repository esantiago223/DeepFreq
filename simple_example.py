"""
Simple Example: Using DeepFreq for Frequency Estimation

This script demonstrates how to:
1. Generate a synthetic signal with multiple frequency components
2. Add noise to the signal
3. Use pre-trained DeepFreq models to estimate frequencies
4. Visualize the results

Usage:
    python simple_example.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal

# Import DeepFreq modules
import util
from data.fr import find_freq
from data.noise import noise_torch


def generate_test_signal(signal_dim=50, frequencies=None, amplitudes=None, snr_db=20):
    """
    Generate a test signal with specified frequencies

    Args:
        signal_dim: Number of samples in the signal
        frequencies: List of normalized frequencies in [-0.5, 0.5]
        amplitudes: List of amplitudes for each frequency
        snr_db: Signal-to-noise ratio in dB

    Returns:
        clean_signal: Clean signal without noise
        noisy_signal: Signal with added Gaussian noise
        true_freqs: True frequency values
    """
    if frequencies is None:
        # Default: 3 frequencies with some separation
        frequencies = [-0.15, 0.05, 0.25]

    if amplitudes is None:
        # Default: equal amplitudes
        amplitudes = [1.0] * len(frequencies)

    # Time samples
    t = np.arange(signal_dim)

    # Generate complex sinusoidal signal
    clean_signal = np.zeros(signal_dim, dtype=complex)
    for freq, amp in zip(frequencies, amplitudes):
        clean_signal += amp * np.exp(2j * np.pi * freq * t)

    # Convert to DeepFreq format: [batch_size, 2, signal_dim]
    # where dimension 1 contains [real_part, imag_part]
    signal_formatted = np.stack([clean_signal.real, clean_signal.imag], axis=0)
    signal_tensor = torch.tensor(signal_formatted, dtype=torch.float32).unsqueeze(0)

    # Add noise
    snr_linear = np.exp(np.log(10) * snr_db / 10)
    noisy_signal_tensor = noise_torch(signal_tensor, snr_linear, 'gaussian')

    return signal_tensor, noisy_signal_tensor, frequencies


def load_pretrained_models():
    """
    Load pre-trained DeepFreq models

    Returns:
        fr_module: Frequency representation module
        xgrid: Frequency grid for the representation

    Note: FC module is skipped due to PyTorch version compatibility issues
    """
    print("Loading pre-trained models...")

    fr_path = 'pretrained_models/DeepFreq/frequency_representation_module.pth'

    # Check if models exist
    if not os.path.exists(fr_path):
        raise FileNotFoundError(
            f"Frequency representation model not found at {fr_path}\n"
            "Please ensure you have the pretrained_models directory."
        )

    # Determine device (use CPU for compatibility)
    device = torch.device('cpu')

    # Load frequency representation module
    fr_module, _, _, _, _ = util.load(fr_path, 'fr', device=device)
    fr_module.eval()
    fr_module.cpu()

    # Create frequency grid
    xgrid = np.linspace(-0.5, 0.5, fr_module.fr_size, endpoint=False)

    print(f"Models loaded successfully!")
    print(f"FR module output size: {fr_module.fr_size}")
    print(f"Note: FC module skipped due to PyTorch version compatibility")

    return fr_module, xgrid


def estimate_frequencies(signal, fr_module, xgrid, known_num_freqs=None):
    """
    Estimate frequencies from a signal using DeepFreq

    Args:
        signal: Input signal tensor
        fr_module: Frequency representation module
        xgrid: Frequency grid
        known_num_freqs: Known number of frequencies (if None, will try to estimate from peaks)

    Returns:
        estimated_freqs: Estimated frequency values
        num_freqs: Number of frequencies used
        freq_representation: Frequency representation from FR module
    """
    with torch.no_grad():
        # Get frequency representation
        freq_representation = fr_module(signal)

        # If number of frequencies is not provided, estimate from peaks
        if known_num_freqs is None:
            # Simple peak detection: count significant peaks
            fr_numpy = freq_representation.cpu().numpy()[0]
            threshold = 0.3 * np.max(fr_numpy)  # 30% of max value
            peaks = []
            for i in range(1, len(fr_numpy) - 1):
                if fr_numpy[i] > fr_numpy[i-1] and fr_numpy[i] > fr_numpy[i+1] and fr_numpy[i] > threshold:
                    peaks.append(i)
            num_freqs = len(peaks)
            num_freqs = max(1, min(num_freqs, 10))  # Limit to 1-10 frequencies
        else:
            num_freqs = known_num_freqs

        # Extract frequency estimates
        estimated_freqs = find_freq(
            freq_representation.cpu().numpy(),
            np.array([num_freqs]),
            xgrid
        )

    return estimated_freqs[0], num_freqs, freq_representation.cpu().numpy()[0]


def visualize_results(clean_signal, noisy_signal, true_freqs, estimated_freqs,
                     freq_representation, xgrid, snr_db):
    """
    Visualize the signal and frequency estimation results
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Extract real parts for visualization
    # Signal shape is [batch_size, 2, signal_dim] where dimension 1 is [real, imag]
    clean_real = clean_signal[0, 0].numpy()
    noisy_real = noisy_signal[0, 0].numpy()
    signal_dim = clean_real.shape[0]

    # Plot 1: Time domain signals
    t = np.arange(signal_dim)
    axes[0].plot(t, clean_real, 'b-', label='Clean signal', linewidth=2)
    axes[0].plot(t, noisy_real, 'r--', label=f'Noisy signal (SNR={snr_db}dB)', alpha=0.7)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Frequency representation from DeepFreq
    axes[1].plot(xgrid, freq_representation, 'b-', linewidth=1.5)
    axes[1].set_xlabel('Normalized Frequency')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('DeepFreq Frequency Representation')
    axes[1].grid(True, alpha=0.3)

    # Mark estimated frequencies
    for freq in estimated_freqs:
        if freq >= -0.5:  # Valid frequency
            axes[1].axvline(x=freq, color='r', linestyle='--', alpha=0.7, linewidth=2)

    # Plot 3: Frequency comparison
    axes[2].stem(true_freqs, np.ones(len(true_freqs)), linefmt='b-', markerfmt='bo',
                basefmt=' ', label='True frequencies')

    valid_estimated = [f for f in estimated_freqs if f >= -0.5]
    if valid_estimated:
        axes[2].stem(valid_estimated, 0.8 * np.ones(len(valid_estimated)),
                    linefmt='r--', markerfmt='rs', basefmt=' ', label='Estimated frequencies')

    axes[2].set_xlabel('Normalized Frequency')
    axes[2].set_ylabel('Indicator')
    axes[2].set_title('Frequency Estimation Results')
    axes[2].set_ylim([0, 1.2])
    axes[2].set_xlim([-0.5, 0.5])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simple_example_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'simple_example_results.png'")
    plt.show()


def calculate_error_metrics(true_freqs, estimated_freqs):
    """
    Calculate error metrics between true and estimated frequencies
    """
    valid_estimated = [f for f in estimated_freqs if f >= -0.5]

    if len(valid_estimated) == 0:
        return float('inf'), 0

    # Chamfer distance (average minimum distance)
    chamfer_dist = 0
    for true_f in true_freqs:
        min_dist = min([abs(true_f - est_f) for est_f in valid_estimated])
        chamfer_dist += min_dist
    chamfer_dist /= len(true_freqs)

    # Counting accuracy
    counting_correct = (len(valid_estimated) == len(true_freqs))

    return chamfer_dist, counting_correct


def main():
    """
    Main function to run the example
    """
    print("=" * 60)
    print("DeepFreq Simple Example")
    print("=" * 60)

    # Configuration
    signal_dim = 50
    true_freqs = [-0.15, 0.05, 0.25]  # Three frequencies
    amplitudes = [1.0, 0.8, 1.2]  # Different amplitudes
    snr_db = 20  # 20 dB SNR

    print(f"\nConfiguration:")
    print(f"  Signal dimension: {signal_dim}")
    print(f"  True frequencies: {true_freqs}")
    print(f"  Amplitudes: {amplitudes}")
    print(f"  SNR: {snr_db} dB")

    # Generate test signal
    print("\nGenerating test signal...")
    clean_signal, noisy_signal, true_freqs = generate_test_signal(
        signal_dim=signal_dim,
        frequencies=true_freqs,
        amplitudes=amplitudes,
        snr_db=snr_db
    )

    # Load pre-trained models
    fr_module, xgrid = load_pretrained_models()

    # Estimate frequencies
    print("\nEstimating frequencies...")
    # We know there are 3 frequencies in this example
    estimated_freqs, num_freqs, freq_representation = estimate_frequencies(
        noisy_signal, fr_module, xgrid, known_num_freqs=len(true_freqs)
    )

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTrue frequencies: {true_freqs}")
    print(f"Number of true frequencies: {len(true_freqs)}")
    print(f"\nEstimated frequencies: {[f for f in estimated_freqs if f >= -0.5]}")
    print(f"Number of frequencies used: {num_freqs}")
    print(f"\nNote: In this example, we use the known number of frequencies.")
    print(f"The FC module is skipped due to PyTorch version compatibility issues.")

    # Calculate error metrics
    chamfer_dist, counting_correct = calculate_error_metrics(true_freqs, estimated_freqs)
    print(f"\nError Metrics:")
    print(f"  Chamfer distance: {chamfer_dist:.6f}")
    print(f"  Counting correct: {counting_correct}")

    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(
        clean_signal, noisy_signal, true_freqs, estimated_freqs,
        freq_representation, xgrid, snr_db
    )

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
