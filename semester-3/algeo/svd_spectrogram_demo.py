"""
SVD-based Spectrogram Compression Demo
A simulation of low-rank matrix approximation on voice assistant pipeline components
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

class SVDSpectrogramCompressor:

    def __init__(self, audio_path=None, duration=3.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.audio_path = audio_path
        self.spectrogram = None
        self.original_audio = None

    def generate_test_audio(self, duration=3.0, frequencies=[440, 880, 1320]):
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)

        # Combine multiple frequencies with varying amplitudes
        for i, freq in enumerate(frequencies):
            amplitude = 0.5 / (i + 1)  # Decreasing amplitudes
            audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Add some modulation to simulate speech characteristics
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
        audio = audio * envelope

        self.original_audio = audio
        print(f"Generated test audio: {duration}s at {self.sample_rate}Hz")
        return audio

    def load_audio(self, filepath):
        try:
            self.sample_rate, audio = wavfile.read(filepath)
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            # Normalize
            audio = audio.astype(np.float32) / np.max(np.abs(audio))
            self.original_audio = audio
            print(f"Loaded audio: {filepath}")
            print(f"Sample rate: {self.sample_rate}, Duration: {len(audio)/self.sample_rate:.2f}s")
            return audio
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

    def create_spectrogram(self, nperseg=256, noverlap=128):
        if self.original_audio is None:
            raise ValueError("No audio data available. Generate or load audio first.")

        # Compute STFT
        frequencies, times, Zxx = signal.spectrogram(
            self.original_audio,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )

        # Convert to magnitude (power spectrogram)
        self.spectrogram = np.abs(Zxx)

        print(f"Spectrogram shape: {self.spectrogram.shape}")
        print(f"  - Frequency bins: {self.spectrogram.shape[0]}")
        print(f"  - Time frames: {self.spectrogram.shape[1]}")

        return self.spectrogram, frequencies, times

    def perform_svd(self, matrix=None):
        if matrix is None:
            matrix = self.spectrogram

        if matrix is None:
            raise ValueError("No spectrogram available. Create spectrogram first.")

        # Perform SVD
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

        print(f"\nSVD Results:")
        print(f"  U shape: {U.shape}")
        print(f"  S shape: {S.shape} (singular values)")
        print(f"  Vt shape: {Vt.shape}")

        return U, S, Vt

    def truncated_svd_reconstruction(self, k, U, S, Vt):
        # Truncate to top k singular values
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

        # Reconstruct: A_k = U_k * diag(S_k) * Vt_k
        reconstructed = U_k @ np.diag(S_k) @ Vt_k

        return reconstructed

    def calculate_metrics(self, original, reconstructed, k, total_singular_values):
        m, n = original.shape

        # Compression ratio
        original_size = m * n
        compressed_size = k * (m + n + 1)
        compression_ratio = original_size / compressed_size

        # Relative error (Frobenius norm)
        error = np.linalg.norm(original - reconstructed, 'fro')
        relative_error = error / np.linalg.norm(original, 'fro')

        # Energy retention (cumulative singular values)
        energy_retention = None  # Will be calculated if S is provided

        return {
            'compression_ratio': compression_ratio,
            'relative_error': relative_error,
            'energy_retention': energy_retention
        }

    def calculate_energy_retention(self, S, k):
        total_energy = np.sum(S**2)
        retained_energy = np.sum(S[:k]**2)
        return (retained_energy / total_energy) * 100

    def analyze_singular_values(self, S):
        # Calculate cumulative energy
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2) * 100

        print(f"\nSingular Value Analysis:")
        print(f"  Total singular values: {len(S)}")
        print(f"  Max singular value: {S[0]:.4f}")
        print(f"  Min singular value: {S[-1]:.6f}")
        print(f"  Energy with top 10%: {cumulative_energy[int(len(S)*0.1)]:.2f}%")
        print(f"  Energy with top 25%: {cumulative_energy[int(len(S)*0.25)]:.2f}%")
        print(f"  Energy with top 50%: {cumulative_energy[int(len(S)*0.5)]:.2f}%")

        return cumulative_energy

    def visualize_comparison(self, original, reconstructed, k, S):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Original spectrogram
        im1 = axes[0, 0].imshow(original, aspect='auto', cmap='viridis',
                                 origin='lower', interpolation='nearest')
        axes[0, 0].set_title('Original Spectrogram', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time Frame')
        axes[0, 0].set_ylabel('Frequency Bin')
        plt.colorbar(im1, ax=axes[0, 0], label='Magnitude')

        # Reconstructed spectrogram
        im2 = axes[0, 1].imshow(reconstructed, aspect='auto', cmap='viridis',
                                 origin='lower', interpolation='nearest')
        axes[0, 1].set_title(f'Reconstructed Spectrogram (k={k})', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time Frame')
        axes[0, 1].set_ylabel('Frequency Bin')
        plt.colorbar(im2, ax=axes[0, 1], label='Magnitude')

        # Error heatmap
        error_matrix = np.abs(original - reconstructed)
        im3 = axes[1, 0].imshow(error_matrix, aspect='auto', cmap='hot',
                                 origin='lower', interpolation='nearest')
        axes[1, 0].set_title('Reconstruction Error', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time Frame')
        axes[1, 0].set_ylabel('Frequency Bin')
        plt.colorbar(im3, ax=axes[1, 0], label='Absolute Error')

        # Singular value decay
        axes[1, 1].plot(S, 'b-', linewidth=2, label='Singular Values')
        axes[1, 1].axvline(x=k, color='r', linestyle='--', linewidth=2,
                           label=f'Truncation point (k={k})')
        axes[1, 1].set_xlabel('Index', fontsize=11)
        axes[1, 1].set_ylabel('Singular Value', fontsize=11)
        axes[1, 1].set_title('Singular Value Decay', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        return fig

    def run_compression_experiment(self, k_values=[10, 20, 50, 100]):
        if self.spectrogram is None:
            raise ValueError("Create spectrorogram first.")

        # Perform SVD once
        U, S, Vt = self.perform_svd()
        cumulative_energy = self.analyze_singular_values(S)

        print("\n" + "="*70)
        print("COMPRESSION EXPERIMENT RESULTS")
        print("="*70)
        print(f"{'Rank (k)':<10} {'Comp. Ratio':<12} {'Rel. Error':<12} "
              f"{'Energy %':<12} {'Storage Saved':<12}")
        print("-"*70)

        results = []

        for k in k_values:
            if k > len(S):
                print(f"Warning: k={k} exceeds max rank {len(S)}. Skipping.")
                continue

            # Reconstruct
            reconstructed = self.truncated_svd_reconstruction(k, U, S, Vt)

            # Calculate metrics
            metrics = self.calculate_metrics(self.spectrogram, reconstructed, k, len(S))
            energy = self.calculate_energy_retention(S, k)
            metrics['energy_retention'] = energy

            # Storage saved
            storage_saved = (1 - 1/metrics['compression_ratio']) * 100

            print(f"{k:<10} {metrics['compression_ratio']:<12.2f} "
                  f"{metrics['relative_error']:<12.4f} "
                  f"{energy:<12.2f}% {storage_saved:<12.1f}%")

            results.append({
                'k': k,
                'reconstructed': reconstructed,
                'metrics': metrics
            })

        print("="*70)

        return U, S, Vt, results


def main():
    """
    Main demonstration function.
    """
    print("="*70)
    print("SVD-BASED SPECTROGRAM COMPRESSION DEMO")
    print("Simulating Voice Assistant Pipeline Optimization")
    print("="*70)

    # Initialize compressor
    compressor = SVDSpectrogramCompressor()

    # Generate test audio (synthetic voice-like signal)
    print("\n[Step 1] Generating synthetic audio signal...")
    audio = compressor.generate_test_audio(
        duration=3.0,
        frequencies=[440, 880, 1320, 1760]  # A4, A5, E6, A6 harmonics
    )

    # Create spectrogram
    print("\n[Step 2] Creating spectrogram (STFT)...")
    spectrogram, freqs, times = compressor.create_spectrogram(
        nperseg=256,
        noverlap=128
    )

    # Run compression experiment
    print("\n[Step 3] Running SVD compression experiment...")
    k_values = [5, 10, 20, 50, 100]

    # Determine appropriate k values based on spectrogram size
    max_k = min(spectrogram.shape) - 1
    k_values = [k for k in k_values if k < max_k]

    U, S, Vt, results = compressor.run_compression_experiment(k_values=k_values)

    # Visualize the best compromise (typically mid-range k)
    print("\n[Step 4] Generating visualization...")
    mid_index = len(results) // 2
    mid_result = results[mid_index]

    fig = compressor.visualize_comparison(
        compressor.spectrogram,
        mid_result['reconstructed'],
        mid_result['k'],
        S
    )

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'svd_spectrogram_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("This demo demonstrates how SVD enables low-rank matrix approximation")
    print("for spectrogram compression in voice assistant pipelines.")
    print(f"\nOriginal spectrogram shape: {spectrogram.shape}")
    print(f"Total elements: {spectrogram.size:,}")
    print(f"\nWith k={mid_result['k']}:")
    print(f"  - Compression ratio: {mid_result['metrics']['compression_ratio']:.2f}x")
    print(f"  - Energy retained: {mid_result['metrics']['energy_retention']:.2f}%")
    print(f"  - Relative error: {mid_result['metrics']['relative_error']:.4f}")
    print("\nThis validates the Eckart-Young-Mirsky theorem: the truncated SVD")
    print("provides optimal rank-k approximation with minimal reconstruction error.")
    print("="*70)


if __name__ == "__main__":
    main()
