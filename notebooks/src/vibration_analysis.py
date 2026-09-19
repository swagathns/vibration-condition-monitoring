"""
Basic vibration signal analysis.

Reads a vibration CSV file and calculates:
- RMS
- Peak
- Crest Factor
- FFT
"""

import numpy as np
import pandas as pd


def calculate_rms(signal):
    """Calculate RMS value."""
    return np.sqrt(np.mean(signal**2))


def calculate_peak(signal):
    """Calculate peak amplitude."""
    return np.max(np.abs(signal))


def calculate_crest_factor(signal):
    """Calculate crest factor."""
    rms = calculate_rms(signal)
    peak = calculate_peak(signal)

    return peak / rms if rms != 0 else 0


def calculate_fft(signal, sampling_frequency):
    """Calculate single-sided FFT spectrum."""

    n = len(signal)

    frequencies = np.fft.rfftfreq(n, d=1 / sampling_frequency)
    spectrum = np.abs(np.fft.rfft(signal)) * 2 / n

    return frequencies, spectrum


if __name__ == "__main__":

    # Load vibration data
    data = pd.read_csv("../data/sample_vibration.csv")

    vibration = data["vibration"].values
    time = data["time_s"].values

    # Sampling frequency
    sampling_frequency = 1 / (time[1] - time[0])

    # Time-domain features
    rms = calculate_rms(vibration)
    peak = calculate_peak(vibration)
    crest_factor = calculate_crest_factor(vibration)

    # Frequency-domain analysis
    frequencies, spectrum = calculate_fft(
        vibration,
        sampling_frequency
    )

    # Dominant frequency
    dominant_index = np.argmax(spectrum[1:]) + 1
    dominant_frequency = frequencies[dominant_index]

    print("VIBRATION ANALYSIS")
    print("------------------")
    print(f"Sampling Frequency : {sampling_frequency:.0f} Hz")
    print(f"Number of Samples  : {len(vibration)}")
    print(f"RMS                : {rms:.4f}")
    print(f"Peak               : {peak:.4f}")
    print(f"Crest Factor       : {crest_factor:.4f}")
    print(f"Dominant Frequency : {dominant_frequency:.2f} Hz")
