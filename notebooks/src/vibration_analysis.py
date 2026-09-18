"""
Basic vibration signal analysis.

This module will be used to calculate time-domain
and frequency-domain features from vibration data.
"""

import numpy as np


def calculate_rms(signal):
    """Calculate RMS value of a vibration signal."""
    return np.sqrt(np.mean(signal**2))


def calculate_peak(signal):
    """Calculate peak amplitude of a vibration signal."""
    return np.max(np.abs(signal))


def calculate_crest_factor(signal):
    """Calculate crest factor."""
    rms = calculate_rms(signal)
    peak = calculate_peak(signal)

    return peak / rms if rms != 0 else 0
