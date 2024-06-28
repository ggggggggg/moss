import numpy as np
import pylab as plt
from dataclasses import dataclass

def fourier_filter(avg_signal, noise_psd, dt, fmax=None, f_3db=None, peak_signal=1.0):
    filter, variance = calc_fourier_filter(avg_signal, noise_psd, dt, fmax, f_3db, peak_signal)
    return Filter(filter, variance, dt, filter_type="fourier")

@dataclass
class Filter:
    filter: np.ndarray
    variance: float
    dt: float
    filter_type: str

    def plot(self, axis=None, **plotkwarg):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        axis.plot(self.frequencies(), self.filter, label="fourier filter", **plotkwarg)
        axis.grid()
        axis.set_title(f"{self.filter_type=}")
        axis.set_ylabel("filter value")
        axis.set_xlabel("Lag Time (s)")
        axis.figure.tight_layout()

    def frequencies(self):
        n = len(self.filter)
        return np.arange(0, n, dtype=float) * 0.5 / ((n - 1) * self.dt)

    def __call__(self, pulse):
        return np.dot(self.filter, pulse)

def apply_fmax(signal_freq_domain, fmax, dt):
    if fmax is None:
        return signal_freq_domain
    freq = np.arange(0, n, dtype=float) * 0.5 / ((n - 1) * sample_time)
    out = signal_freq_domain[:]
    out[freq>fmax]=0
    return out

def apply_f_3db(signal_freq_domain, f_3db, dt):
    if f_3db is None:
        return signal_freq_domain
    freq = np.arange(0, n, dtype=float) * 0.5 / ((n - 1) * sample_time)
    return signal_freq_domain / (1 + (freq * 1.0 / f_3db)**2)

def normalize_filter(filter):
    return filter/np.sqrt(np.dot(filter, filter))

def calc_fourier_filter(avg_signal, noise_psd, dt, fmax=None, f_3db=None, peak_signal=1.0):
    """Compute the Fourier-domain filter and variances for signal processing.

    Args:
    - avg_signal (np.ndarray): Average signal data.
    - noise_psd (np.ndarray): Power Spectral Density (PSD) of the noise.
    - shorten (int, optional): Amount to shorten the signal by (default: 0).
    - fmax (float, optional): Maximum frequency limit (default: None).
    - f_3db (float, optional): 3dB frequency limit (default: None).
    - sample_time (float, optional): Sampling time (default: 1.0).
    - peak_signal (float, optional): Peak signal value (default: 1.0).

    Returns:
    - filt_fourier (np.ndarray): Filtered signal in Fourier domain.
    - variances (dict): Dictionary containing variance 'fourier'.
    """
    signal_freq_domain1 = np.fft.rfft(avg_signal)

    if len(signal_freq_domain1) != len(noise_psd):
        raise ValueError(f"Signal DFT and noise PSD are not the same length ({len(signal_freq_domain1)} and {len(noise_psd)})")
    n = len(signal_freq_domain1)
    signal_freq_domain2 = signal_freq_domain1 / noise_psd

    signal_freq_domain3 = apply_fmax(signal_freq_domain2, fmax, dt)
    signal_freq_domain4 = apply_f_3db(signal_freq_domain3, f_3db, dt)

    filter = np.fft.irfft(signal_freq_domain4)

    noise_ft_squared = (len(noise_psd) - 1) / dt * noise_psd
    kappa = (np.abs(signal_freq_domain4 * peak_signal)**2 / noise_ft_squared)[1:].sum()
    variance = 1 / kappa
    return normalize_filter(filter), variance

