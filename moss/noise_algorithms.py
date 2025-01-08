import numpy as np
import pylab as plt #type: ignore
from dataclasses import dataclass
from numpy import ndarray
from typing import Tuple, Optional
import matplotlib.pyplot as plt

def calc_autocorrelation(data):
    ntraces, nsamples = data.shape
    ac = np.zeros(nsamples, dtype=float)

    for i in range(ntraces):
        pulse = data[i,:]
        pulse -= pulse.mean()
        ac += np.correlate(pulse, pulse, 'full')[nsamples - 1:]

    ac /= ntraces
    ac /= nsamples - np.arange(nsamples, dtype=float)
    return ac

def calc_autocorrelation_times(n, dt):
    return np.arange(n)*dt

def autocorrelation(data, dt):
    return AutoCorrelation(calc_autocorrelation(data), 
                           calc_autocorrelation_times(data.shape[1], dt))

@dataclass
class AutoCorrelation:
    ac: np.ndarray
    times: np.ndarray
    

    def plot(self, axis=None, **plotkwarg):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        axis.plot(self.times[1:], self.ac[1:], **plotkwarg)
        axis.grid()
        axis.set_ylabel("Autocorrelation (abrs?)")
        axis.set_xlabel("Lag Time (s)")
        axis.figure.tight_layout()

def psd_2d(Nt: ndarray, dt: float) -> ndarray:
    # Nt is size (n,m) with m records of length n
    (n, m) = Nt.shape
    df = 1 / n / dt  # the frequency bin spacing of the rfft
    # take the absolute value of the rfft of each record, then average all records
    Nabs = np.mean(np.abs(np.fft.rfft(Nt, axis=0)), axis=1)
    # PSD = 2*Nabs^2/n/df
    # the 2 accounts for the power that would be in the negative frequency bins, due to use of rfft
    # n comes from parseval's theorm
    # df normalizes binsize since rfft doesn't know the bin size
    psd = 2 * Nabs**2 / n / df
    # Handle the DC component and Nyquist frequency differently (no factor of 2)
    psd[0] /= 2  # DC component
    if n % 2 == 0:  # If even number of samples
        psd[-1] /= 2  # Nyquist frequency
    return psd

def calc_autocorrelation_vec(data: ndarray) -> ndarray:
    import scipy.signal
    import time
    data = data[:10,:]
    ntraces, nsamples = data.shape
    ac = np.zeros(nsamples, dtype=float)

    for i in range(ntraces):
        pulse = data[i,:]
        pulse = pulse - pulse.mean() # don't use -= here in case input array is read only
        tstart = time.time()
        ac += scipy.signal.correlate(pulse, pulse, 'full')[nsamples - 1:]
        elapsed_s = time.time() - tstart
        if elapsed_s*ntraces > 10:
            print(f"Warning: autocorrelation is slow, projected to take {elapsed_s*ntraces:.2f} s for {ntraces} traces")
        # was np.correlate before, but that is slower due to not using fft
        # should give the same result

    ac /= ntraces
    return ac

def calc_psd_frequencies(nbins: int, dt: float) -> ndarray:
    return np.arange(nbins, dtype=float) / (2 * dt * nbins)

def noise_psd(data: ndarray, dt: float, window: None=None) -> "NoisePSD":
    assert window is None, "windowing not implemented"
    psd = psd_2d(data.T, dt) 
    nbins = len(psd)
    autocorr_vec = calc_autocorrelation_vec(data)
    frequencies = calc_psd_frequencies(nbins, dt)
    return NoisePSD(psd=psd,
                    autocorr_vec=autocorr_vec,
                    frequencies=frequencies)

@dataclass
class NoisePSD:
    psd: np.ndarray
    autocorr_vec: np.ndarray
    frequencies: np.ndarray
    

    def plot(self, axis: Optional[plt.axis]=None, arb_to_unit_scale_and_label: Tuple[int, str]=(1, "arb"), sqrt_psd: bool=True, loglog: bool=True, **plotkwarg):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        arb_to_unit_scale, unit_label = arb_to_unit_scale_and_label
        psd = self.psd[1:] * (arb_to_unit_scale**2)
        freq = self.frequencies[1:]
        if sqrt_psd:
            axis.plot(freq, np.sqrt(psd), **plotkwarg)
            axis.set_ylabel(f"Amplitude Spectral Density ({unit_label}$/\\sqrt{{Hz}}$)")
        else:
            axis.plot(freq, psd, **plotkwarg)
            axis.set_ylabel(f"Power Spectral Density ({unit_label}$^2$ Hz$^{{-1}}$)")
        if loglog: plt.loglog()
        axis.grid()
        axis.set_xlabel("Frequency (Hz)")
        plt.title(f"noise from records of length {len(self.frequencies)*2-2}")
        axis.figure.tight_layout()


