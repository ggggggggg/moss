import numpy as np
import numba
import scipy as sp
import scipy.optimize
from dataclasses import dataclass, field
import mass

def drift_correct(indicator, uncorrected):
    slope, dc_info = \
            mass.core.analysis_algorithms.drift_correct(indicator, uncorrected)
    offset = dc_info["median_pretrig_mean"]
    return DriftCorrection(slope=slope, offset=offset)

# def compute_kernel_ft(nbins, stepsize, smooth_sigma):
#     kernel = np.exp(-0.5 * (np.arange(nbins) * stepsize / smooth_sigma) ** 2)
#     kernel[1:] += kernel[-1:0:-1]  # Handle the negative frequencies
#     kernel /= kernel.sum()
#     return np.fft.rfft(kernel)

# def compute_smooth_histogram(values, nbins, limits, kernel_ft):
#     contents, _ = np.histogram(values, nbins, limits)
#     ftc = np.fft.rfft(contents)
#     csmooth = np.fft.irfft(kernel_ft * ftc)
#     csmooth[csmooth < 0] = 0
#     return csmooth

# @dataclass
# class HistogramSmoother:
#     smooth_sigma: float
#     limits: np.ndarray
#     nbins: int = field(init=False)
#     stepsize: float = field(init=False)
#     kernel_ft: np.ndarray = field(init=False)

#     def __post_init__(self):
#         self.limits = np.asarray(self.limits, dtype=float)

#         stepsize = 0.4 * self.smooth_sigma
#         dlimits = self.limits[1] - self.limits[0]
#         nbins = int(dlimits / stepsize + 0.5)
#         pow2 = 1024
#         while pow2 < nbins:
#             pow2 *= 2
#         self.nbins = pow2
#         self.stepsize = dlimits / self.nbins

#         self.kernel_ft = compute_kernel_ft(self.nbins, self.stepsize, self.smooth_sigma)

#     def __call__(self, values):
#         return compute_smooth_histogram(values, self.nbins, self.limits, self.kernel_ft)

# def make_smooth_histogram(values, smooth_sigma, limit, upper_limit=None):
#     if upper_limit is None:
#         limit, upper_limit = 0, limit
#     smoother = HistogramSmoother(smooth_sigma, [limit, upper_limit])
#     return smoother(values)

# def drift_correct_entropy(param, indicator, uncorrected, smoother):
#     corrected = uncorrected * (1 + indicator * param)
#     hsmooth = smoother(corrected)
#     w = hsmooth > 0
#     return -(np.log(hsmooth[w]) * hsmooth[w]).sum()

# def drift_correct(indicator, uncorrected, limit=None):
#     offset = np.median(indicator)
#     indicator = np.array(indicator) - offset

#     if limit is None:
#         pct99 = np.percentile(uncorrected, 99)
#         limit = 1.25 * pct99

#     smoother = HistogramSmoother(0.5, [0, limit])

#     def entropy(param):
#         return drift_correct_entropy(param, indicator, uncorrected, smoother)

#     slope = sp.optimize.brent(entropy, brack=[0, .001])

#     return DriftCorrection(slope=slope, offset=offset)

@dataclass
class DriftCorrection:
    offset: float
    slope: float

    def __call__(self, uncorrected):
        return uncorrected*(1+(indicator-self.offset)*self.slope)

