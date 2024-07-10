import numpy as np
import pylab as plt

def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


def sigma_mad(x):
    return median_absolute_deviation(x) * 1.4826


def outlier_resistant_nsigma_above_mid(x, nsigma=5):
    mid = np.median(x)
    mad = np.median(np.abs(x - mid))
    sigma_mad = mad * 1.4826
    return mid + nsigma * sigma_mad

def midpoints_and_step_size(x):
    d = np.diff(x)
    step_size = d[0]
    assert np.allclose(d, step_size, atol=1e-9)
    return x[:-1] + step_size, step_size

def hist_of_series(series, bin_edges):
    bin_centers, step_size = midpoints_and_step_size(bin_edges)
    counts = series.rename("count").hist(
        bin_edges, include_category=False, include_breakpoint=False
    )[1:-1]
    return bin_centers, counts.to_numpy().T[0]

def plot_hist_of_series(series, bin_edges, axis=None, **plotkwarg):
    if axis is None:
        plt.figure()
        axis = plt.gca()
    bin_centers, step_size = midpoints_and_step_size(bin_edges)
    hist = series.rename("count").hist(
        bin_edges, include_category=False, include_breakpoint=False
    )[1:-1]
    axis.plot(bin_centers, hist, label=series.name, **plotkwarg)
    axis.set_xlabel(series.name)
    axis.set_ylabel(f"counts per {step_size:.2f} unit bin")
    return axis

def plot_a_vs_b_series(a, b, axis=None, **plotkwarg):
    if axis is None:
        plt.figure()
        axis = plt.gca()
    axis.plot(a, b, ".", label=b.name, **plotkwarg)
    axis.set_xlabel(a.name)
    axis.set_ylabel(b.name)