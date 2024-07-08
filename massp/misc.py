import numpy as np

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
    assert all(d == step_size)
    return x[:-1] + step_size, step_size

def hist_of_series(series, bin_edges):
    bin_centers, step_size = midpoints_and_step_size(bin_edges)
    counts = series.rename("count").hist(
        bin_edges, include_category=False, include_breakpoint=False
    )[1:-1]
    return bin_centers, counts.to_numpy().T[0]