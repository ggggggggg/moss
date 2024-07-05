import marimo

__generated_with = "0.6.26"
app = marimo.App(width="medium", app_title="Massp Example")


@app.cell
def __():
    import massp
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    return massp, mo, np, pl, plt


@app.cell
def __(massp):
    noise_path = r"C:\Users\oneilg\Desktop\python\src\mass\tests\ljh_files\20230626\0000\20230626_run0000_chan4102.ljh"
    pulse_path = r"C:\Users\oneilg\Desktop\python\src\mass\tests\ljh_files\20230626\0001\20230626_run0001_chan4102.ljh"

    ljh_noise = massp.LJHFile(noise_path)
    df_noise, header_df_noise = ljh_noise.to_polars()
    ljh = massp.LJHFile(pulse_path)
    df, header_df = ljh.to_polars()
    return (
        df,
        df_noise,
        header_df,
        header_df_noise,
        ljh,
        ljh_noise,
        noise_path,
        pulse_path,
    )


@app.cell
def __(df, mo):
    mo.plain(df)
    return


@app.cell
def __(df, np, plt):
    peak_ind = np.argmax(df["pulse"][0])
    plt.plot(df["pulse"][0])
    plt.plot(peak_ind, df["pulse"][0].to_numpy()[peak_ind], "o")
    return peak_ind,


@app.cell
def __():
    return


@app.cell
def __(header_df):
    header_df
    return


@app.cell
def __(df, pl):
    df2 = df.with_columns(
        df.select(pl.from_epoch(pl.col("posix_usec"), time_unit="us").alias("timestamp"))
    )
    return df2,


@app.cell
def __(df, df2, header_df, massp, mo, peak_ind, pl):
    df3 = pl.concat(
        pl.from_numpy(
            massp.pulse_algorithms.summarize_data_numba(
                df_iter["pulse"].to_numpy(),
                header_df["Timebase"][0],
                peak_samplenumber=peak_ind,
                pretrigger_ignore=0,
                nPresamples=header_df["Presamples"][0],
            )
        )
        for df_iter in df.iter_slices()
    ).with_columns(df2)
    mo.plain(df3)
    return df3,


@app.cell
def __(df_noise, mo):
    mo.plain(df_noise)
    return


@app.cell
def __(df_noise):
    df_noise["pulse"].to_numpy().shape
    return


@app.cell
def __(df_noise, mo, np, pl):
    def excursion2d(noise_trace):
        return np.amax(noise_trace, axis=1) - np.amin(noise_trace, axis=1)


    def excursion(noise_trace):
        return np.amax(noise_trace) - np.amin(noise_trace)


    df_noise2 = df_noise.with_columns(
        df_noise["pulse"].map_elements(excursion, return_dtype=pl.Int64).alias("excursion")
    )
    mo.plain(df_noise2)
    return df_noise2, excursion, excursion2d


@app.cell
def __(np):
    def median_absolute_deviation(x):
        return np.median(np.abs(x - np.median(x)))


    def sigma_mad(x):
        return median_absolute_deviation(x) * 1.4826


    def outlier_resistant_nsigma_above_mid(x, nsigma=5):
        mid = np.median(x)
        mad = np.median(np.abs(x - mid))
        sigma_mad = mad * 1.4826
        return mid + nsigma * sigma_mad
    return (
        median_absolute_deviation,
        outlier_resistant_nsigma_above_mid,
        sigma_mad,
    )


@app.cell
def __(df_noise2, outlier_resistant_nsigma_above_mid):
    max_excursion = outlier_resistant_nsigma_above_mid(df_noise2["excursion"], nsigma=5)
    max_excursion
    return max_excursion,


@app.cell
def __(df_noise2, max_excursion, pl):
    # warning do not use select("pulse") as that returns a DF, we want to call to_numpy on a series
    noise_data = (
        df_noise2.filter(pl.col("excursion") < max_excursion)
        .limit(10000)["pulse"]
        .to_numpy()
    )
    return noise_data,


@app.cell
def __(noise_data):
    noise_data.shape
    return


@app.cell
def __(header_df, massp, noise_data):
    spectrum = massp.noise_psd(noise_data, dt=header_df["Timebase"][0])
    return spectrum,


@app.cell
def __(mo, plt, spectrum):
    spectrum.plot()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(df3, np, outlier_resistant_nsigma_above_mid, pl):
    # choose max pretrigger_rms and post_peak_deriv to find clean pulses for average pulse
    max_postpeak_deriv = outlier_resistant_nsigma_above_mid(
        df3["postpeak_deriv"].to_numpy(), nsigma=20
    )
    max_pretrig_rms = outlier_resistant_nsigma_above_mid(
        df3["pretrig_rms"].to_numpy(), nsigma=20
    )
    good_expr = (pl.col("postpeak_deriv") < max_postpeak_deriv).and_(
        pl.col("pretrig_rms") < max_pretrig_rms
    )
    df_good = df3.lazy().filter(good_expr)
    df_bad = df3.lazy().filter(good_expr.not_())
    df_bad_pretrig = df3.lazy().filter(pl.col("pretrig_rms") > max_pretrig_rms)
    _avg_pulse = (
        df_good.select("pulse").limit(1000).collect()["pulse"].to_numpy().mean(axis=0)
    )
    avg_pulse = (
        df_good.filter(
            np.abs(pl.col("pulse_rms") / np.median(pl.col("pulse_rms")) - 1) < 0.1
        )
        .select("pulse")
        .limit(1000)
        .collect()["pulse"]
        .to_numpy()
        .mean(axis=0)
    )
    return (
        avg_pulse,
        df_bad,
        df_bad_pretrig,
        df_good,
        good_expr,
        max_postpeak_deriv,
        max_pretrig_rms,
    )


@app.cell
def __(avg_pulse, df_bad, df_bad_pretrig, df_good, mo, plt):
    _fig, _axes = plt.subplots(2, 2)
    plt.sca(_axes[0, 0])
    plt.plot(df_good.select("pulse").limit(10).collect()["pulse"].to_numpy().T)
    plt.title("good pulses")
    plt.sca(_axes[0, 1])
    plt.plot(df_bad.select("pulse").limit(20).collect()["pulse"].to_numpy().T)
    plt.title("bad pulses")
    plt.sca(_axes[1, 0])
    plt.plot(avg_pulse)
    plt.title("avg pulse")
    plt.sca(_axes[1, 1])
    plt.plot(df_bad_pretrig.select("pulse").limit(10).collect()["pulse"].to_numpy().T)
    plt.title("bad pulses pretrig_only")
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(avg_pulse, header_df, massp, mo, plt, spectrum):
    filter = massp.fourier_filter(
        avg_signal=avg_pulse, noise_psd=spectrum.psd, dt=header_df["Timebase"][0]
    )
    # filter.filter-=np.mean(filter.filter)
    filter.plot()
    mo.mpl.interactive(plt.gcf())
    return filter,


@app.cell
def __(avg_pulse, df3, filter, mo, np, pl):
    avg_pulse_size = np.amax(avg_pulse) - avg_pulse[0]


    def apply_filter(pulse):
        return np.dot(filter.filter, pulse) * avg_pulse_size * 2e-5


    filt_value = (
        df3["pulse"]
        .map_elements(apply_filter, return_dtype=pl.Float64)
        .rename("filt_value")
    )
    df4 = df3.with_columns(filt_value)
    mo.plain(filt_value)
    return apply_filter, avg_pulse_size, df4, filt_value


@app.cell
def __(df4, np):
    def midpoints_and_step_size(x):
        d = np.diff(x)
        step_size = d[0]
        assert all(d == step_size)
        return x[:-1] + step_size, step_size


    bin_edges = np.arange(0, 10000, 10)
    hist = (
        df4["filt_value"]
        .rename("count")
        .hist(np.arange(0, 10000, 10), include_category=False, include_breakpoint=False)[
            1:-1
        ]
    )
    hist
    None  # None hides output
    return bin_edges, hist, midpoints_and_step_size


@app.cell
def __(df4, midpoints_and_step_size, np, plt):
    def plot_hist(series, bin_edges, axis=None, **plotkwarg):
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


    plot_hist(df4["filt_value"], np.arange(0, 10000, 10))
    return plot_hist,


@app.cell
def __(df4, good_expr, mo, plt):
    def plot_a_vs_b_series(a, b, axis=None, **plotkwarg):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        axis.plot(a, b, ".", label=b.name, **plotkwarg)
        axis.set_xlabel(a.name)
        axis.set_ylabel(b.name)


    _fig, _axes = plt.subplots(1, 2)
    plot_a_vs_b_series(
        df4.filter(good_expr)["pretrig_mean"],
        df4.filter(good_expr)["filt_value"],
        _axes[0],
    )
    plot_a_vs_b_series(
        df4.filter(good_expr)["timestamp"], df4.filter(good_expr)["pretrig_mean"], _axes[1]
    )
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return plot_a_vs_b_series,


@app.cell
def __(mo):
    mo.md("# drift correction")
    return


@app.cell
def __(df4, good_expr, massp):
    df_dc = df4.filter(good_expr).select("pretrig_mean", "filt_value")
    dc = massp.drift_correct(
        indicator=df_dc["pretrig_mean"].to_numpy(),
        uncorrected=df_dc["filt_value"].to_numpy(),
    )
    dc
    return dc, df_dc


@app.cell
def __(dc, df4, pl):
    m, b = dc.slope, dc.offset
    df5 = df4.select(
        (pl.col("filt_value") * (1 + m * (pl.col("pretrig_mean") - b))).alias(
            "filt_value_dc"
        )
    ).with_columns(df4)
    # df5 = df4.select("filt_value", gain=(1+m*(pl.col("pretrig_mean")-b))).select("gain",filt_value_dc=pl.col("filt_value")*pl.col("gain"))
    # df5
    # df4.select(pl.col("pretrig_mean")-b)
    return b, df5, m


@app.cell
def __(df5, good_expr, mo, plot_a_vs_b_series, plt):
    plot_a_vs_b_series(
        df5.filter(good_expr)["pretrig_mean"], df5.filter(good_expr)["filt_value"]
    )
    plot_a_vs_b_series(
        df5.filter(good_expr)["pretrig_mean"],
        df5.filter(good_expr)["filt_value_dc"],
        plt.gca(),
    )
    plt.legend()
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(df5, mo, np, plot_hist, plt):
    plot_hist(df5["filt_value"], np.arange(0, 10000, 1))
    plot_hist(df5["filt_value_dc"], np.arange(0, 10000, 1), axis=plt.gca())
    plt.legend()
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(df5, good_expr, mo, plot_a_vs_b_series, plt):
    _fig, _axes = plt.subplots(1, 2)
    plot_a_vs_b_series(
        df5.filter(good_expr)["promptness"],
        df5.filter(good_expr)["filt_value_dc"],
        _axes[0],
    )
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(df5, header_df, pl):
    ch = pl.Series("ch", dtype=pl.Int64).extend_constant(header_df["Channel"], len(df5))
    df_export = (
        df5.select("filt_value_dc", "timestamp")
        .with_columns(ch)
        .select("ch", "timestamp", "filt_value_dc")
    )  # re-order
    df_export
    return ch, df_export


@app.cell
def __(df_export):
    df_export.write_parquet("out.parquet")
    return


@app.cell
def __(mo):
    mo.md("#5lag filter")
    return


@app.cell
def __(np):
    def filter_data_5lag(filter_values, pulses):
        # These parameters fit a parabola to any 5 evenly-spaced points
        fit_array = (
            np.array(
                ((-6, 24, 34, 24, -6), (-14, -7, 0, 7, 14), (10, -5, -10, -5, 10)),
                dtype=float,
            )
            / 70.0
        )
        conv = np.zeros((5, pulses.shape[0]), dtype=float)
        conv[0, :] = np.dot(pulses[:, 0:-4], filter_values)
        conv[1, :] = np.dot(pulses[:, 1:-3], filter_values)
        conv[2, :] = np.dot(pulses[:, 2:-2], filter_values)
        conv[3, :] = np.dot(pulses[:, 3:-1], filter_values)
        conv[4, :] = np.dot(pulses[:, 4:], filter_values)

        param = np.dot(fit_array, conv)
        peak_x = -0.5 * param[1, :] / param[2, :]
        peak_y = param[0, :] - 0.25 * param[1, :] ** 2 / param[2, :]
        return peak_x, peak_y
    return filter_data_5lag,


@app.cell
def __(avg_pulse, header_df, massp, noise_data):
    spectrum5lag = massp.noise_psd(noise_data[:, 2:-2], dt=header_df["Timebase"][0])
    filter5lag = massp.fourier_filter(
        avg_signal=avg_pulse[2:-2],
        noise_psd=spectrum5lag.psd,
        dt=header_df["Timebase"][0],
        f_3db=10e3,
    )
    # filter5lag.filter -= np.mean(filter5lag.filter)
    return filter5lag, spectrum5lag


@app.cell
def __(df5, filter5lag, filter_data_5lag, pl):
    df_out = pl.DataFrame(schema={"peak_x": pl.Float64, "peak_y": pl.Float64})
    dfs = []
    for df_iter in df5.iter_slices(10000):
        peak_x, peak_y = filter_data_5lag(filter5lag.filter, df_iter["pulse"].to_numpy())
        dfs.append(pl.DataFrame({"peak_x": peak_x, "peak_y": peak_y}))
    df6 = pl.concat(dfs).with_columns(df5)
    df6
    return df6, df_iter, df_out, dfs, peak_x, peak_y


@app.cell
def __(df7, df8, good_expr, mo, plot_a_vs_b_series, plt):
    _fig, _axes = plt.subplots(1, 2)
    plot_a_vs_b_series(
        df8.filter(good_expr)["peak_x"],
        df7.filter(good_expr)["filt_value_5lag_dc"],
        _axes[0],
    )
    plot_a_vs_b_series(
        df7.filter(good_expr)["peak_x"], df7.filter(good_expr)["filt_value_dc"], _axes[1]
    )
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(filter, filter5lag, mo, plt):
    filter5lag.plot()
    filter.plot(axis=plt.gca())
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(b, df6, m, massp, pl):
    _dc = massp.drift_correct(
        indicator=df6["pretrig_mean"].to_numpy(),
        uncorrected=df6["peak_y"].to_numpy(),
    )
    _dc
    _m, _b = _dc.slope, _dc.offset
    df7 = df6.select(
        (pl.col("peak_y") * (1 + m * (pl.col("pretrig_mean") - b))).alias(
            "filt_value_5lag_dc"
        )
    ).with_columns(df6)
    return df7,


@app.cell
def __(mo):
    mo.md("# Rough calibration")
    return


@app.cell
def __(df7):
    import mass

    peak_ph_vals, _peak_heights = mass.algorithms.find_local_maxima(
        df7["filt_value_5lag_dc"].to_numpy(), gaussian_fwhm=200
    )
    name_e, energies_out, opt_assignments = mass.algorithms.find_opt_assignment(
        peak_ph_vals,
        line_names=["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
        maxacc=0.1,
    )
    return energies_out, mass, name_e, opt_assignments, peak_ph_vals


@app.cell
def __(df7, good_expr, mo, np, opt_assignments, plot_hist, plt):
    plot_hist(df7.filter(good_expr)["filt_value_5lag_dc"], np.arange(0, 70000, 10))
    plt.plot(opt_assignments, np.zeros(len(opt_assignments)), "o")
    plt.legend()
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(energies_out, mo, np, opt_assignments, plt):
    gain = opt_assignments / energies_out
    pfit_gain = np.polynomial.Polynomial.fit(opt_assignments, gain, deg=2)
    x = np.arange(0, 100000, 100)
    y = pfit_gain(x)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("filt_value_5lag_dc")
    plt.ylabel("gain")
    mo.mpl.interactive(plt.gcf())
    return gain, pfit_gain, x, y


@app.cell
def __(df7, good_expr, pfit_gain, pl):
    df8 = (
        df7.select(
            (pl.col("filt_value_5lag_dc") / pfit_gain(pl.col("filt_value_5lag_dc"))).alias(
                "energy_rough"
            )
        )
        .with_columns(df7)
        .filter(good_expr)
    )
    return df8,


@app.cell
def __(
    df8,
    energies_out,
    good_expr,
    mo,
    np,
    opt_assignments,
    plot_hist,
    plt,
):
    plot_hist(df8.filter(good_expr)["energy_rough"], np.arange(0, 10000, 1))
    plt.plot(energies_out, np.zeros(len(opt_assignments)), "o")
    plt.legend()
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(midpoints_and_step_size):
    def histo(series, bin_edges):
        bin_centers, step_size = midpoints_and_step_size(bin_edges)
        counts = series.rename("count").hist(
            bin_edges, include_category=False, include_breakpoint=False
        )[1:-1]
        return bin_centers, counts.to_numpy().T[0]
    return histo,


@app.cell
def __(df8, histo, mass, np):
    model = mass.get_model("MnKAlpha", has_linear_background=False, has_tails=False)
    dlo, dhi = 50, 50
    binsize = 0.5
    pe = model.spect.peak_energy
    _bin_edges = np.arange(pe - dlo, pe + dhi, binsize)
    bin_centers, counts = histo(df8["energy_rough"], _bin_edges)
    params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
    params["dph_de"].set(1.0, vary=False)
    result = model.fit(counts, params, bin_centers=bin_centers, minimum_bins_per_fwhm=3)
    return bin_centers, binsize, counts, dhi, dlo, model, params, pe, result


@app.cell
def __(mo):
    mo.md("# 3.16 eV fwhm")
    return


@app.cell
def __(mo, plt, result):
    result.plotm()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(df8, good_expr, mo, plot_a_vs_b_series, plt):
    _fig, _axes = plt.subplots(1, 2)
    plot_a_vs_b_series(
        df8.filter(good_expr)["pretrig_mean"],
        df8.filter(good_expr)["energy_rough"],
        _axes[0],
    )
    plot_a_vs_b_series(
        df8.filter(good_expr)["peak_x"], df8.filter(good_expr)["energy_rough"], _axes[1]
    )
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __():
    from dataclasses import dataclass, field
    import functools
    return dataclass, field, functools


@app.cell
def __(mo):
    mo.md("#1 channel with history")
    return


@app.cell
def __(
    dataclass,
    df_noise,
    excursion2d,
    functools,
    massp,
    outlier_resistant_nsigma_above_mid,
    pl,
):
    @dataclass(frozen=True)
    class NoiseChannel:
        df: pl.DataFrame | pl.LazyFrame  # DO NOT MUTATE THIS!!!
        header_df: pl.DataFrame | pl.LazyFrame  # DO NOT MUTATE THIS!!
        frametime_s: float

        @functools.cache
        def calc_max_excursion(
            self, trace_col_name="pulse", n_limit=10000, excursion_nsigma=5
        ):
            noise_traces = self.df.limit(n_limit)[trace_col_name].to_numpy()
            _excursion = excursion2d(noise_traces)
            max_excursion = outlier_resistant_nsigma_above_mid(
                _excursion, nsigma=excursion_nsigma
            )
            df_noise2 = df_noise.with_columns(excursion=_excursion)
            return df_noise2, max_excursion

        @functools.cache
        def spectrum(
            self,
            trace_col_name="pulse",
            n_limit=10000,
            excursion_nsigma=5,
            trunc_front=0,
            trunc_back=0,
        ):
            df_noise2, max_excursion = self.calc_max_excursion(
                trace_col_name, n_limit, excursion_nsigma
            )
            noise_traces_clean = (
                df_noise2.filter(pl.col("excursion") < max_excursion)
                .limit(10000)["pulse"]
                .to_numpy()
            )
            if trunc_back == 0:
                noise_traces_clean2 = noise_traces_clean[:, trunc_front:]
            elif trunc_back > 0:
                noise_traces_clean2 = noise_traces_clean[:, trunc_front:-trunc_back]
            else:
                raise ValueError(f"trunc_back must be >= 0")
            spectrum = massp.noise_psd(noise_traces_clean2, dt=self.frametime_s)
            return spectrum

        def __hash__(self):
            # needed to make functools.cache work
            # if self or self.anything is mutated, assumptions will be broken
            # and we may get nonsense results
            return hash(id(self))

        def __eq__(self, other):
            return id(self) == id(other)
    return NoiseChannel,


@app.cell
def __(dataclass, field, pl):
    import os


    @dataclass(frozen=True)
    class ChannelHeader:
        description: str  # filename or date/run number, etc
        ch_num: int
        frametime_s: float
        n_presamples: int
        n_samples: int
        df: pl.DataFrame | pl.LazyFrame = field(repr=False)

        @classmethod
        def from_ljh_header_df(cls, df):
            return ChannelHeader(
                description=os.path.split(df["Filename"][0])[-1],
                ch_num=df["Channel"][0],
                frametime_s=df["Timebase"][0],
                n_presamples=df["Presamples"][0],
                n_samples=df["Total Samples"][0],
                df=df,
            )
    return ChannelHeader, os


@app.cell
def __(
    Bool,
    CalSteps,
    ChannelHeader,
    Filter5LagStep,
    NoiseChannel,
    RoughCalibrationStep,
    SummarizeStep,
    dataclass,
    field,
    mass,
    massp,
    np,
    outlier_resistant_nsigma_above_mid,
    peak_ind,
    pl,
):
    @dataclass(frozen=True)
    class Channel:
        df: pl.DataFrame | pl.LazyFrame = field(repr=False)
        header: ChannelHeader = field(repr=True)
        noise: NoiseChannel = field(repr=False)
        good_expr: Bool | pl.Expr = True
        df_history: list[pl.DataFrame | pl.LazyFrame] = field(
            default_factory=list, repr=False
        )
        cal: CalSteps = field(default_factory=CalSteps.new_empty)

        def dbg_plot(self, step_ind):
            step = self.cal[step_ind]
            if step_ind + 1 == len(self.df_history):
                df_after = self.df
            else:
                df_after = self.df_history[step_ind]
            return step.dbg_plot(df_after)

        def plot_hist(self, col, bin_edges, axis=None):
            return plot_hist(self.df[col], bin_edges, axis)

        def rough_cal(
            self, line_names, uncalibrated_col, calibrated_col, ph_smoothing_fwhm
        ):
            # this is meant to filter the data, then select down to the columsn we need, then materialize them, all without copying our pulse records again
            uncalibrated = (
                self.df.lazy()
                .filter(self.good_expr)
                .select(pl.col(uncalibrated_col))
                .collect()[uncalibrated_col]
                .to_numpy()
            )
            peak_ph_vals, _peak_heights = mass.algorithms.find_local_maxima(
                uncalibrated, gaussian_fwhm=ph_smoothing_fwhm
            )
            name_e, energies_out, opt_assignments = mass.algorithms.find_opt_assignment(
                peak_ph_vals,
                line_names=line_names,
                maxacc=0.1,
            )
            gain = opt_assignments / energies_out
            pfit_gain = np.polynomial.Polynomial.fit(opt_assignments, gain, deg=2)

            def rough_cal_f(uncalibrated):
                calibrated = uncalibrated / pfit_gain(uncalibrated)
                return calibrated

            predicted_energies = rough_cal_f(np.array(opt_assignments))
            energy_residuals = predicted_energies - energies_out
            # if any(np.abs(energy_residuals) > max_residual_ev):
            #     raise Exception(f"too large residuals: {energy_residuals=} eV")
            step = RoughCalibrationStep(
                [uncalibrated_col],
                calibrated_col,
                self.good_expr,
                rough_cal_f,
                name_e,
                energies_out,
                energy_residuals,
            )
            return self.with_step(step)

        def with_step(self, step):
            df2 = step.calc_from_df(self.df)
            ch2 = Channel(
                df=df2,
                header=self.header,
                noise=self.noise,
                good_expr=self.good_expr,
                df_history=self.df_history + [self.df],
                cal=self.cal.with_step(step),
            )
            return ch2

        def with_good_expr_pretrig_mean_and_postpeak_deriv(self):
            max_postpeak_deriv = outlier_resistant_nsigma_above_mid(
                self.df["postpeak_deriv"].to_numpy(), nsigma=20
            )
            max_pretrig_rms = outlier_resistant_nsigma_above_mid(
                self.df["pretrig_rms"].to_numpy(), nsigma=20
            )
            good_expr = (pl.col("postpeak_deriv") < max_postpeak_deriv).and_(
                pl.col("pretrig_rms") < max_pretrig_rms
            )
            return Channel(
                df=self.df,
                header=self.header,
                noise=self.noise,
                good_expr=good_expr,
                df_history=self.df_history,
                cal=self.cal,
            )

        def summarize_pulses(self):
            step = SummarizeStep(
                inputs=["pulse"],
                output="many",
                good_expr=self.good_expr,
                f=None,
                frametime_s=self.header.frametime_s,
                peak_index=peak_ind,
                pulse_col="pulse",
                pretrigger_ignore=0,
                n_presamples=self.header.n_presamples,
            )
            return self.with_step(step)

        def filter5lag(self, pulse_col="pulse", peak_y_col="5lagy", peak_x_col="5lagx", f_3db=25e3, use_expr=True):
            avg_pulse = (
                self.df.lazy()
                .filter(self.good_expr)
                .filter(use_expr)
                .select(pulse_col)
                .limit(2000)
                .collect()[pulse_col]
                .to_numpy()
                .mean(axis=0)
            )
            spectrum5lag = self.noise.spectrum(trunc_front=2, trunc_back=2)
            filter5lag = massp.fourier_filter(
                avg_signal=avg_pulse[2:-2],
                noise_psd=spectrum5lag.psd,
                dt=self.header.frametime_s,
                f_3db=f_3db,
            )
            step = Filter5LagStep(
                inputs=["pulse"],
                output=[peak_x_col, peak_y_col],
                good_expr=self.good_expr,
                f=None,
                filter=filter5lag,
                spectrum=spectrum5lag,
                use_expr=use_expr
            )
            return self.with_step(step)

        def __hash__(self):
            # needed to make functools.cache work
            # if self or self.anything is mutated, assumptions will be broken
            # and we may get nonsense results
            return hash(id(self))

        def __eq__(self, other):
            # needed to make functools.cache work
            # if self or self.anything is mutated, assumptions will be broken
            # and we may get nonsense results
            # only checks if the ids match, does not try to be equal if all contents are equal
            return id(self) == id(other)
    return Channel,


@app.cell
def __(NoiseChannel, df_noise, header_df, header_df_noise, mo, plt):
    noise_ch = NoiseChannel(df_noise, header_df_noise, header_df["Timebase"][0])
    _spectrum = noise_ch.spectrum()
    noise_ch.spectrum().plot()
    mo.mpl.interactive(plt.gcf())
    return noise_ch,


@app.cell
def __(Channel, ChannelHeader, df, header_df, noise_ch):
    channel = Channel(df, ChannelHeader.from_ljh_header_df(header_df), noise_ch)
    channel2 = (
        channel.summarize_pulses()
        .with_good_expr_pretrig_mean_and_postpeak_deriv()
        .rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="peak_value",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        )
    )
    channel3 = (channel2.filter5lag(f_3db=10e3).rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="5lagy",
            calibrated_col="energy_5lagy",
            ph_smoothing_fwhm=50,
        )
               )
    return channel, channel2, channel3


@app.cell
def __(channel2, mo, plt):
    # channel2.plot_hist("energy_pulse_rms", np.arange(0,10000,10))
    channel2.dbg_plot(step_ind=1)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(channel3, mo, plt):
    channel3.dbg_plot(step_ind=3)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(channel3):
    channel3.df.columns
    return


@app.cell
def __(
    Callable,
    Filter,
    dataclass,
    energies_out,
    filter5lag,
    filter_data_5lag,
    massp,
    np,
    pl,
    plot_hist,
    spectrum,
):
    @dataclass(frozen=True)
    class CalStep:
        inputs: list[str]
        output: str
        good_expr: pl.Expr
        f: Callable[..., float]

        def __call__(self, *args, **kwargs) -> float:
            return self.f(*args, **kwargs)

        def call_by_dict(self, arg_dict) -> float:
            return self(*arg_dict.values())

        def calc_from_df(self, df):
            # uses polars map_elements, which is slow as shit apparently
            # the docs warn that it is slow, and the "correct" way is to implement a
            # user define function in rust https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.map_elements.html
            df2 = df.select(
                pl.struct(self.inputs)
                .map_elements(self.call_by_dict, return_dtype=pl.Float32)
                .alias(self.output)
            ).with_columns(df)
            return df2

        def calc_from_df_np(self, df):
            # only works with in memory data, but just takes it as numpy data and calls function
            # is much faster than map_elements approach, but wouldn't work with out of core data without some extra book keeping
            inputs_np = [df[input].to_numpy() for input in self.inputs]
            out = self.f(*inputs_np)
            df_out = pl.DataFrame({self.output: out})
            return df_out


    @dataclass(frozen=True)
    class RoughCalibrationStep(CalStep):
        line_names: list[str]
        energies: np.ndarray
        energy_residuals: np.ndarray

        def dbg_plot(self, df, bin_edges=np.arange(0, 10000, 1), axis=None, plotkwarg={}):
            series = (
                df.lazy()
                .filter(self.good_expr)
                .select(pl.col(self.output))
                .collect()[self.output]
            )
            axis = plot_hist(series, bin_edges)
            axis.plot(self.energies, np.zeros(len(energies_out)), "o")
            for line_name, energy in zip(self.line_names, self.energies):
                axis.annotate(line_name, (energy, 0), rotation=90)
            np.set_printoptions(precision=2)
            axis.set_title(f"RoughCalibrationStep dbg_plot\n{self.energy_residuals=}")
            return axis


    @dataclass(frozen=True)
    class SummarizeStep(CalStep):
        frametime_s: float
        peak_index: int
        pulse_col: str
        pretrigger_ignore: int
        n_presamples: int

        def calc_from_df(self, df):
            df2 = pl.concat(
                pl.from_numpy(
                    massp.pulse_algorithms.summarize_data_numba(
                        df_iter[self.pulse_col].to_numpy(),
                        self.frametime_s,
                        peak_samplenumber=self.peak_index,
                        pretrigger_ignore=self.pretrigger_ignore,
                        nPresamples=self.n_presamples,
                    )
                )
                for df_iter in df.iter_slices()
            ).with_columns(df)
            return df2


    @dataclass(frozen=True)
    class Filter5LagStep(CalStep):
        filter: Filter
        spectrum: spectrum
        use_expr: pl.Expr

        def calc_from_df(self, df):
            dfs = []
            for df_iter in df.iter_slices(10000):
                peak_x, peak_y = filter_data_5lag(
                    filter5lag.filter, df_iter[self.inputs[0]].to_numpy()
                )
                dfs.append(pl.DataFrame({"peak_x": peak_x, "peak_y": peak_y}))
            df2 = pl.concat(dfs).with_columns(df)
            df2 = df2.rename({"peak_x":self.output[0], "peak_y":self.output[1]})
            return df2

        def dbg_plot(self, df):
            return self.filter.plot()


    @dataclass(frozen=True)
    class CalSteps:
        # leaves many optimizations on the table, but is very simple
        # 1. we could calculate filt_value_5lag and filt_phase_5lag at the same time
        # 2. we could calculate intermediate quantities optionally and not materialize all of them
        steps: list[CalStep]

        def calc_from_df(self, df):
            "return a dataframe with all the newly calculated info"
            for step in self.steps:
                df = step.calc_from_df(df).with_columns(df)
            return df

        def calc_from_df_np(self, df):
            "return a dataframe with all the newly calculated info"
            for step in self.steps:
                df = step.calc_from_df_np(df).with_columns(df)
            return df

        @classmethod
        def new_empty(cls):
            return cls([])

        def __getitem__(self, key):
            return self.steps[key]

        # def copy(self):
        #     # copy by creating a new list containing all the entires in the old list
        #     # a list entry, aka a CalStep, should be immutable
        #     return CalSteps(self.steps[:])

        def with_step(self, step: CalStep):
            # return a new CalSteps with the step added, no mutation!
            return CalSteps(self.steps + [step])
    return (
        CalStep,
        CalSteps,
        Filter5LagStep,
        RoughCalibrationStep,
        SummarizeStep,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
