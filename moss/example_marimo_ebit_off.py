import marimo

__generated_with = "0.7.9"
app = marimo.App(width="medium", app_title="MOSS intro")


@app.cell
def __():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    return mo, np, pl, plt


@app.cell
def __():
    import moss
    return moss,


@app.cell
def __():
    import mass 
    import os
    massroot = os.path.split(os.path.split(mass.__file__)[0])[0]
    massroot
    return mass, massroot, os


@app.cell
def __():
    off_folder = r"C:\Users\oneilg\Downloads\ebit_20240722_0006"
    return off_folder,


@app.cell
def __(mo):
    mo.md(
        """
        # Load data
        Here we load the data, then we explore the internals a bit to show how MOSS is built.
        """
    )
    return


@app.cell
def __(mass, moss, off_folder):
    off_paths = moss.ljhutil.find_ljh_files(off_folder,".off")
    off_paths
    off = mass.off.OffFile(off_paths[0])
    return off, off_paths


@app.cell
def __(mass, moss, pl):
    def from_off_paths(cls, off_paths):
        channels = {}
        for path in off_paths:
            ch = from_off(moss.Channel, mass.off.OffFile(path))
            channels[ch.header.ch_num] =ch
        return moss.Channels(channels, "from_off_paths")

    def from_off(cls, off):
        df = pl.from_numpy(off._mmap)
        df = df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp")).with_columns(df).select(pl.exclude("unixnano"))
        header = moss.ChannelHeader(f"{off}",
                       off.header["ChannelNumberMatchingName"],
                      off.framePeriodSeconds,
        off._mmap["recordPreSamples"][0],
        off._mmap["recordSamples"][0],
        pl.DataFrame(off.header))
        ch = cls(df,
                         header)
        return ch
    return from_off, from_off_paths


@app.cell
def __(from_off_paths, moss, off_paths):
    data = from_off_paths(moss.Channels, off_paths)
    data
    return data,


@app.cell
def __(data, mo):
    data2 = data.with_experiment_state_by_path(r"C:\Users\oneilg\Downloads\ebit_20240722_0006\ebit_20240722_0006\20240722_run0006_experiment_state.txt")
    mo.plain(data2.channels[1].df)
    return data2,


@app.cell
def __(data2, mo):
    ch = data2.channels[1]
    mo.plain(ch.df)
    return ch,


@app.cell
def __(ch, mo, pl, plt):
    ch2 = ch.rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "AlKAlpha", "ClKAlpha", "KKAlpha","ScKAlpha","MgKAlpha"],
            uncalibrated_col="filtValue",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        use_expr=pl.col("state_label")=="START"
        )
    ch2.step_plot(0)
    mo.mpl.interactive(plt.gcf())
    return ch2,


@app.cell
def __(ch, moss, np, pl):
    # Import library
    from findpeaks import findpeaks

    bin_centers, counts = moss.misc.hist_of_series(ch.good_series("filtValue", use_expr=pl.col("state_label")=="START"), np.arange(0,50000,50))

    # Initialized
    fp = findpeaks(method='peakdetect', whitelist=["peak"], lookahead=5, verbose=0 )

    # Example data:
    X = fp.import_example('1dpeaks')

    # Peak detection
    results = fp.fit(counts)

    # Plot
    fp.plot()
    return X, bin_centers, counts, findpeaks, fp, results


@app.cell
def __(results):
    results
    return


@app.cell
def __(moss, np):
    def find_local_maxima(pulse_heights, gaussian_fwhm):
        """Smears each pulse by a gaussian of gaussian_fhwm and finds local maxima,
        returns a list of their locations in pulse_height units (sorted by number of
        pulses in peak) AND their peak values as: (peak_locations, peak_intensities)

        Args:
            pulse_heights (np.array(dtype=float)): a list of pulse heights (eg p_filt_value)
            gaussian_fwhm = fwhm of a gaussian that each pulse is smeared with, in same units as pulse heights
        """
        # kernel density estimation (with a gaussian kernel)
        n = 128 * 1024
        gaussian_fwhm = float(gaussian_fwhm)
        # The above ensures that lo & hi are floats, so that (lo-hi)/n is always a float in python2
        sigma = gaussian_fwhm / (np.sqrt(np.log(2) * 2) * 2)
        tbw = 1.0 / sigma / (np.pi * 2)
        lo = np.min(pulse_heights) - 3 * gaussian_fwhm
        hi = np.max(pulse_heights) + 3 * gaussian_fwhm
        hist, bins = np.histogram(pulse_heights, np.linspace(lo, hi, n + 1))
        tx = np.fft.rfftfreq(n, (lo - hi) / n)
        ty = np.exp(-tx**2 / 2 / tbw**2)
        x = (bins[1:] + bins[:-1]) / 2
        y = np.fft.irfft(np.fft.rfft(hist) * ty)

        flag = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
        lm = np.arange(1, n - 1)[flag]
        lm = lm[np.argsort(-y[lm])]
        bin_centers, step_size = moss.misc.midpoints_and_step_size(bins)
        return np.array(x[lm]), np.array(y[lm]), (hist, bin_centers, y)
    return find_local_maxima,


@app.cell
def __(np):
    import itertools

    def find_pfit_gain_residual(ph, e):
        assert len(ph) == len(e)
        gain = ph/e
        pfit_gain = np.polynomial.Polynomial.fit(ph, gain, deg=2)
        def ph2energy(ph):
            return ph/pfit_gain(ph)
        predicted_e = ph2energy(ph)
        residual_e = e-predicted_e
        return residual_e, pfit_gain

    def find_best_residual_among_all_possible_assignments(ph, e):
        assert len(ph) >= len(e)
        ph=np.sort(ph)
        assignments_inds = itertools.combinations(np.arange(len(ph)), len(e))
        best_rms_residual = np.inf
        best_ph_assigned = None
        best_residual_e = None
        for i, assignment_inds in enumerate(assignments_inds):
            assignment_inds = np.array(assignment_inds)
            ph_assigned = np.array(ph[assignment_inds])
            residual_e, pfit_gain = find_pfit_gain_residual(ph_assigned,e)
            rms_residual = np.std(residual_e)
            if rms_residual < best_rms_residual:
                best_rms_residual=rms_residual
                best_ph_assigned = ph_assigned
                best_residual_e = residual_e
                best_assignment_inds = assignment_inds
        return best_rms_residual, best_ph_assigned, best_residual_e, best_assignment_inds

    return (
        find_best_residual_among_all_possible_assignments,
        find_pfit_gain_residual,
        itertools,
    )


@app.cell
def __(
    ch2,
    find_best_residual_among_all_possible_assignments,
    find_local_maxima,
    mass,
    mo,
    np,
    pl,
    plt,
):
    line_names = ["AlKAlpha", "MgKAlpha", "ClKAlpha", "ScKAlpha", "CoKAlpha","MnKAlpha","VKAlpha","CuKAlpha"]
    uncalibrated_col, calibrated_col = "filtValue", "energy_filtValue"
    use_expr = pl.col("state_label")=="START"
    uncalibrated = ch2.good_series(uncalibrated_col, use_expr=use_expr).to_numpy()
    peak_ph_vals, _peak_heights, (count, bin_centers2, count_smooth) = find_local_maxima(
        uncalibrated, gaussian_fwhm=50
    )
    (names, ee) = mass.algorithms.line_names_and_energies(line_names)
    n = len(ee)+2
    best_rms_residual, best_ph_assigned, best_residual_e, best_assignment_inds=find_best_residual_among_all_possible_assignments(peak_ph_vals[:n], ee)

    plt.plot(bin_centers2, count)
    plt.plot(bin_centers2, count_smooth)
    plt.plot(peak_ph_vals[:n], _peak_heights[:n],"o")
    plt.plot(best_ph_assigned, np.zeros(len(best_ph_assigned)),"s")
    plt.xlabel("ph")
    plt.ylabel("intensity")
    plt.title(f"{best_rms_residual=:.2f}")
    mo.mpl.interactive(plt.gcf())



    # name_e, energies_out, opt_assignments = mass.algorithms.find_opt_assignment(
    #     peak_ph_vals,
    #     line_names=line_names,
    #     maxacc=0.1,
    # )
    # gain = opt_assignments / energies_out
    # gain_pfit = np.polynomial.Polynomial.fit(opt_assignments, gain, deg=2)

    # def ph2energy(uncalibrated):
    #     calibrated = uncalibrated / gain_pfit(uncalibrated)
    #     return calibrated

    # predicted_energies = ph2energy(np.array(opt_assignments))
    return (
        best_assignment_inds,
        best_ph_assigned,
        best_residual_e,
        best_rms_residual,
        bin_centers2,
        calibrated_col,
        count,
        count_smooth,
        ee,
        line_names,
        n,
        names,
        peak_ph_vals,
        uncalibrated,
        uncalibrated_col,
        use_expr,
    )


@app.cell
def __(best_ph_assigned, ee, plt):
    plt.plot(best_ph_assigned, ee, "o")
    return


@app.cell
def __(mo, np, plt):
    fv = np.sort(np.array([7025, 8725, 22575, 26825, 31325, 33625, 36025, 40825, 43175, 48175]))
    e = np.sort(np.array([1253.635, 1486.69, 4090.699, 4952.216, 5898.802, 6404.01, 6930.38, 8047.83, 8638.91, 9886.52]))
    pfit = np.polynomial.Polynomial.fit(fv, e, deg=2)
    pfit_gain = np.polynomial.Polynomial.fit(fv, fv/e, deg=2)
    fvp = np.hstack([[0], fv])
    plt.plot(fv, e,"o")
    plt.plot(fvp, pfit(fvp))
    plt.plot(fvp, fvp/pfit_gain(fvp))
    plt.xlabel("fv")
    plt.ylabel("e")
    mo.mpl.interactive(plt.gcf())
    return e, fv, fvp, pfit, pfit_gain


@app.cell
def __(e, fv, mo, pfit, pfit_gain, plt):
    resid = e-pfit(fv)
    resid_gain = e-fv/pfit_gain(fv)
    plt.plot(e, resid,"o", label="residual normal")
    plt.plot(e, resid_gain,"o", label="resid_gain")
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return resid, resid_gain


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        f"""
        # inspecting the data

        Internally, the data is stored in polars `DataFrame`s. Lets take a look. To access the dataframe for one channel we do `data2.channels[4102].df`. In `marimo` we can get a nice UI element to browse through the data by returning the `DataFrame` as the last element in a cell. marimo's nicest display doesn't work with array columns like our pulse column, so lets leave that out for now.
        """
    )
    return


@app.cell
def __(data2, mo):
    mo.md(
        f"""
        To enable online analysis, we have to keep track of all the steps of our calibration, so each channel has a history of its steps that we can replay. Here we interpolate it into the markdown, each entry is a step name followed by the time it took to perform the step. 

        {data2.channels[1].step_summary()=}
        """
    )
    return


@app.cell
def __(data2, mo):
    _ch_nums = list(str(_ch_num) for _ch_num in data2.channels.keys())
    dropdown_ch = mo.ui.dropdown(
        options=_ch_nums, value=_ch_nums[0], label="channel number"
    )
    _energy_cols = [col for col in data2.dfg().columns if col.startswith("energy")]
    _energy_cols
    dropdown_col = mo.ui.dropdown(
        options=_energy_cols, value=_energy_cols[0], label="energy col"
    )
    mo.md(
        f"""MOSS has some convenient fitting and plotting methods. We can combine them with marimo's super easy ui element to pick a channel number {dropdown_ch} and an energy column name {dropdown_col}. we can use it to make plots!

        MOSS fitters are based on the best avaialble spectral shap in the literature, and the fwhm resolution value refers to only the detector portion of the resolution."""
    )
    return dropdown_ch, dropdown_col


@app.cell
def __(data2, dropdown_ch, dropdown_col, mo, plt):
    _ch_num, _col = int(dropdown_ch.value), dropdown_col.value
    _ch = data2.channels[int(_ch_num)]
    result = _ch.linefit("MnKAlpha", col=_col)
    result.plotm()
    plt.title(f"reative plot of {_ch_num=} and {_col=} for you")
    mo.mpl.interactive(plt.gcf())
    return result,


@app.cell
def __(result):
    result.fit_report()
    return


@app.cell
def __(mo):
    mo.md(
        """
        # replay
        here we'll apply the same steps to the original dataframe for one channel to show the replay capability

        `ch = data.channels[4102]` is one way to access the `Channel` from before all the analysis steps. Notice how it only has 3 columns, instead of the many you see for `data.channels[4102]`. The display of steps could really be improved!
        """
    )
    return


@app.cell
def __(data2):
    steps = data2.channels[4102].steps
    steps
    return steps,


@app.cell
def __(mo):
    mo.md(
        """
        # apply steps
        marimo has an outline feature look on the left for the scroll looking icon, and click on it. You can navigate around this notebook with it!

        below we apply all the steps that were saved in data2 to our orignial channel, which recall was saved in `data`.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("# make sure the results are the same!")
    return


@app.cell
def __(mo):
    mo.md("# step plots")
    return


@app.cell
def __(ch2, mo, plt):
    ch2.step_plot(4)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(ch2, mo, plt):
    ch2.step_plot(5)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(ch2, mo, plt):
    ch2.step_plot(2)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(mo):
    mo.md(
        """
        # cuts? good_expr and use_expr
        We're using polars expressions in place of cuts. Each `Channel` can work with two of these, `good_expr` which is intended to isolate clean pulses that will yield good resolution, and `use_expr` which is intended to time slice to seperate out different states of the experiment. However, there is nothing binding these behaviors.

        `good_expr` is stored in the `Channel` and use automatically in many functions, including plots. `use_expr` is passed on a per function basis, and is not generally stored, although some steps will store the `use_expr` provided during that step. Many functions have something like `plot(df.filter(good_expr).filter(use_expr))` in them.
        """
    )
    return


@app.cell
def __(ch2):
    ch2.good_expr
    return


@app.cell
def __(ch2, pl):
    # here we use a good_expr to drift correct over a smaller energy range
    ch3 = ch2.driftcorrect(
        indicator="pretrig_mean",
        uncorrected="energy_5lagy_dc",
        use_expr=(pl.col("energy_5lagy_dc").is_between(2800, 2850)),
    )
    return ch3,


@app.cell
def __(ch3, mo, plt):
    # here we make the debug plot for that last step, and see that it has stored the use_expr and used it for its plot
    # we can see that this line has a non-optimal drift correction when learning from the full energy range, but is improved when we learn just from it's energy range
    ch3.step_plot(6)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(ch3):
    ch3.steps[6].use_expr
    return


@app.cell
def __(ch3, mo):
    mo.md(f"{str(ch3.good_expr)=} remains unchanged")
    return


@app.cell
def __(data3):
    # now lets combine the data by calling data.dfg()
    # to get one combined dataframe from all channels
    # and we downselect to just to columns we want for further processing
    dfg = data3.dfg().select("timestamp", "energy_5lagy_dc", "state_label", "ch_num")
    dfg
    return dfg,


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # todos!
         * external trigger
         * check accuracy of psd level and filter vdv
         * start automated tests
         * move drift correct into moss
         * move fitting into moss (maybe seperate package?)
         * open multi ljh example
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # fine calibration
        first we show off the MultiFit class, then we use it to run a fine calibration step
        then we use the same multifit spec to calibrate our data, and make the debug plot
        """
    )
    return


@app.cell
def __(data3, mo, moss, plt):
    multifit = moss.MultiFit(default_fit_width=80, default_bin_size=0.6)
    multifit = multifit.with_line("MnKAlpha").with_line("CuKAlpha").with_line("PdLAlpha")
    multifit_with_results = multifit.fit_ch(data3.channels[4102], "energy_5lagy_dc")
    multifit_with_results.plot_results()
    mo.mpl.interactive(plt.gcf())
    return multifit, multifit_with_results


@app.cell
def __(multifit_with_results):
    pd_result, mn_result, cu_result = multifit_with_results.results
    assert mn_result.params["fwhm"].value < 3.34
    assert cu_result.params["fwhm"].value < 3.7
    return cu_result, mn_result, pd_result


@app.cell
def __(data3, multifit):
    data4= data3.transform_channels(lambda ch: ch.multifit_spline_cal(multifit, previous_cal_step_index=5, calibrated_col="energy2_5lagy_dc"))
    return data4,


@app.cell
def __(data4, mo, plt):
    data4.channels[4102].step_plot(6)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data4):
    steps_dict = {ch_num:ch.steps for ch_num, ch in data4.channels.items()}
    return steps_dict,


@app.cell
def __(moss, steps_dict):
    moss.misc.pickle_object(steps_dict, filename="example_steps_dict.pkl")
    return


@app.cell
def __(data4):
    data4.dfg().select("timestamp", "energy2_5lagy_dc", "state_label", "ch_num").write_parquet("example_result.parquet")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # "multi-ljh analysis"
        we can easily concatenate a `Channel`s or `Channels`s with `Channel.contcat_df`, `Channel.concat_ch`, and `channels.concat_data`. For now the steps and df history are dropped, since it's not quite clear how to use them helpfully. Internally this relies on polars ability to concat `DataFrame`s without allocation.
        """
    )
    return


@app.cell
def __(ch):
    # here we concatenate two channels and check that the length has double
    ch_concat = ch.concat_ch(ch)
    assert 2*len(ch.df) == len(ch_concat.df)
    return ch_concat,


@app.cell
def __(data4):
    # here we concatenate two `Channels` objects and check that the length of the resulting dfg (remember, this is the df of good pulses) has doubled
    data_concat = data4.concat_data(data4)
    assert 2*len(data4.dfg())==len(data_concat.dfg())
    return data_concat,


@app.cell
def __(mo):
    mo.md(r"# final coadded spectrum")
    return


@app.cell
def __(data4, mo, plt):
    _result = data4.linefit("MnKAlpha",col="energy2_5lagy_dc")
    _result.plotm()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(data2):
    ch6=data2.channels[4102]
    return ch6,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
