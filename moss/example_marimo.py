import marimo

__generated_with = "0.7.1"
app = marimo.App(width="medium", app_title="MOSS intro")


@app.cell
def __(mo):
    mo.md(
        """
        #MOSS internals introdution
        MOSS is the Microcalorimeter Online Spectral Software, a replacement for MASS. MOSS support many algorithms for pulse filtering, calibration, and corrections. MOSS is built on modern open source data science software, including pola.rs and marimo. MOSS supports some key features that MASS struggled with including:
          * consecutive data set analysis
          * online (aka realtime) analysis
          * easily supporting different analysis chains
        """
    )
    return


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


@app.cell(disabled=True)
def __(mo):
    file_browser = mo.ui.file_browser(
        initial_path=r"C:\Users\oneilg\Desktop\python\src\mass\tests\ljh_files",
        multiple=True,
        label="pick microcal data directory",
        selection_mode="directory",
    )
    mo.md(
        f"# File Picker\npick microcal data directory\nthe top line should be the folder that contains all the folders you want to search for data\n{file_browser}"
    )
    return file_browser,


@app.cell
def __(mo, moss, pl):
    _root_path = r"C:\Users\oneilg\Desktop\python\src\mass\tests\ljh_files"
    _extensions = ["ljh"]
    _folders = moss.ljhutil.find_folders_with_extension(_root_path, _extensions)
    _df = pl.DataFrame({"folder": _folders})
    pulse_table = mo.ui.table(_df, label="puck pulse files")
    noise_table = mo.ui.table(_df, label="pick noise file", selection="single")
    start_button = mo.ui.run_button(label="submit file choices")
    mo.md(
        f"#File Picker\nMarimo suports very easy reactive UI. You should read the docs. The you can can try it out here if you want. You can edit the line defining `_root_path` to point to a folder with LJH files, and the above ui element will update. Then pick one noise file and one pulse file and the rest of the notebook will update. Though the calibration will likely break since the actual lines chosen will be wrong. So maybe come back here later.\npick files for noise \n{noise_table}\n#pick files for pulses \n{pulse_table}\nthen click {start_button}"
    )
    return noise_table, pulse_table, start_button


@app.cell
def __(mo):
    mo.md(
        """
        # file picker overide
        below here we manually define some default ljh files for this notebook to work on, but if you use the file picker it will override the default values.
        """
    )
    return


@app.cell
def __(noise_table, pulse_table):
    if len(noise_table.value) == 1 and len(pulse_table.value) >= 1:
        noise_folder = noise_table.value["folder"][0]
        pulse_folders = pulse_table.value["folder"]
        pulse_folder = pulse_folders[0]
    else:
        noise_folder = "C:\\Users\\oneilg\\Desktop\\python\\src\\mass\\tests\\ljh_files\\20230626\\0000"
        pulse_folder = "C:\\Users\\oneilg\\Desktop\\python\\src\\mass\\tests\\ljh_files\\20230626\\0001"
    return noise_folder, pulse_folder, pulse_folders


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
def __(moss, noise_folder, pulse_folder):
    data = moss.Channels.from_ljh_folder(
        pulse_folder=pulse_folder, noise_folder=noise_folder
    )
    return data,


@app.cell
def __(mo):
    mo.md(
        """
        # basic analysis
        The variables `data` is the conventional name for a `Channels` object. It contains a list of `Channel` objects, conventinally assigned to a variable `ch` when accessed individualy. One `Channel` represents a single pixel, whiles a `Channels` is a collection of pixels, like a whole array.

        The data tends to consist of pulse shapes (arrays of length 100 to 1000 in general) and per pulse quantities, such as the pretrigger mean. These data are stored internally as pola.rs `DataFrame` objects. 

        The next cell shows a basic analysis on multiple channels. The function `data.transform_channels` takes a one argument function, where the one argument is a `Channel` and the function returns a `Channel`, `data.transform_channels` returns a `Channels`. There is no mutation, and we can't re-use variable names in a reactive notebook, so we store the result in a new variable `data2`.
        """
    )
    return


@app.cell
def __(data):
    data2 = data.transform_channels(
        lambda channel: channel.summarize_pulses()
        .with_good_expr_pretrig_mean_and_postpeak_deriv()
        .rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="peak_value",
            calibrated_col="energy_peak_value",
            ph_smoothing_fwhm=50,
        )
        .filter5lag(f_3db=10e3)
        .rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="5lagy",
            calibrated_col="energy_5lagy",
            ph_smoothing_fwhm=50,
        )
        .driftcorrect()
        .rough_cal(
            ["MnKAlpha", "MnKBeta", "CuKAlpha", "CuKBeta", "PdLAlpha", "PdLBeta"],
            uncalibrated_col="5lagy_dc",
            calibrated_col="energy_5lagy_dc",
            ph_smoothing_fwhm=50,
        )
    )
    return data2,


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
def __(data2, pl):
    data2.channels[4102].df.select(pl.exclude("pulse"))
    return


@app.cell
def __(data2, mo):
    mo.md(
        f"""
        To enable online analysis, we have to keep track of all the steps of our calibration, so each channel has a history of its steps that we can replay. Here we interpolate it into the markdown, each entry is a step name followed by the time it took to perform the step. 

        {data2.channels[4102].step_summary()=}
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
def __(data):
    ch = data.channels[4102]
    ch.df
    return ch,


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
def __(ch, steps):
    step = steps[0]
    _ch = ch
    for step in steps:
        _ch = _ch.with_step(step)
    ch2 = _ch
    ch2.df
    return ch2, step


@app.cell
def __(mo):
    mo.md("# make sure the results are the same!")
    return


@app.cell
def __(ch2, data2):
    ch2.df == data2.channels[4102].df
    return


@app.cell
def __(ch2):
    # to help you remember not to mutate, everything is as immutable as python will allow! this cell errors out!
    ch2.df = 4
    return


@app.cell
def __(data, data2, mo, np):
    mo.md(
        f"""
        # don't worry about all the copies
        we are copying dataframes, but we aren't copying the underlying memory, so our memory usage is about the same as it would be if we used a mutating style of coding.

        `{np.shares_memory(data.channels[4102].df["rowcount"].to_numpy(), data2.channels[4102].df["rowcount"].to_numpy())=}`
        `{np.shares_memory(data.channels[4102].df["pulse"].to_numpy(), data2.channels[4102].df["pulse"].to_numpy())=}`
        """
    )
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
        use_expr=(2800 < pl.col("energy_5lagy_dc")) & (pl.col("energy_5lagy_dc") < 2850),
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
def __(data2):
    # here we have a very simple experiment_state_file
    df_es = data2.get_experiment_state_df()
    df_es
    return df_es,


@app.cell
def __(data2, pl):
    # due to the way timestamps were define in ljh files, we actually have a non-monotonic timestamp in channel 4109, so lets fix that, then apply the experiment state
    # we also drop the "pulse" column here because sorting will acually copy the underlying data
    # and we dont want to do that
    data3 = data2.transform_channels(
        lambda ch: ch.with_df2(ch.df.select(pl.exclude("pulse")).sort(by="timestamp"))
    )
    data3 = data3.with_experiment_state_by_path()
    return data3,


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
         * calibration plan for fitting multiple lines
         * check accuracy of psd level and filter vdv
         * start automated tests
         * move drift correct into moss
         * move fitting into moss
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"# fine calibration")
    return


@app.cell
def __(Optional, moss, np, numpy, pl, plt):
    from dataclasses import dataclass, field
    import mass
    import lmfit
    import copy
    import math


    def handle_none(val, default):
        if val is None:
            return copy.copy(default)
        return val


    @dataclass(frozen=True)
    class FitSpec:
        model: mass.GenericLineModel
        bin_edges: numpy.ndarray
        use_expr: pl.Expr
        params_update: lmfit.parameter.Parameters

        def params(self):
            params = self.model.make_params()
            params["dph_de"].set(1.0, vary=False)
            params = params.update(self.params_update)
            return params

        def fit_series(self, series):
            bin_centers, counts = moss.misc.hist_of_series(series, self.bin_edges)
            params = self.params()
            bin_centers, bin_size = moss.misc.midpoints_and_step_size(self.bin_edges)
            result = self.model.fit(counts, params, bin_centers=bin_centers)
            result.set_label_hints(
                binsize=bin_size,
                ds_shortname="??",
                unit_str="eV",
                attr_str=series.name,
                states_hint=f"{self.use_expr}",
                cut_hint=f"",
            )
            return result


    @dataclass(frozen=True)
    class MultiFit:
        default_d: float = 50
        default_bin_size: float = 0.5
        default_use_expr: bool = True
        default_params_update: dict = field(default_factory=lmfit.Parameters)
        fitspecs: list[FitSpec] = field(default_factory=list)
        results: Optional[list] = None

        def with_line(
            self, line, dlo=None, dhi=None, bin_size=None, use_expr=None, params_update=None
        ):
            model = mass.getmodel(line)
            peak_energy = model.spect.peak_energy
            dlo = handle_none(dlo, self.default_d)
            dhi = handle_none(dhi, self.default_d)
            bin_size = handle_none(bin_size, self.default_bin_size)
            params_update = handle_none(params_update, self.default_params_update)
            use_expr = handle_none(use_expr, self.default_use_expr)
            bin_edges = np.arange(-dlo, dhi + bin_size, bin_size) + peak_energy
            fitspec = FitSpec(model, bin_edges, use_expr, params_update)
            return self.with_fitspec(fitspec)

        def with_fitspec(self, fitspec):
            return MultiFit(
                self.default_d,
                self.default_bin_size,
                self.default_use_expr,
                self.default_params_update,
                self.fitspecs + [fitspec],
                self.results,
            )

        def with_results(self, results):
            return MultiFit(
                self.default_d,
                self.default_bin_size,
                self.default_use_expr,
                self.default_params_update,
                self.fitspecs,
                results,
            )

        def fit_series(self, series):
            results = [fitspec.fit_series(series) for fitspec in self.fitspecs]
            return self.with_results(results)

        def plot_results(self):
            n = len(self.results)
            cols = min(3, n)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))  # Adjust figure size as needed
            
            # If there's only one subplot, axes is not a list but a single Axes object.
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.ravel()
            
            for result, ax in zip(self.results, axes):
                result.plotm(ax=ax)
            
            # Hide any remaining empty subplots
            for ax in axes[len(self.results):]:
                ax.axis('off')
            
            plt.tight_layout()
            return fig, axes
    return (
        FitSpec,
        MultiFit,
        copy,
        dataclass,
        field,
        handle_none,
        lmfit,
        mass,
        math,
    )


@app.cell
def __(MultiFit):
    fits = MultiFit(default_d=40, default_bin_size=0.6)
    fits = fits.with_line("MnKAlpha").with_line("CuKAlpha").with_line("PdLAlpha")
    return fits,


@app.cell
def __(data3, fits):
    fits_with_results = fits.fit_series(data3.channels[4102].df["energy_5lagy_dc"])
    return fits_with_results,


@app.cell
def __(fits_with_results, mo, plt):
    fits_with_results.plot_results()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()