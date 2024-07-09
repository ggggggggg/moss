import marimo

__generated_with = "0.6.26"
app = marimo.App(width="medium", app_title="Massp2")


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
    import massp
    return massp,


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
def __(massp, mo, pl):
    _root_path = r"C:\Users\oneilg\Desktop\python\src\mass\tests\ljh_files"
    _extensions = ["ljh"]
    _folders = massp.ljhutil.find_folders_with_extension(_root_path, _extensions)
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
    if len(noise_table.value) == 1 and len(pulse_table.value)>=1:
        noise_folder = noise_table.value["folder"][0]
        pulse_folders = pulse_table.value["folder"]
        pulse_folder = pulse_folders[0]
    else:
        noise_folder = (
            "C:\\Users\\oneilg\\Desktop\\python\\src\\mass\\tests\\ljh_files\\20230626\\0000"
        )
        pulse_folder = (
            "C:\\Users\\oneilg\\Desktop\\python\\src\\mass\\tests\\ljh_files\\20230626\\0001"
        )
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
def __(massp, noise_folder, pulse_folder):
    data = massp.Channels.from_ljh_folder(pulse_folder=pulse_folder, noise_folder=noise_folder)
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


@app.cell
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
    dropdown_ch = mo.ui.dropdown(options=_ch_nums, value=_ch_nums[0], label="channel number")
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
    data2.channels[int(_ch_num)].linefit("MnKAlpha", col=_col).plotm()
    plt.title(f"reative plot of {_ch_num=} and {_col=} for you")
    mo.mpl.interactive(plt.gcf())
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
    step=steps[0]
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
def __():
    return


if __name__ == "__main__":
    app.run()