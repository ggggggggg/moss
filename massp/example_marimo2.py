import marimo

__generated_with = "0.6.26"
app = marimo.App(width="medium", app_title="Massp2")


@app.cell
def __():
    import polars as pl
    import pylab as plt
    import numpy as np
    import marimo as mo
    import time
    import collections
    import joblib
    import os
    from os import path
    import re
    return collections, joblib, mo, np, os, path, pl, plt, re, time


@app.cell
def __():
    from dataclasses import dataclass, field
    import functools
    return dataclass, field, functools


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
        f"pick microcal data directory\nthe top line should be the folder that contains all the folders you want to search for data\n{file_browser}"
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
        f"#pick files for noise \n{noise_table}\n#pick files for pulses \n{pulse_table}\nthen click {start_button}"
    )
    return noise_table, pulse_table, start_button


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
def __(massp, noise_folder, pulse_folder):
    channels = massp.Channels.from_ljh_folder(pulse_folder=pulse_folder, noise_folder=noise_folder)
    return channels,


@app.cell
def __(data):
    data.dfg()
    return


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
def __():
    a = [1,2,3]
    a[:5]
    return a,


@app.cell
def __(channels):
    data = channels.transform_channels(
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
    return data,


@app.cell
def __(data):
    data.channels[4102].step_summary()
    return


@app.cell(hide_code=True)
def __(data, mo):
    _keys = list(str(key) for key in data.channels.keys())
    dropdown_ch = mo.ui.dropdown(options=_keys, value=_keys[0], label="channel number")
    _energy_cols = [col for col in data.dfg().columns if col.startswith("energy")]
    _energy_cols
    dropdown_col = mo.ui.dropdown(
        options=_energy_cols, value=_energy_cols[0], label="energy col"
    )
    mo.md(
        f"yo dude, why dont you choose a channel {dropdown_ch} and an energy col {dropdown_col}. it'd be cool"
    )
    return dropdown_ch, dropdown_col


@app.cell
def __(data, dropdown_ch, dropdown_col, mo, plt):
    _ch, _col = int(dropdown_ch.value), dropdown_col.value
    data.channels[int(_ch)].linefit("MnKAlpha", col=_col).plotm()
    plt.title(f"hey dude, I fit and plotted {_ch=} and {_col=} for you")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
