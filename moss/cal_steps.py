import os
from dataclasses import dataclass, field
import polars as pl
import pylab as plt
import functools
import typing
import numpy as np
import moss


@dataclass(frozen=True)
class CalStep:
    inputs: list[str]
    output: str
    good_expr: pl.Expr
    f: typing.Callable[..., float]

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
class DriftCorrectStep(CalStep):
    dc: typing.Any
    use_expr: pl.Expr

    def calc_from_df(self, df):
        indicator, uncorrected = self.inputs
        slope, offset = self.dc.slope, self.dc.offset
        df2 = df.select(
            (pl.col(uncorrected) * (1 + slope * (pl.col(indicator) - offset))).alias(self.output)
        ).with_columns(df)
        return df2

    def dbg_plot(self, df):
        indicator, uncorrected = self.inputs
        # breakpoint()
        df_small = (
            df.lazy()
            .filter(self.good_expr)
            .filter(self.use_expr)
            .select(self.inputs + [self.output])
            .collect()
        )
        moss.misc.plot_a_vs_b_series(df_small[indicator], df_small[uncorrected])
        moss.misc.plot_a_vs_b_series(
            df_small[indicator],
            df_small[self.output],
            plt.gca(),
        )
        plt.legend()
        plt.tight_layout()
        return plt.gca()


@dataclass(frozen=True)
class RoughCalibrationStep(CalStep):
    line_names: list[str]
    line_energies: np.ndarray
    predicted_energies: np.ndarray

    def dbg_plot(self, df, bin_edges=np.arange(0, 10000, 1), axis=None, plotkwarg={}):
        series = (
            df.lazy()
            .filter(self.good_expr)
            .select(pl.col(self.output))
            .collect()[self.output]
        )
        axis = moss.misc.plot_hist_of_series(series, bin_edges)
        axis.plot(self.line_energies, np.zeros(len(self.line_energies)), "o")
        for line_name, energy in zip(self.line_names, self.line_energies):
            axis.annotate(line_name, (energy, 0), rotation=90)
        np.set_printoptions(precision=2)
        energy_residuals = self.predicted_energies-self.line_energies
        axis.set_title(f"RoughCalibrationStep dbg_plot\n{energy_residuals=}")
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
                moss.pulse_algorithms.summarize_data_numba(
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
    filter: moss.Filter
    spectrum: moss.NoisePSD
    use_expr: pl.Expr

    def calc_from_df(self, df):
        dfs = []
        for df_iter in df.iter_slices(10000):
            peak_x, peak_y = moss.filters.filter_data_5lag(
                self.filter.filter, df_iter[self.inputs[0]].to_numpy()
            )
            dfs.append(pl.DataFrame({"peak_x": peak_x, "peak_y": peak_y}))
        df2 = pl.concat(dfs).with_columns(df)
        df2 = df2.rename({"peak_x": self.output[0], "peak_y": self.output[1]})
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