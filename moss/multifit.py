from dataclasses import dataclass, field
import lmfit
import copy
import mass
import math
import numpy as np
import polars as pl
from typing import Optional
import moss
import pylab as plt


def handle_none(val, default):
    if val is None:
        return copy.copy(default)
    return val



@dataclass(frozen=True)
class FitSpec:
    model: mass.GenericLineModel
    bin_edges: np.ndarray
    use_expr: pl.Expr
    params_update: lmfit.parameter.Parameters

    def params(self, bin_centers, counts):
        params = self.model.make_params()
        params = self.model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        params = params.update(self.params_update)
        return params

    def fit_series_without_use_expr(self, series):
        bin_centers, counts = moss.misc.hist_of_series(series, self.bin_edges)
        params = self.params(bin_centers, counts)
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
    
    def fit_df(self, df: pl.DataFrame, col: str, good_expr: pl.Expr):       
        series = moss.good_series(df, col, good_expr, use_expr=self.use_expr)
        return self.fit_series_without_use_expr(series)
    
    def fit_ch(self, ch: moss.Channel, col: str):
        return self.fit_df(ch.df, col, ch.good_expr)

@dataclass(frozen=True)
class MultiFit:
    default_fit_width: float = 50
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
        dlo = handle_none(dlo, self.default_fit_width/2)
        dhi = handle_none(dhi, self.default_fit_width/2)
        bin_size = handle_none(bin_size, self.default_bin_size)
        params_update = handle_none(params_update, self.default_params_update)
        use_expr = handle_none(use_expr, self.default_use_expr)
        bin_edges = np.arange(-dlo, dhi + bin_size, bin_size) + peak_energy
        fitspec = FitSpec(model, bin_edges, use_expr, params_update)
        return self.with_fitspec(fitspec)

    def with_fitspec(self, fitspec):
        return MultiFit(
            self.default_fit_width,
            self.default_bin_size,
            self.default_use_expr,
            self.default_params_update,
            sorted(self.fitspecs + [fitspec], key = lambda x: x.model.spect.peak_energy),
            # make sure they're always sorted by energy
            self.results,
        )

    def with_results(self, results):
        return MultiFit(
            self.default_fit_width,
            self.default_bin_size,
            self.default_use_expr,
            self.default_params_update,
            self.fitspecs,
            results,
        )

    def results_params_as_df(self):
        result = self.results[0]
        param_names = result.params.keys()
        d = {}
        d["line"] = [fitspec.model.spect.shortname for fitspec in self.fitspecs]
        d["peak_energy_ref"] = [fitspec.model.spect.peak_energy for fitspec in self.fitspecs]
        for param_name in param_names:
            d[param_name]=[result.params[param_name].value for result in self.results]
            d[param_name+"_strerr"]=[result.params[param_name].stderr for result in self.results]
        return pl.DataFrame(d)

    def fit_series_without_use_expr(self, series: pl.Series):
        results = [fitspec.fit_series_without_use_expr(series) for fitspec in self.fitspecs]
        return self.with_results(results)

    def fit_df(self, df: pl.DataFrame, col: str, good_expr: pl.Expr):
        results = []
        for fitspec in self.fitspecs:
            result = fitspec.fit_df(df, col, good_expr)
            results.append(result)
        return self.with_results(results)   
    
    def fit_ch(self, ch: moss.Channel, col: str):
        return self.fit_df(ch.df, col, ch.good_expr)

    def plot_results(self):
        assert self.results is not None
        n = len(self.results)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4)
        )  # Adjust figure size as needed

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
        for ax in axes[len(self.results) :]:
            ax.axis("off")

        plt.tight_layout()
        return fig, axes