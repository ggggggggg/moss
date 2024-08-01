import os
from dataclasses import dataclass, field
import polars as pl
import pylab as plt
import functools
import moss
from moss import NoiseChannel, CalSteps, DriftCorrectStep, SummarizeStep, Filter5LagStep
import typing
import numpy as np
from typing import Union
import time
import mass

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
        return cls(
            description=os.path.split(df["Filename"][0])[-1],
            ch_num=df["Channel"][0],
            frametime_s=df["Timebase"][0],
            n_presamples=df["Presamples"][0],
            n_samples=df["Total Samples"][0],
            df=df,
        )

@dataclass(frozen=True)
class Channel:
    df: pl.DataFrame | pl.LazyFrame = field(repr=False)
    header: ChannelHeader = field(repr=True)
    noise: typing.Optional[NoiseChannel] = field(default=None, repr=False)
    good_expr: pl.Expr = True #changed from bool | pl.Expr to just pl.Expr
    df_history: list[pl.DataFrame | pl.LazyFrame] = field(
        default_factory=list, repr=False
    )
    steps: CalSteps = field(default_factory=CalSteps.new_empty)
    steps_elapsed_s: list[float] = field(default_factory=list)

    def step_plot(self, step_ind, **kwargs):
        step = self.steps[step_ind]
        if step_ind + 1 == len(self.df_history):
            df_after = self.df
        else:
            df_after = self.df_history[step_ind + 1]
        return step.dbg_plot(df_after, **kwargs)

    def plot_hist(self, col, bin_edges, axis=None):
        return moss.misc.plot_hist_of_series(self.df[col], bin_edges, axis)
    
    def good_series(self, col, use_expr):
        return moss.good_series(self.df, col, self.good_expr, use_expr)
    
    def rough_gain_cal(
        self, line_names, uncalibrated_col, calibrated_col, ph_smoothing_fwhm,
        use_expr=True
    ):
        # this is meant to filter the data, then select down to the columsn we need, then materialize them, all without copying our pulse records again
        uncalibrated = self.good_series(uncalibrated_col, use_expr=use_expr).to_numpy()
        peak_ph_vals, _peak_heights = mass.algorithms.find_local_maxima(
            uncalibrated, gaussian_fwhm=ph_smoothing_fwhm
        )
        name_e, energies_out, opt_assignments = mass.algorithms.find_opt_assignment(
            peak_ph_vals,
            line_names=line_names,
            maxacc=0.1,
        )
        gain = opt_assignments / energies_out
        gain_pfit = np.polynomial.Polynomial.fit(opt_assignments, gain, deg=2)

        def ph2energy(uncalibrated):
            calibrated = uncalibrated / gain_pfit(uncalibrated)
            return calibrated

        predicted_energies = ph2energy(np.array(opt_assignments))
        # energy_residuals = predicted_energies - energies_out
        # if any(np.abs(energy_residuals) > max_residual_ev):
        #     raise Exception(f"too large residuals: {energy_residuals=} eV")
        step = moss.RoughCalibrationGainStep(
            [uncalibrated_col],
            [calibrated_col],
            self.good_expr,
            use_expr=use_expr,
            line_names=name_e,
            line_energies=energies_out,
            predicted_energies=predicted_energies,
            ph2energy = ph2energy
        )
        return self.with_step(step)

    def rough_cal(
        self, line_names, uncalibrated_col, calibrated_col, 
        ph_smoothing_fwhm, n_extra=3,
        use_expr=True
    ):
        step = moss.RoughCalibrationStep.learn(self, line_names, 
                                             uncalibrated_col=uncalibrated_col,
                                             calibrated_col=calibrated_col,
                                             ph_smoothing_fwhm=ph_smoothing_fwhm,
                                             n_extra=n_extra,
                                             use_expr=use_expr)
        return self.with_step(step)

    def with_step(self, step):
        t_start = time.time()
        df2 = step.calc_from_df(self.df)
        elapsed_s = time.time() - t_start
        ch2 = Channel(
            df=df2,
            header=self.header,
            noise=self.noise,
            good_expr=step.good_expr,
            df_history=self.df_history + [self.df],
            steps=self.steps.with_step(step),
            steps_elapsed_s=self.steps_elapsed_s + [elapsed_s],
        )
        return ch2
    
    def with_steps(self, steps):
        ch2 = self
        for step in steps:
            ch2 = ch2.with_step(step)
        return ch2

    def with_good_expr(self, good_expr):
        return Channel(
            df=self.df,
            header=self.header,
            noise=self.noise,
            good_expr=good_expr,
            df_history=self.df_history,
            steps=self.steps,
            steps_elapsed_s=self.steps_elapsed_s,
        )

    def with_good_expr_pretrig_mean_and_postpeak_deriv(self):
        max_postpeak_deriv = moss.misc.outlier_resistant_nsigma_above_mid(
            self.df["postpeak_deriv"].to_numpy(), nsigma=20
        )
        max_pretrig_rms = moss.misc.outlier_resistant_nsigma_above_mid(
            self.df["pretrig_rms"].to_numpy(), nsigma=20
        )
        good_expr = (pl.col("postpeak_deriv") < max_postpeak_deriv).and_(
            pl.col("pretrig_rms") < max_pretrig_rms
        )
        return self.with_good_expr(good_expr)
    
    def with_good_expr_below_nsigma_outlier_resistant(self, col_nsigma_pairs, and_ = None, and_prev_good_expr=False, use_prev_good_expr=True):
        if use_prev_good_expr:
            df = self.df.lazy().select(pl.exclude("pulse")).filter(self.good_expr).collect()
        else:
            df = self.df
        for i, (col, nsigma) in enumerate(col_nsigma_pairs):
            max_for_col = moss.misc.outlier_resistant_nsigma_above_mid(
                df[col].to_numpy(), nsigma=nsigma
            )
            this_iter_good_expr = pl.col(col).is_between(0, max_for_col)
            if i == 0:
                good_expr = this_iter_good_expr
            else:
                good_expr = good_expr.and_(this_iter_good_expr)
        if and_ is not None:
            good_expr = good_expr.and_(and_)
        if and_prev_good_expr:
            good_expr = self.good_expr.and_(good_expr)
        return self.with_good_expr(good_expr)

    @functools.cache
    def typical_peak_ind(self, col="pulse"):
        return int(np.median(self.df.limit(100)[col].to_numpy().argmax(axis=1)))

    def summarize_pulses(self, col="pulse"):
        step = SummarizeStep(
            inputs=[col],
            output="many",
            good_expr=self.good_expr,
            use_expr=True,
            frametime_s=self.header.frametime_s,
            peak_index=self.typical_peak_ind(col),
            pulse_col=col,
            pretrigger_ignore=0,
            n_presamples=self.header.n_presamples,
        )
        return self.with_step(step)

    def filter5lag(
        self,
        pulse_col="pulse",
        peak_y_col="5lagy",
        peak_x_col="5lagx",
        f_3db=25e3,
        use_expr=True,
    ):
        avg_pulse = (
            self.df.lazy()
            .filter(self.good_expr)
            .filter(use_expr)
            .select(pulse_col)
            .limit(2000)
            .collect()
            .to_series()
            .to_numpy()
            .mean(axis=0)
        )
        spectrum5lag = self.noise.spectrum(trunc_front=2, trunc_back=2)
        filter5lag = moss.fourier_filter(
            avg_signal=avg_pulse[2:-2],
            noise_psd=spectrum5lag.psd,
            dt=self.header.frametime_s,
            f_3db=f_3db,
        )
        step = Filter5LagStep(
            inputs=["pulse"],
            output=[peak_x_col, peak_y_col],
            good_expr=self.good_expr,
            use_expr=use_expr,
            filter=filter5lag,
            spectrum=spectrum5lag,
        )
        return self.with_step(step)

    def good_df(self, cols=pl.all(), use_expr=True):
        return (self.df.lazy()
            .filter(self.good_expr)
            .filter(use_expr)
            .select(cols)
            .collect())

    def good_serieses(self, cols, use_expr):
        df2 = self.good_df(cols, use_expr)
        return [df2[col] for col in cols]

    def driftcorrect(
        self,
        indicator_col="pretrig_mean",
        uncorrected_col="5lagy",
        corrected_col=None,
        use_expr=True,
    ):
        # by defining a seperate learn method that takes ch as an argument,
        # we can move all the code for the step outside of Channel
        step = DriftCorrectStep.learn(ch=self,
                                      indicator_col=indicator_col,
                                      uncorrected_col=uncorrected_col,
                                      corrected_col=corrected_col,
                                      use_expr=use_expr)
        return self.with_step(step)

    def linefit(
        self,
        line,
        col,
        use_expr=True,
        has_linear_background=False,
        has_tails=False,
        dlo=50,
        dhi=50,
        binsize=0.5,
    ):
        model = mass.get_model(line, has_linear_background=False, has_tails=False)
        pe = model.spect.peak_energy
        _bin_edges = np.arange(pe - dlo, pe + dhi, binsize)
        df_small = (
            self.df.lazy().filter(self.good_expr).filter(use_expr).select(col).collect()
        )
        bin_centers, counts = moss.misc.hist_of_series(df_small[col], _bin_edges)
        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        result = model.fit(
            counts, params, bin_centers=bin_centers, minimum_bins_per_fwhm=3
        )
        result.set_label_hints(
            binsize=bin_centers[1] - bin_centers[0],
            ds_shortname=self.header.description,
            unit_str="eV",
            attr_str=col,
            states_hint=f"{use_expr=}",
            cut_hint="",
        )
        return result

    def step_summary(self):
        return [
            (type(a).__name__, b) for (a, b) in zip(self.steps, self.steps_elapsed_s)
        ]

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


    @classmethod
    def from_ljh(cls, path, noise_path=None, keep_posix_usec=False):
        if noise_path is None:
            noise_channel = None
        else:
            noise_channel = moss.NoiseChannel.from_ljh(noise_path)
        ljh = moss.LJHFile(path)
        df, header_df = ljh.to_polars(keep_posix_usec)
        header = moss.ChannelHeader.from_ljh_header_df(header_df)
        channel = moss.Channel(df, header=header, noise=noise_channel)
        return channel
    
    def with_experiment_state_df(self, df_es):
        df2 = self.df.join_asof(df_es, on="timestamp", strategy="backward")
        return self.with_replacement_df(df2)

    def with_replacement_df(self, df2):
        return Channel(
                df=df2,
                header=self.header,
                noise=self.noise,
                good_expr=self.good_expr,
                df_history=self.df_history,
                steps=self.steps,
            ) 
    
    def with_columns(self, df2):
        df3 = df2.with_columns(self.df)
        return self.with_replacement_df(df3)
    

    def multifit_spline_cal(
        self, multifit: moss.MultiFit, previous_cal_step_index, 
        calibrated_col, use_expr=True
    ):
        step = moss.MultiFitSplineStep.learn(self, multifit=multifit,
                                             previous_cal_step_index=previous_cal_step_index,
                                             calibrated_col=calibrated_col,
                                             use_expr=use_expr)
        return self.with_step(step)
    
    def concat_df(self, df):
        ch2 = moss.Channel(pl.concat([self.df, df]),
                        self.header,
                        self.noise,
                        self.good_expr
                        ) 
        # we won't copy over df_history and steps. I don't think you should use this when those are filled in?
        return ch2
    
    def concat_ch(self, ch):
        ch2 = self.concat_df(ch.df)
        return ch2