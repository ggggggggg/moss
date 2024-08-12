from dataclasses import dataclass, field
import polars as pl
import pylab as plt #type:ignore
import functools
import moss
import numpy as np
from moss.noise_algorithms import NoisePSD
from numpy import float64
from polars.dataframe.frame import DataFrame
from typing import Tuple

@dataclass(frozen=True)
class NoiseChannel:
    df: pl.DataFrame | pl.LazyFrame  # DO NOT MUTATE THIS!!!
    header_df: pl.DataFrame | pl.LazyFrame  # DO NOT MUTATE THIS!!
    frametime_s: float

    @functools.cache
    def calc_max_excursion(
        self, trace_col_name: str="pulse", n_limit: int=10000, excursion_nsigma: int=5
    ) -> Tuple[DataFrame, float64]:
        def excursion2d(noise_trace):
            return np.amax(noise_trace, axis=1) - np.amin(noise_trace, axis=1)
        noise_traces = self.df.limit(n_limit)[trace_col_name].to_numpy()
        excursion = excursion2d(noise_traces)
        max_excursion = moss.misc.outlier_resistant_nsigma_above_mid(
            excursion, nsigma=excursion_nsigma
        )
        df_noise2 = self.df.with_columns(excursion=excursion)
        return df_noise2, max_excursion

    @functools.cache
    def spectrum(
        self,
        trace_col_name: str="pulse",
        n_limit: int=10000,
        excursion_nsigma: int=5,
        trunc_front: int=0,
        trunc_back: int=0,
    ) -> NoisePSD:
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
        spectrum = moss.noise_psd(noise_traces_clean2, dt=self.frametime_s)
        return spectrum

    def __hash__(self) -> int:
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    @classmethod
    def from_ljh(cls, path: str) -> "NoiseChannel":
        ljh = moss.LJHFile(path)
        df, header_df = ljh.to_polars()
        noise_channel = cls(df, header_df, header_df["Timebase"][0])
        return noise_channel
