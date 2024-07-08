from dataclasses import dataclass, field
import polars as pl
import pylab as plt
import functools
import collections
import massp

@dataclass(frozen=True)
class Channels:
    channels: collections.OrderedDict[int, massp.Channel]
    description: str

    @functools.cache
    def dfg(self, exclude="pulse"):
        # return a dataframe containing good pulses from each channel,
        # exluding "pulse" by default
        # and including columns "key" (to be removed?) and "ch_num"
        # the more common call should be to wrap this in a convenient plotter
        dfs = []
        for ch_num, channel in self.channels.items():
            df = channel.df.select(pl.exclude(exclude)).filter(channel.good_expr)
            # key_series = pl.Series("key", dtype=pl.Int64).extend_constant(key, len(df))
            assert ch_num == channel.header.ch_num
            ch_series = pl.Series("ch_num", dtype=pl.Int64).extend_constant(
                channel.header.ch_num, len(df)
            )
            dfs.append(df.with_columns(ch_series))
        return pl.concat(dfs)

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
        df_small = self.dfg().lazy().filter(use_expr).select(col).collect()
        bin_centers, counts = massp.misc.hist_of_series(df_small[col], _bin_edges)
        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        result = model.fit(
            counts, params, bin_centers=bin_centers, minimum_bins_per_fwhm=3
        )
        result.set_label_hints(
            binsize=bin_centers[1] - bin_centers[0],
            ds_shortname=self.description,
            unit_str="eV",
            attr_str=col,
            states_hint=f"{use_expr=}",
            cut_hint="",
        )
        return result

    def transform_channels(self, f, allow_throw=True):
        new_channels = collections.OrderedDict()
        for key, channel in self.channels.items():
            try:
                new_channels[key] = f(channel)
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                if allow_throw:
                    raise
                print(f"{key=} {channel=} failed this step")
        return Channels(new_channels, self.description)

    def linefit_joblib(self, line, col, prefer="threads", n_jobs=4):
        def work(key):
            channel = self.channels[key]
            return channel.linefit(line, col)

        parallel = joblib.Parallel(
            n_jobs=n_jobs, prefer=prefer
        )  # its not clear if threads are better.... what blocks the gil?
        results = parallel(joblib.delayed(work)(key) for key in self.channels.keys())
        return results

    def __hash__(self):
        # needed to make functools.cache work
        # if self or self.anything is mutated, assumptions will be broken
        # and we may get nonsense results
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    @classmethod
    def from_ljh_path_pairs(cls, pulse_noise_pairs, description):
        _channels = collections.OrderedDict()
        for pulse_path, noise_path in pulse_noise_pairs:
            channel = massp.Channel.from_ljh(pulse_path, noise_path)
            _channels[channel.header.ch_num] = channel
        return cls(_channels, description)

    @classmethod
    def from_ljh_folder(cls, pulse_folder, noise_folder=None, limit=None):
        if noise_folder is None:
            paths = massp.ljhutil.find_ljh_files(pulse_folder)
            pairs = ((path, None) for path in paths)
        else:
            pairs = massp.ljhutil.match_files_by_channel(pulse_folder, noise_folder, limit=limit)
        description = f"from_ljh_folder {pulse_folder=} {noise_folder=}"
        return cls.from_ljh_path_pairs(pairs, description)