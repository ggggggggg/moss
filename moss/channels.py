from dataclasses import dataclass, field
import polars as pl
import pylab as plt
import numpy as np
import functools
import mass
import moss
import joblib
import traceback
import lmfit
import pathlib


@dataclass(frozen=True)
class Channels:
    channels: dict[int, moss.Channel]
    description: str
    bad_channels: dict[int, moss.BadChannel] = field(default_factory=dict)

    @property
    def ch0(self):
        assert len(self.channels) > 0, "channels must be non-empty"
        for v in self.channels.values():
            return v

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
        params_update=lmfit.Parameters()
    ):
        model = mass.get_model(line,
                               has_linear_background=has_linear_background,
                               has_tails=has_tails)
        pe = model.spect.peak_energy
        _bin_edges = np.arange(pe - dlo, pe + dhi, binsize)
        df_small = (
            self.dfg().lazy().filter(use_expr).select(col).collect()
        )
        bin_centers, counts = moss.misc.hist_of_series(df_small[col], _bin_edges)
        params = model.guess(counts, bin_centers=bin_centers, dph_de=1)
        params["dph_de"].set(1.0, vary=False)
        print(f"before update {params=}")
        params = params.update(params_update)
        print(f"after update {params=}")
        result = model.fit(
            counts, params, bin_centers=bin_centers, minimum_bins_per_fwhm=3
        )
        result.set_label_hints(
            binsize=bin_centers[1] - bin_centers[0],
            ds_shortname=f"{len(self.channels)} channels, {self.description}",
            unit_str="eV",
            attr_str=col,
            states_hint=f"{use_expr=}",
            cut_hint="",
        )
        return result

    def plot_hist(self, col, bin_edges, use_expr=True, axis=None):
        df_small = self.dfg().lazy().filter(use_expr).select(col).collect()
        ax = moss.misc.plot_hist_of_series(df_small[col], bin_edges, axis)
        ax.set_title(f"{len(self.channels)} channels, {self.description}")

    def plot_hists(self, col, bin_edges, group_by_col, axis=None,
                   use_expr=None, skip_none=True):
        """
        Plots histograms for the given column, grouped by the specified column.

        Parameters:
        - col (str): The column name to plot.
        - bin_edges (array-like): The edges of the bins for the histogram.
        - group_by_col (str): The column name to group by. This is required.
        - axis (matplotlib.Axes, optional): The axis to plot on. If None, a new figure is created.
        """
        if axis is None:
            _, ax = plt.subplots()  # Create a new figure if no axis is provided
        else:
            ax = axis

        if use_expr is None:
            df_small = (self.dfg().lazy().
                        select(col, group_by_col)
                        ).collect().sort(group_by_col, descending=False)
        else:
            df_small = (self.dfg().lazy().filter(use_expr).
                        select(col, group_by_col)
                        ).collect().sort(group_by_col, descending=False)

        # Plot a histogram for each group
        for (group_name,), group_data in df_small.group_by(group_by_col, maintain_order=True):
            if group_name is None and skip_none:
                continue
            # Get the data for the column to plot
            values = group_data[col]
            # Plot the histogram for the current group
            if group_name == "EBIT":
                ax.hist(values, bins=bin_edges, alpha=0.9, color="k", label=str(group_name))
            else:
                ax.hist(values, bins=bin_edges, alpha=0.5, label=str(group_name))
            # bin_centers, counts = moss.misc.hist_of_series(values, bin_edges)
            # plt.plot(bin_centers, counts, label=group_name)

        # Customize the plot
        ax.set_xlabel(str(col))
        ax.set_ylabel('Frequency')
        ax.set_title(f"Coadded Histogram of {col} grouped by {group_by_col}")

        # Add a legend to label the groups
        ax.legend(title=group_by_col)

        plt.tight_layout()

    def map(self, f, allow_throw=False):
        new_channels = {}
        new_bad_channels = {}
        for key, channel in self.channels.items():
            try:
                new_channels[key] = f(channel)
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                error_type: type = type(ex)
                error_message: str = str(ex)
                backtrace: str = traceback.format_exc()
                if allow_throw:
                    raise
                print(f"{key=} {channel=} failed this step")
                print(f"{error_type=}")
                print(f"{error_message=}")
                new_bad_channels[key] = channel.as_bad(error_type, error_message, backtrace)
        new_bad_channels = moss.misc.merge_dicts_ordered_by_keys(self.bad_channels, new_bad_channels)

        return Channels(new_channels, self.description, bad_channels=new_bad_channels)

    def set_bad(self, ch_num, msg, require_ch_num_exists=True):
        new_channels = {}
        new_bad_channels = {}
        if require_ch_num_exists:
            assert ch_num in self.channels.keys(), f"{ch_num} can't be set bad because it does not exist"
        for key, channel in self.channels.items():
            if key == ch_num:
                new_bad_channels[key] = channel.as_bad(None, msg, None)
            else:
                new_channels[key] = channel
        return Channels(new_channels, self.description, bad_channels=new_bad_channels)

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
        channels = {}
        print(f"in from_ljh_path_pairs {len(pulse_noise_pairs)}")
        for pulse_path, noise_path in pulse_noise_pairs:
            channel = moss.Channel.from_ljh(pulse_path, noise_path)
            assert channel.header.ch_num not in channels.keys()
            channels[channel.header.ch_num] = channel
        print(f"in from_ljh_path_pairs {len(channels)=}")
        return cls(channels, description)

    @classmethod
    def from_off_paths(cls, off_paths, description):
        channels = {}
        for path in off_paths:
            ch = moss.Channel.from_off(mass.off.OffFile(path))
            channels[ch.header.ch_num] = ch
        return cls(channels, description)

    @classmethod
    def from_ljh_folder(cls, pulse_folder, noise_folder=None, limit=None, exclude_ch_nums=[]):
        # exclude_ch_nums is a list of channel numbers to exclude
        import os
        assert os.path.isdir(pulse_folder), f"{pulse_folder=} {noise_folder=}"
        if noise_folder is None:
            paths = moss.ljhutil.find_ljh_files(pulse_folder)
            pairs = [(path, None) for path in paths]
        else:
            assert os.path.isdir(noise_folder), f"{pulse_folder=} {noise_folder=}"
            pairs = moss.ljhutil.match_files_by_channel(pulse_folder, noise_folder, limit=limit, exclude_ch_nums=exclude_ch_nums)
        description = f"from_ljh_folder {pulse_folder=} {noise_folder=}"
        print(f"{description}")
        print(f"in from_ljh_folder has {len(pairs)} pairs")
        data = cls.from_ljh_path_pairs(pairs, description)
        print(f"and the Channels obj has {len(data.channels)} pairs")
        return data

    def get_an_ljh_path(self):
        return pathlib.Path(self.ch0.header.df["Filename"][0])

    def get_path_in_output_folder(self, filename):
        ljh_path = self.get_an_ljh_path()
        base_name, post_chan = ljh_path.name.split('_chan')
        date, run_num = base_name.split("_run")        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ljh_path.parent.parent/f"{run_num}moss_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir/filename

    def get_experiment_state_df(self, experiment_state_path=None):
        if experiment_state_path is None:
            ljh_path = self.get_an_ljh_path()
            experiment_state_path = moss.ljhutil.experiment_state_path_from_ljh_path(ljh_path)
        df = pl.read_csv(experiment_state_path, new_columns=["unixnano", "state_label"])
        # _col0, _col1 = df.columns
        df_es = df.select(pl.from_epoch("unixnano", time_unit="ns").dt.cast_time_unit("us").alias("timestamp"))
        # strip whitespace from state_label column
        sl_series = df.select(pl.col("state_label").str.strip_chars()).to_series()
        df_es = df_es.with_columns(state_label=pl.Series(values=sl_series, dtype=pl.Categorical))
        return df_es

    def with_experiment_state_by_path(self, experiment_state_path=None):
        df_es = self.get_experiment_state_df(experiment_state_path)
        return self.with_experiment_state_df(df_es)

    def with_external_trigger_by_path(self, path=None):
        raise Exception("not implemented")

    def with_external_trigger_df(self, df_ext):
        raise Exception("not implemented")

    def with_experiment_state_df(self, df_es):
        # this is not as performant as making use_exprs for states
        # and using .set_sorted on the timestamp column
        ch2s = {}
        for ch_num, ch in self.channels.items():
            ch2s[ch_num] = ch.with_experiment_state_df(df_es)
        return Channels(ch2s, self.description)

    def with_steps_dict(self, steps_dict):
        def load_steps(channel):
            try:
                steps = steps_dict[channel.header.ch_num]
            except KeyError:
                raise Exception("steps dict did not contain steps for this ch_num")
            return channel.with_steps(steps)
        return self.map(load_steps)

    def save_steps(self, filename):
        steps = {}
        for channum, ch in self.channels.items():
            steps[channum] = ch.steps[:]
        moss.misc.pickle_object(steps, filename)
        return steps
    
    def load_steps(self, filename):
        steps = moss.misc.unpickle_object(filename)
        return self.with_steps_dict(steps)

    def parent_folder_path(self):
        import pathlib
        parent_folder_path = pathlib.Path(self.ch0.header.df["Filename"][0]).parent.parent
        print(f"{parent_folder_path=}")
        return parent_folder_path

    def concat_data(self, other_data):
        # sorting here to show intention, but I think set is sorted by insertion order as
        # an implementation detail so this may not do anything
        ch_nums = sorted(list(set(self.channels.keys()).intersection(other_data.channels.keys())))
        new_channels = {}
        for ch_num in ch_nums:
            ch = self.channels[ch_num]
            other_ch = other_data.channels[ch_num]
            combined_df = moss.misc.concat_dfs_with_concat_state(ch.df, other_ch.df)
            new_ch = ch.with_replacement_df(combined_df)
            new_channels[ch_num] = new_ch
        return moss.Channels(new_channels, self.description+other_data.description)

    @classmethod
    def from_df(cls, df, frametime_s=np.nan, n_presamples=None, n_samples=None, description="from Channels.channels_from_df"):
        # requres a column named "ch_num" containing the channel number
        keys_df = df.partition_by(by=["ch_num"], as_dict=True)
        dfs = {keys[0]: df for (keys, df) in keys_df.items()}
        channels = {}
        for ch_num, df in dfs.items():
            channels[ch_num] = moss.Channel(df, header=moss.ChannelHeader(description="from df",
                                                                          ch_num=ch_num,
                                                                          frametime_s=frametime_s,
                                                                          n_presamples=n_presamples,
                                                                          n_samples=n_samples,
                                                                          df=df))
        return Channels(channels, description)
