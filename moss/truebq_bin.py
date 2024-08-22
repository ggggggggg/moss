import numpy as np
import polars as pl
import pylab as plt
from dataclasses import dataclass
from pathlib import Path
from numba import njit
import numba.typed
import moss

header_dtype = np.dtype(
    [
        ("format", np.uint32),
        ("schema", np.uint32),
        ("sample_rate_hz", np.float64),
        ("data_reduction_factor", np.int16),
        ("voltage_scale", np.float64),
        ("aquisition_flags", np.uint16),
        ("start_time", np.uint64, 2),
        ("stop_time", np.uint64, 2), # often wrong, written at end of run
        ("number_of_samples", np.uint64), # often wrong, written at end of run
    ]
)

@dataclass(frozen=True)
class TriggerResult:
    data_source: "TrueBqBin"
    filter_in: np.ndarray
    threshold: float
    trig_inds: np.ndarray
    limit_samples: int

    def plot(self, decimate=10, n_limit=100000, offset=0, x_axis_time_s=False, ax=None):
        if ax is None:
            plt.figure()
            ax=plt.gca()
        plt.sca(ax)
        Noffset = offset*decimate
        N = Noffset+n_limit*decimate
        data = self.data_source.data
        if x_axis_time_s:
            x_axis_scale = self.data_source.frametime_s*decimate
        else:
            x_axis_scale = 1
        filter_out = fast_apply_filter(data[Noffset:N],
                                       self.filter_in)
        plt.plot(np.arange(n_limit)*x_axis_scale, data[Noffset:N:decimate], ".", label="data")
        filter_out_decimated = filter_out[::decimate]
        plt.plot(np.arange(len(filter_out_decimated))*x_axis_scale, filter_out_decimated, label="filter_out")
        plt.axhline(self.threshold, label="threshold")
        df = pl.DataFrame({"trig_inds":self.trig_inds})
        trig_inds_plot = df.filter(pl.col("trig_inds").is_between(Noffset, N)).to_series().to_numpy()
        plt.plot(((trig_inds_plot-Noffset)/decimate)*x_axis_scale, filter_out[trig_inds_plot-Noffset], "o", label="trig_inds")
        plt.title(f"{self.data_source.description}, trigger result debug plot")
        plt.legend()
        if x_axis_time_s:
            plt.xlabel(f"time with arb offset / s")
        else:
            plt.xlabel(f"sample number with arb offset after decimation={decimate}")
        plt.ylabel("signal (arb)")

    def get_noise(self, n_dead_samples_after_pulse_trigger, 
                           n_record_samples, max_noise_triggers=200):
        noise_trigger_inds = get_noise_trigger_inds(self.trig_inds, n_dead_samples_after_pulse_trigger, 
                           n_record_samples, max_noise_triggers)
        pulses = gather_pulse_from_inds(self.data_source.data, npre=0, 
                                        nsamples=n_record_samples, 
                                        inds=noise_trigger_inds)
        assert pulses.shape[1] == n_record_samples
        df = pl.DataFrame({"pulse":pulses})
        noise = moss.NoiseChannel(df, 
                                  header_df=self.data_source.header_df,
                                  frametime_s=self.data_source.frametime_s)
        return noise
    
    def to_channel(self, noise_n_dead_samples_after_pulse_trigger, npre, npost, invert=False):
        noise = self.get_noise(noise_n_dead_samples_after_pulse_trigger, npre+npost, max_noise_triggers=1000)
        pulses = gather_pulse_from_inds(self.data_source.data, npre=npre,
                                        nsamples=npre+npost,
                                        inds = self.trig_inds)
        if invert:
            df = pl.DataFrame({"pulse":pulses*-1, "framecount":self.trig_inds})
        else:
            df = pl.DataFrame({"pulse":pulses, "framecount":self.trig_inds})
        ch_header = moss.ChannelHeader(self.data_source.description,
                                       self.data_source.channel_number,
                                       self.data_source.frametime_s,
                                       npre, 
                                       npre+npost,
                                       self.data_source.header_df)
        ch = moss.Channel(df, ch_header, noise)
        return ch


@dataclass(frozen=True)
class TrueBqBin:
    bin_path: Path
    description: str
    channel_number: int
    header_df: pl.DataFrame
    frametime_s: float
    voltage_scale: float
    data: np.ndarray
    # the bin file is a continuous data aqusition, untriggered

    @classmethod
    def load(cls, bin_path):
        bin_path = Path(bin_path)
        channel_number = int(str(bin_path.parent)[-1])
        desc = str(bin_path.parent.parent.stem)
        header_np = np.memmap(bin_path, dtype=header_dtype, mode="r", offset=0, shape=1)
        sample_rate_hz = header_np["sample_rate_hz"][0]
        header_df = pl.from_numpy(header_np)
        data = np.memmap(bin_path, dtype=np.int16, mode="r", offset=68)
        return cls(
            bin_path,
            desc,
            channel_number,
            header_df,
            1/sample_rate_hz,
            header_np["voltage_scale"][0],
            data
        )
    
    def trigger(self, filter_in, threshold, limit_hours=None):
        if limit_hours is None:
            limit_samples = len(self.data)
        else:
            limit_samples = int(limit_hours*3600/self.frametime_s)
        trig_inds = _fasttrig_filter_trigger_with_cache(self.data, filter_in, threshold, limit_samples, self.bin_path, verbose=True)
        return TriggerResult(
            self,
            filter_in,
            threshold,
            trig_inds, 
            limit_samples
        )
    
    
@njit
def fasttrig_filter_trigger(data, filter_in, threshold):
    assert threshold>0, "algorithm assumes we trigger with positiv threshold, change sign of filter_in to accomodate"
    filter_len = len(filter_in)
    inds = []
    jmax = len(data)-filter_len-1
    # njit only likes float64s, so I'm trying to force float64 use without allocating a ton of memory
    cache = np.zeros(len(filter_in))
    filter = np.zeros(len(filter_in))
    filter[:]=filter_in
    # intitalize a,b,c
    j=0
    cache[:] = data[j:(j+filter_len)]
    b = np.dot(cache, filter)
    a=b # won't be used, just need same type
    j=1
    cache[:] = data[j:(j+filter_len)]
    c = np.dot(cache, filter)
    j=2
    ready = False
    prog_step = jmax//100
    prog_ticks = 0
    while j <= jmax:
        if j%prog_step==0:
            prog_ticks+=1
            print(f"fasttrig_filter_trigger {prog_ticks}/{100}")
        a,b = b,c
        cache[:] = data[j:(j+filter_len)]
        c = np.dot(cache, filter)
        if b>threshold and b>=c and b>a and ready:
            inds.append(j)
            ready = False
        if b<0: # hold off on retriggering until we see opposite sign slope
            ready=True
        j+=1
    return np.array(inds)

def gather_pulse_from_inds(data, npre, nsamples, inds):
    pulses = np.zeros((len(inds), nsamples))
    for i, ind in enumerate(inds):
        pulses[i,:] = data[ind-npre:ind+nsamples-npre]
    return pulses

def filter_and_residual_rms(data, chosen_filter, avg_pulse, trig_inds, npre, nsamples, polarity):
    filt_value = np.zeros(len(trig_inds))
    residual_rms = np.zeros(len(trig_inds))
    filt_value_template = np.zeros(len(trig_inds))
    template = avg_pulse-np.mean(avg_pulse)
    template = template/np.sqrt(np.dot(template, template))
    for i in range(len(trig_inds)):
        j = trig_inds[i]
        pulse = data[j-npre:j+nsamples-npre]*polarity
        pulse = pulse - pulse.mean()
        filt_value[i] = np.dot(chosen_filter, pulse)
        filt_value_template[i] = np.dot(template, pulse)
        residual = pulse-template*filt_value_template[i]
        residual_rms_val = moss.misc.root_mean_squared(residual)
        residual_rms[i] = residual_rms_val  
    return filt_value, residual_rms, filt_value_template

@njit 
def fast_apply_filter(data, filter_in):
    cache = np.zeros(len(filter_in))
    filter = np.zeros(len(filter_in))
    filter[:] = filter_in
    filter_len = len(filter)
    filter_out = np.zeros(len(data)-len(filter))
    j = 0
    jmax = len(data)-filter_len-1
    while j <= jmax:
        cache[:] = data[j:(j+filter_len)]
        filter_out[j] = np.dot(cache, filter)
        j+=1
    return filter_out

def get_noise_trigger_inds(pulse_trigger_inds, n_dead_samples_after_previous_pulse, 
                           n_record_samples, max_noise_triggers):
    diffs = np.diff(pulse_trigger_inds)
    inds = []
    for i in range(len(diffs)):
        if diffs[i] > n_dead_samples_after_previous_pulse:
            n_make = (diffs[i]-n_dead_samples_after_previous_pulse)//n_record_samples
            ind0 = pulse_trigger_inds[i]+n_dead_samples_after_previous_pulse
            for j in range(n_make):
                inds.append(ind0+n_record_samples*j)
                if len(inds) == max_noise_triggers:
                    return np.array(inds)
    return np.array(inds)

def _fasttrig_filter_trigger_with_cache(data, filter_in, threshold, limit_samples, bin_path, verbose=True):
    import hashlib
    bin_full_path = Path(bin_path).absolute()
    file_size = bin_full_path.stat().st_size
    to_hash_str = str(filter_in)+str(threshold)+str(limit_samples)+str(bin_full_path)+str(file_size)
    key = hashlib.sha256(to_hash_str.encode()).hexdigest()
    fname = f".{key}.truebq_trigger_cache.npy"
    cache_dir_path = bin_full_path.parent/"_truebq_bin_cache"
    cache_dir_path.mkdir(exist_ok=True)
    file_path = cache_dir_path/fname
    try:
        trig_inds = np.load(file_path)
        if verbose: 
            print(f"cache hit for {file_path}")
    except FileNotFoundError:
        if verbose:
            print(f"cache miss for {file_path}")
        data_trunc = data[:limit_samples]
        trig_inds = fasttrig_filter_trigger(data_trunc, filter_in, threshold)
        np.save(file_path, trig_inds)        
    return trig_inds