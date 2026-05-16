from imports import yaml
from imports import datetime
from imports import functional
# from imports import snntorch as snn
from imports import torch
from imports import plt
from imports import Figure


# load the config.yml
def load_config(path: str = "config.yml") -> tuple[dict, dict]:
    with open(path, "r") as file:
        configs = yaml.safe_load(file)

    return configs["data"], configs["model"]

class DataHandler():
    def __init__(
        self,
        recorder: functional.probe.OutputMonitor,
        time_steps: int,
        datapath: str = "data/",
    ) -> None:
        
        self.recorder = recorder
        self.path = datapath
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.time_steps = time_steps


    def measure_tendency(
        self
    ) -> dict:
        
        #
        measurements = {}

        for layer in self.recorder.monitored_layers:
            # create tensor from recording
            layerlist = self.recorder[layer][:self.time_steps]
            spikes = torch.stack([x[0,:] for x, _ in layerlist]).T
            print(spikes.shape)
            
            # calculate the average isi, synchrony
            measurements[layer] = {
                "rates": spikes.sum(1),
                "smoothed_rates": self.measure_rate(spikes),
                "rsync": self.measure_rsync(spikes),
                "isis": self.measure_isis(spikes)
            }
            # for neuron in range(spikes.shape[0]):
            #     returnvalue = self.measure_rate(spikes[neuron])
            #     if returnvalue:
            #         tmp_rate.append(returnvalue["no_spk"])              # type: ignore
            #     else:
            #         tmp_rate.append(torch.tensor([0]))
            #     # self.measure_latency()
            # rate.append(torch.tensor(tmp_rate))
        
        return measurements

    def measure_isis(
        self,
        spikes: torch.Tensor
    ) -> torch.Tensor:
        
        if spikes.isnan().any():
            raise ValueError(
                "Some Value in the Spike-Train is nan.\n" + 
                f"Offending Value(s): {spikes[spikes.isnan().nonzero(as_tuple = True)]}\n" +
                f"At indices: {spikes.isnan().nonzero()}"
            )
        if spikes.isinf().any():
            raise ValueError(
                "Some Value in the Spike-Train is inf.\n" + 
                f"Offending Value(s): {spikes[spikes.isinf().nonzero(as_tuple = True)]}\n" +
                f"At indices: {spikes.isinf().nonzero()}"
            )
        
        spk_neuron, spk_times = spikes.nonzero(as_tuple = True)
        isis = []
        # over spikes.shape[0], because not every neuron neccessarily spikes
        for neuron in range(spikes.shape[0]):
            # only select relevant neurons, still works if mask is empty
            mask = spk_neuron[spk_neuron == neuron]
            # calculates the "forward difference", so N+1 - N
            isis.append(torch.diff(spk_times[mask]))
        
        return torch.nested.nested_tensor(isis)
        # breakpoint()

        # # measure isis, get variance
        # # since spike train is a vector, the tuple only has one entry.
        # spike_loc = spikes.nonzero(as_tuple = True)[0]

        # if len(spike_loc) <= 1:
        #     return False
        
        # spike_loc, _ = torch.sort(spike_loc)

        # # there will be one less isi than the length of the spiketrain
        # isi = torch.zeros((len(spike_loc - 1)))

        # # starting at index 1, calculate the isi for [0-1] and [1-2].
        # # then jump to index 3, calculate [2-3] and [3-4]
        # # keep in mind that if a list is might not have the 1+ith entry
        # for i in range(1,len(spike_loc), 2):
        #     isi[i-1] = spike_loc[i] - spike_loc[i-1]
        #     if i+1 < len(spike_loc):
        #         isi[i] = spike_loc[i+1] - spike_loc[i]
        
        # return {
        #     "no_spk": spike_train.sum(),
        #     "rate": spike_train.sum() / 1e3,
        #     "avg": isi.mean(),
        #     "med": isi.median(),
        #     "var": isi.var(),  
        #     "stdev": isi.std()
        # }

    def measure_latency(
        self,
        spike_train: torch.Tensor
    ):
        raise NotImplementedError
        if spike_train.isnan().any():
            raise ValueError(
                "Some Value in the Spike-Train is nan.\n" + 
                f"Offending Value(s): {spike_train[spike_train.isnan().nonzero(as_tuple = True)]}\n" +
                f"At indices: {spike_train.isnan().nonzero()}"
            )
        if spike_train.isinf().any():
            raise ValueError(
                "Some Value in the Spike-Train is inf.\n" + 
                f"Offending Value(s): {spike_train[spike_train.isinf().nonzero(as_tuple = True)]}\n" +
                f"At indices: {spike_train.isinf().nonzero()}"
            )

    def measure_rate(
        self,
        spikes: torch.Tensor,
        dt: float = 0.001,
        tau: float = 0.02,
    ) -> torch.Tensor:
        """
        Estimate instantaneous firing rate from spike trains using
        exponential kernel smoothing.

        Parameters
        ----------
        spikes : torch.Tensor
            Binary spike tensor of shape [neurons, time_steps].
            Values should be 0/1 (or spike counts per bin).

        dt : float
            Time step size in seconds.
            Example: dt=0.001 for 1 ms bins.

        tau : float
            Exponential kernel time constant in seconds.
            Example: tau=0.02 for 20 ms smoothing.

        Returns
        -------
        rates : torch.Tensor
            Smoothed firing rates in Hz.
            Shape: [neurons, time_steps]
        """

        # kernel length: ~5 tau captures most of exponential decay
        kernel_length = int(5 * tau / dt)

        t = torch.arange(
            kernel_length, 
            device = spikes.device, 
            dtype = spikes.dtype
        ) * dt

        # causal exponential kernel
        kernel = torch.exp(-t / tau)

        # normalize kernel so output is in spikes/sec (Hz)
        kernel = kernel / kernel.sum() / dt

        # reshape for conv1d
        kernel = kernel.view(1, 1, -1)

        # input shape for conv1d: [batch, channels, time]
        x = spikes.unsqueeze(1)

        # left-pad for causal filtering
        x = torch.nn.functional.pad(x, (kernel_length - 1, 0))

        # convolution
        rates = torch.nn.functional.conv1d(x, kernel)

        return rates.squeeze(1)

    # from https://github.com/rainsummer613/snn-saliency-familiarity-coding/blob/main/src/measure.py, adapted to pytorch
    def measure_rsync(
        self,
        spike_train: torch.Tensor
    ):
        """
        Computes the rsync measure. Adapted to work with pytorch from https://github.com/rainsummer613/snn-saliency-familiarity-coding/blob/main/src/measure.py.
        
        :param spike_train: Spike train of a layer or a model. Shape: (n_cells, time_steps)
        :type spike_train: torch.Tensor, required

        :returns: Tensor with the computed RSync values
        :rtype: torch.Tensor

        :raises ValueError: If any value in spike_train is inf or NaN.
        """
        if spike_train.isnan().any():
            raise ValueError(
                "Some Value in the Spike-Train is nan.\n" + 
                f"Offending Value(s): {spike_train[spike_train.isnan().nonzero(as_tuple = True)]}\n" +
                f"At indices: {spike_train.isnan().nonzero()}"
            )
        if spike_train.isinf().any():
            raise ValueError(
                "Some Value in the Spike-Train is inf.\n" + 
                f"Offending Value(s): {spike_train[spike_train.isinf().nonzero(as_tuple = True)]}\n" +
                f"At indices: {spike_train.isinf().nonzero()}"
            )
        
        # --- exponential convolution kernel ---
        tau = 3.0  # ms

        device = spike_train.device
        dtype = spike_train.dtype

        exp_kernel_time_steps = torch.arange(
            0,
            int(tau * 10),
            device=device,
            dtype=dtype
        )

        exp_kernel = torch.exp(-exp_kernel_time_steps / tau)

        # shape for conv1d: (out_channels, in_channels, kernel_size)
        kernel = exp_kernel.view(1, 1, -1)

        # input shape for conv1d: (batch, channels, time)
        x = spike_train.unsqueeze(1)

        # same-padding
        padding = kernel.shape[-1] // 2

        # convolve each neuron independently
        spike_train = torch.nn.functional.conv1d(
            x, 
            kernel, 
            padding = padding
        ).squeeze(1)

        # match np.convolve(..., mode="same") output length for even kernels
        spike_train = spike_train[:, :x.shape[-1]]

        # --- rsync computation ---
        meanfield = torch.mean(spike_train, dim = 0)     # spatial mean across cells
        variances = torch.var(spike_train, dim = 1)      # temporal variance per cell

        rsync = torch.var(meanfield) / torch.mean(variances)

        if rsync.isnan().any():
            rsync = torch.tensor(0.0, device = device, dtype = dtype)

        return rsync
    
    def visualise(
        self,
        blocking: bool = False
    ) -> Figure:
        
        # instantiate plot
        fig, axes = plt.subplots(
            nrows = 2,
            ncols = len(self.recorder.monitored_layers),
            squeeze = False,
            sharex = 'all',
            figsize = (14,5),
            dpi = 200
        )

        # disentangle the data
        # of structure:
        # recorder
        #   dict (with keys monitored_layers)
        #       list (len time_steps)
        #           tuple (spikes, membrane)
        #               spikes / membrane ofc have a shape of [minibatch, out_features]
        for i, layer in enumerate(self.recorder.monitored_layers):
            # i'm only interested in the first sample for now
            layerlist = self.recorder[layer][:self.time_steps]
            spikes = torch.stack([x[0,:] for x, _ in layerlist]).T
            membrane = torch.stack([x[0,:] for _, x in layerlist])

            print(f"Layer: {layer}")
            print(f"Spikes: Max: {spikes.max()}, Sum: {spikes.sum()}, Size: {spikes.shape}")
            print(f"Membrane: Max: {membrane.max()}, Sum: {membrane.sum()}, Size: {membrane.shape}")

            axes[0][i].spy(
                spikes,
                markersize = max(i * .2, 0.1)
            )
            axes[0][i].set_aspect('auto')
            axes[1][i].plot(
                membrane
            )
            axes[1][i].set_aspect('auto')
        plt.tight_layout(pad = 2.0)
        plt.show()
        return fig

    def visualise_tendencies(
        self,
        output
    ) -> Figure:
        fig, axes = plt.subplots(
            nrows = 1, 
            ncols = 3,
            squeeze = True,
            figsize = (14,5),
            dpi = 200
        )

        for i, layer in enumerate(output):
            layer = torch.where(layer == 0, torch.tensor(float('nan')), layer)
            axes[i].scatter(torch.arange(len(layer)), layer)
        
        
        plt.tight_layout()
        plt.show()
        return fig