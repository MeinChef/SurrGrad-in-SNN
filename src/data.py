from imports import yaml
from imports import Path
from imports import os
from imports import datetime
from imports import functional
from imports import torch
from imports import plt
from imports import Figure, Axes
from imports import cm
from imports import PCA


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


    def measure_tendencies(
        self
    ) -> dict:

        measurements = {}

        for layer in self.recorder.monitored_layers:
            # create tensor from recording
            layerlist = self.recorder[layer][:self.time_steps]
            # only interested in spikes and first sample
            #               first sample   membrane potentials
            #                       |    spikes |
            #                       |        |  |
            spikes = torch.stack([x[0,:] for x, _ in layerlist]).T
            
            # calculate the average isi, synchrony
            measurements[layer] = {
                "spikes": spikes,
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
        
        self._tendencies = measurements
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
        
        return torch.nested.nested_tensor(isis, layout = torch.jagged)

    def measure_rate(
        self,
        spikes: torch.Tensor,
        dt: float = 0.001,
        tau: float = 0.02,
    ) -> torch.Tensor:
        # TODO: rewrite to be in line with other docstrings
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
    
    ######################
    ### Fancy Plotting ###
    ######################

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
        save: bool = True,
        name_ext: str | None = None,
        blocking: bool = False
    ) -> Figure:

        fig, axes = plt.subplots(
            nrows = 4,                              # raster plot, heatmap of smoothed rates, rsync, pca trajectory of rates? (idk about last one, slopmachine suggested that)
            ncols = len(self._tendencies),          # layers as cols
            squeeze = True,
            figsize = (16,16),
            dpi = 100
        )

        for i, layer in enumerate(self._tendencies):
            axes[0, i] = self._plot_spikes(axes[0, i], layer)
            axes[1, i] = self._plot_rate_heatmap(fig, axes[1, i], layer)
            # axes[2, i] = self._plot_rsync(axes[2, i], layer)
            axes[3, i] = self._plot_pca_trajectory(axes[3, i], layer)
        
        
        fig.tight_layout()

        if save:
            fig.savefig(
                os.path.join(
                    Path(__file__).parent.parent,
                    "img",
                    "tendencies-" + name_ext if name_ext else self.now
                ),
                format = "svg"
            )
        if blocking:
            plt.show()

        return fig
    

    def _plot_spikes(
        self,
        axes: Axes,
        layer: str = "neuron1"
    ) -> Axes:

        axes.scatter(
            *torch.nonzero(
                self._tendencies[layer]["spikes"].cpu().T,
                as_tuple = True
            ),
            s = 1.5,
            c = "black"
        )

        axes.set_xlabel("Time")
        axes.set_ylabel("Neurons")
        axes.set_title("Spikes")

        return axes

    def _plot_rate_heatmap(
        self,
        fig: Figure,
        axes: Axes,
        layer: str,
        dt = 0.001
    ) -> Axes:

        rates = self._tendencies[layer]["smoothed_rates"].cpu().numpy()

        cmap = cm.get_cmap("viridis")
        im = axes.imshow(
            rates,
            aspect = 'auto',
            origin = 'lower',
            cmap = cmap,
            extent = (0, rates.shape[1] * dt, 0, rates.shape[0])
        )

        fig.colorbar(
            im,
            ax = axes,
            label = 'Firing rate (Hz)'
        )
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Neuron')
        axes.set_title('Instantaneous firing rates')

        return axes
    
    # def _plot_rsync(
    #     self,
    #     axes: Axes,
    #     layer: str = "neuron1"
    # ) -> Axes:
    #     # doing this is kinda stupid, since there is only one value per layer
    #     rsync = self._tendencies[layer]["rsync"]
    #     return axes

    def _plot_pca_trajectory(
        self,
        axes: Axes,
        layer: str = "neuron1"
    ) -> Axes:

        # shape: [neurons, time]
        X = self._tendencies[layer]["smoothed_rates"].cpu().numpy().T

        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(X)

        axes.scatter(X_pca[:, 0], X_pca[:, 1])

        axes.set_xlabel("PC1")
        axes.set_ylabel("PC2")
        axes.set_title("Neural population trajectory")

        return axes