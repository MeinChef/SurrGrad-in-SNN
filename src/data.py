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
from imports import NOW
from synth_model import SynthModel

# load the config.yml
def load_config(
        path: str | None = None
) -> tuple[dict, dict]:
    """
    Convenience function to load a .yml file containing two dictionaries (`data` and `model`). 
    Default keys for both dictionaries can be found in the `config.yml` at the root of this repository.
    
    If path is not given, will assume the config file to be in the root folder 
    of the repository (`../config.yml` from this file).
    
    :param path: An absolute path pointing to a .yml file.
    :type path: str or None, optional (Default: None) 
    :returns: Tuple containing both Dictionaries
    :rtype: tuple[dict, dict]
    """

    # load default config located at ../config.yml
    if path is None:
        path = os.path.join(
            Path(__file__).parent.parent,
            "config.yml"
        )

    with open(path, "r") as file:
        configs = yaml.safe_load(file)

    return configs["data"], configs["model"]

def save_model(
    model: SynthModel,
    identifier: str | None = None,
) -> None:
    """
    Helper function to save the model.
    
    :param model: An instance of the SynthModel class
    :type model: SynthModel, required
    :param identifier: Filename the model should be saved to. 
                    If not provided, a timestamp is being used.
    :type identifier: str or None, optional (Default: None)
    """
    if identifier is None:
        identifier = NOW + ".pt"

    if "." not in identifier:
        raise ValueError(
            "Expected file extentsion. Try again with a '.fileextension'."
        )
    
    os.makedirs(
        os.path.join(
            Path(__file__).parent.parent,
            "model"
        ),
        exist_ok = True
    )
    modelpath = os.path.join(
        Path(__file__).parent.parent,
        "model",
        "synthmodel-" + identifier
    )
    torch.save(
        model.state_dict(),
        modelpath
    )

    print("Saved model to: " + modelpath)

def load_model(
    identifier: str | None = None,
) -> SynthModel:
    """
    A helper function to nicely load a Model previously saved to Disk.

    :param identifier: A string to load a spceific model from Disk. 
                If not given, will use the same timestamp as save_model.
    :type identifier: str or None, optional (Default: None)

    :returns: An Instance of SynthModel, fully inferred from the according file
    :rtype: SynthModel
    """

    if identifier is None:
        identifier = NOW + ".pt"

    if "." not in identifier:
        raise ValueError(
            "Expected file extentsion. Try again with a '.fileextension'."
        )
    

    # load config
    _, cfg = load_config(
        os.path.join(
            Path(__file__).parent.parent,
            "config.yml"
        )
    )
    
    # load model
    model = SynthModel(cfg)
    model.load_state_dict(
        torch.load(
            os.path.join(
                Path(__file__).parent.parent,
                "model",
                "synthmodel-" + identifier
            )
        )
    )

    model.eval()
    return model

class DataHandler():
    """
    A Class for neatly wrapping an OutputMonitor.
    Capable of measuring various metrics, and neatly visualising them.
    """
    def __init__(
        self,
        recorder: functional.probe.OutputMonitor,
        time_steps: int = 1000,
        datapath: str = "data/",
    ) -> None:
        """
        A Class for neatly wrapping an OutputMonitor.
        Capable of measuring various metrics, and neatly visualising them.

        :param recorder: An instance of the snntorch OutputMonitor, already wrapping a model.
        :type recorder: snntorch.functional.probe.OutputMonitor
        :param time_steps: The time-length of the samples passed through the network.  
        :type time_steps: int
        :param datapath: Path to a folder where the output should be saved to.
        :type datapath: str, optional
        """

        self.recorder = recorder
        self.path = datapath
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.time_steps = time_steps
        self._tendencies = None

        self.recorder.disable()

    def enable(self) -> None:
        """
        Wrapper for enabling the OutputMonitor
        """
        self.recorder.enable()

    def disable(self) -> None:
        """
        Wrapper for disabling the OutputMonitor
        """
        self.recorder.disable()

    def clear_recorded_data(self) -> None:
        """
        Wrapper for clearing the recorded data of the OutputMonitor
        """
        self.recorder.clear_recorded_data()


    def measure_tendencies(
        self,
        data: torch.utils.data.DataLoader
    ) -> dict:
        """
        Measures various metrics and returns the results in a dictionary.
        
        The first batch of the passed Dataloader will be inspected, and for each sample in this batch, 
        across every layer of the model, the following metrics will be calculated:\n
        - Instantaneous firing rate (Exponential Kernel)\n
        - RSync (Adapted from Zemliack et. al (2025))\n
        - Inter Spike Intervals\n
        
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013304

        :param data: A dataloader containing a (probably condensed) 
                    set of Samples which should be evaluated.
        :type data: torch.DataLoader
        :returns: A Dictionary, with keys `sample-#`, which itself is a dictionary 
                    containing the measurements (another dictionary) and the label.
        :rtype: dict 
        """
        
        all_measure = {}
        
        # assuming only one batch, since that's the cleanest way
        inputs, label = next(iter(data))
        for n, cls in enumerate(label):
            
            all_measure[f"sample-{n}"] = {}
            all_measure[f"sample-{n}"]["measurements"] = {}
            all_measure[f"sample-{n}"]["class"] = cls
            all_measure[f"sample-{n}"]["input"] = inputs[n]

            for layer in self.recorder.monitored_layers:
                # create tensor from recording
                layerlist = self.recorder[layer][:self.time_steps]
                # only interested in spikes and first sample
                #               first sample   membrane potentials
                #                       |    spikes |
                #                       |        |  |
                spikes = torch.stack([x[n,:] for x, _ in layerlist]).T
                membrane = torch.stack([x[n,:] for _, x in layerlist]).T
                
                # calculate the average isi, synchrony
                all_measure[f"sample-{n}"]["measurements"][layer] = {
                    "spikes": spikes,
                    "membrane": membrane,
                    "neurons": spikes.shape[0],
                    "time_steps": spikes.shape[1],
                    "rates": spikes.sum(1),
                    "smoothed_rates": self.measure_rate(spikes),
                    "rsync": self.measure_rsync(spikes),
                    "isis": self.measure_isis(spikes)
                }

        self._tendencies = all_measure
        return all_measure

    def measure_isis(
        self,
        spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Function to calculate the time-difference between the spikes of each neuron.
        Since not every neuron spikes the same amount of times, the returning tensor is jagged in the second dimension.

        :param spikes: The spike-train of a layer.
        :type spikes: torch.Tensor
        :returns: A Tensor with the same first dim, but jagged in the second.
        :rtype: torch.Tensor
        """
        if spikes.device != "cpu":
            spikes = spikes.to(torch.device('cpu'))

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
            mask = torch.where(spk_neuron == neuron)[0]
            # calculates the "forward difference", so N+1 - N
            isis.append(torch.diff(spk_times[mask]))
        
        try:
            # this did at some point _once_ raise a cuda assertion error. IDK why.
            # /pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111: operator(): block: [0,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
            # Traceback (most recent call last):
            #   File "/home/user/Documents/code/SurrGrad-in-SNN/src/test.py", line 84, in <module>
            #     main()
            #   File "/home/user/Documents/code/SurrGrad-in-SNN/src/test.py", line 78, in main
            #     handler.measure_tendencies()
            #   File "/home/user/Documents/code/SurrGrad-in-SNN/src/data.py", line 135, in measure_tendencies
            #     "isis": self.measure_isis(spikes)
            #             ^^^^^^^^^^^^^^^^^^^^^^^^^
            #   File "/home/user/Documents/code/SurrGrad-in-SNN/src/data.py", line 176, in measure_isis
            #     return torch.nested.nested_tensor(isis, layout = torch.jagged)
            #            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #   File "/home/user/miniconda3/envs/snn/lib/python3.12/site-packages/torch/nested/__init__.py", line 272, in nested_tensor
            #     nt, _ = jagged_from_list(
            #             ^^^^^^^^^^^^^^^^^
            #   File "/home/user/miniconda3/envs/snn/lib/python3.12/site-packages/torch/nested/_internal/nested_tensor.py", line 522, in jagged_from_list
            #     torch.tensor(
            # torch.AcceleratorError: CUDA error: device-side assert triggered
            # 
            # IF error is triggered: isis are two empty tensors
            # I think this should be resolved by the spikes.to("cpu") earlier

            out = torch.nested.nested_tensor(isis, layout = torch.jagged)
        except Exception as e:
            out = torch.tensor([0])
            breakpoint()
        
        return out

    def measure_rate(
        self,
        spikes: torch.Tensor,
        dt: float = 0.001,
        tau: float = 0.02,
    ) -> torch.Tensor:
        """
        Estimate instantaneous firing rate from spike trains using
        exponential kernel smoothing.

        :param spikes:  Binary spike tensor of shape [neurons, time_steps].
                    Values should be 0/1 (or spike counts per bin).
        :type spikes: torch.Tensor
        :param dt: Time step size in seconds.
                    Example: dt=0.001 for 1 ms bins.
        :type dt: float
        :param tau: Exponential kernel time constant in seconds.
                    Example: tau=0.02 for 20 ms smoothing.
        :type tau: float
            
        :returns: Smoothed firing rates in Hz.
                    Shape: [neurons, time_steps]
        :rtype: torch.Tensor
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

    # from https://github.com/rainsummer613/snn-saliency-familiarity-coding/blob/main/src/measure.py
    # adapted to pytorch
    def measure_rsync(
        self,
        spike_train: torch.Tensor
    ):
        """
        Computes the rsync measure. Adapted to work with pytorch from 
        https://github.com/rainsummer613/snn-saliency-familiarity-coding/blob/main/src/measure.py.
        
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
    ) -> None:
        """
        Function for visualising the previously calculated measurements.
        Creates multiple pyplot figures, one for each sample of the batch.
        Each figure either gets saved and discarded (if `save` and not `blocking`) or put in RAM (`blocking = False`).

        The figures are structured as follows:\n
        Row one contains the spike trains of each layer. 
        Row two contains heatmaps of the instantaneous firing rate.
        Row three contains histograms of the InterSpike-Intervals.
        
        **Caution!**\n
        If blocking is true, each figure will stay in RAM until every sample has its own figure. 
        This can be very memory expensive!

        :param save: Whether to save the figures to files.
        :type save: bool, optional (Default: True)
        :parame name_ext: An identifier that gets added to each filename, 
                    only relevant when save=True. If None, uses a timestamp.
        :type name_ext: str or None, optional (Default: None)
        :param blocking: Whether to "block" the execution of the function by showing the figures.
        :type blocking: bool, optional (Default: False)
        :raises ValueError: If measure_tendencies has not been called before 
                    and or measurements cannot be found.
        """
        
        if self._tendencies is None:
            raise ValueError(
                "Tendencies have not been calculated before. " \
                "Please call measure_tendencies() before you call this function."
            )
        
        if save:
            os.makedirs(
                os.path.join(
                    Path(__file__).parent.parent,
                    "img",
                    self.now
                ),
                exist_ok = True
            )
            os.environ["MPLBACKEND"] = "svg"

        FIG_SIZE = (16,16)
        DPI = 300
        NROWS = 5
        HEIGTH_RATIOS = [23,23,1,12,23]
        
        fig = plt.figure(
            num = 1, 
            clear = True,
            figsize = FIG_SIZE,
            dpi = DPI   
        )

        for key in self._tendencies.keys():
            measurements = self._tendencies[key]["measurements"]
            
            if save and not blocking:
                axes = fig.subplots(
                    nrows = NROWS,                                           
                    # raster plot, heatmap of smoothed rates, rsync, pca trajectory of rates? 
                    # (idk about last one, slopmachine suggested that)
                    ncols = len(measurements),  # layers as cols
                    squeeze = True,
                    height_ratios = HEIGTH_RATIOS
                )
            else:
                fig, axes = plt.subplots(
                    nrows = NROWS,                                           
                    # raster plot, heatmap of smoothed rates, rsync, pca trajectory of rates? 
                    # (idk about last one, slopmachine suggested that)
                    ncols = len(measurements),  # layers as cols
                    squeeze = True,
                    figsize = FIG_SIZE,
                    dpi = DPI,
                    height_ratios = HEIGTH_RATIOS
                )

            for i, layer in enumerate(measurements):
                self._plot_spikes(axes[0, i], measurements[layer])
                if i == len(measurements) - 1:
                    self._plot_membrane(axes[0, i], measurements[layer])
                self._plot_rate_heatmap(
                    fig = fig, 
                    axes = axes[1, i],
                    cax = axes[2, i],
                    data = measurements[layer]
                )
                self._plot_isis(axes[3, i], measurements[layer])
                self._plot_pca_trajectory(axes[4, i], measurements[layer])
        
            fig.suptitle(
                f"Spike Analysis of Class {self._tendencies[key]["class"].item()}"
            )
            fig.tight_layout()

            if save:
                suffix = name_ext if name_ext else self.now
                fig.savefig(
                    os.path.join(
                        Path(__file__).parent.parent,
                        "img",
                        self.now,
                        f"tendencies-{suffix}-{key}.svg"
                    ),
                    format = "svg"
                )

                fig.clear()


        if blocking:
            plt.show()


    def _plot_spikes(
        self,
        axes: Axes,
        data: dict
    ) -> Axes:
        """
        Convenience function for plotting spikes on an Axis.
        Will also put the Rsync of the spikes in the title.

        :param axes: An axes of e.g. a plt.subplots()
        :type axes: plt.Axes
        :param data: A dictionary containing the neccessary data.
                Required Keys: spikes, neurons, time_steps and rsync.
        :type data: dict
        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """
        s = None
        if data["neurons"] > 5 and data["neurons"] <= 100:
            s = 2
        elif data["neurons"] > 100:
            s = 1

        axes.scatter(
            *torch.nonzero(
                data["spikes"].cpu().T,
                as_tuple = True
            ),
            s = s,
            c = "black",
            marker = "|"
        )

        axes.set_xlabel("Time (ms)")
        axes.set_xlim(0, data["time_steps"])
        axes.set_ylabel("Neurons")
        axes.set_ylim(-0.5, data["neurons"] - 0.5)
        axes.set_title("Spikes - RSync of " + "{:.2f}".format(data["rsync"]))

        return axes

    def _plot_membrane(
        self,
        axes: Axes,
        data: dict
    ) -> Axes:
        """
        Convenience function for plotting the membrane potential on an Axis.

        :param axes: An axes of e.g. a plt.subplots()
        :type axes: plt.Axes
        :param data: A dictionary containing the neccessary data.
                Required Keys: membrane, neurons, time_steps.
        :type data: dict
        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """
        
        mem = data["membrane"].cpu().numpy()
        # offset each membrane potential to display them above each other
        for neuron in range(mem.shape[0]):
            mem[neuron] += neuron * 2
    
        secax = axes.twinx()
        secax.plot(
            mem.T,
            c = "orange",
            alpha = 0.5
        )
        secax.set_ylim(-1, mem.shape[0] * 2 - 1)
        secax.set_ylabel("Membrane Voltage")
        return secax


    def _plot_rate_heatmap(
        self,
        fig: Figure,
        axes: Axes,
        data: dict,
        cax: Axes | None = None
    ) -> Axes:
        """
        Convenience function for plotting a heatmap of the instantaneous firing rates.

        :param fig: The Figure that plots will be put on.
        :type fig: plt.Figure
        :param axes: Axes the Heatmap will be plotted onto.
                If `cax` is, some space of the Axes will be used to plot the colorbar (legend). 
        :type axes: plt.Axes
        :param data: A dictionary containing the neccessary data. 
                Required Keys: smoothed_rates, time_steps
        :type data: dict
        :param cax: An optional Axes to put the colorbar onto
                (legend for the heatmap).
                If None, will steal space from the `axes`
        :type cax: plt.Axes or None, optional (Default: None)

        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """
        
        rates = data["smoothed_rates"].cpu().numpy()

        cmap = cm.get_cmap("viridis")
        im = axes.imshow(
            rates,
            aspect = 'auto',
            origin = 'lower',
            cmap = cmap,
            # extent = (0, rates.shape[1] * dt, 0, rates.shape[0])
        )

        fig.colorbar(
            im,
            ax = axes if not cax else None,
            cax = cax if cax else None,
            label = 'Firing rate (Hz)',
            location = "bottom"
        )
        axes.set_xlabel('Time (ms)')
        axes.set_xlim(0, data["time_steps"])
        axes.set_ylabel('Neuron')
        axes.set_title('Instantaneous firing rates')

        return axes
    
    def _plot_isis(
        self,
        axes: Axes,
        data: dict
    ) -> Axes:
        """
        Convenience function for plotting a histogram of ISIs onto a given Axes.
        Will filter out 0-ISIs, since those (should) not exist, and or be fill-values.

        :param axes: Axes to plot the ISIs onto.
        :type axes: plt.Axes
        :param data: A dictionary containing the neccessary data. 
                Required Keys: isis
        :type data: dict
                
        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """
        # doing this is kinda stupid, since there is only one value per layer
        # breakpoint()

        data_clean = torch.nested.to_padded_tensor(
            data["isis"],
            padding = 0
        ).to(torch.float)

        if data_clean.numel() == 0:
            data_clean = torch.tensor([0], dtype = torch.float)
        
        data_hist = torch.histc(
            input = data_clean,
            bins = 100,
            min = 1,
            max = max(2, data_clean.amax().item())
        ).numpy()

        axes.stairs(
            values = data_hist,
            fill = True
        )
        axes.set_xlabel("Bins")
        axes.set_ylabel("Counts")
        axes.set_yscale("log")
        axes.set_title("Histogram of ISIs")

        return axes

    def _plot_pca_trajectory(
        self,
        axes: Axes,
        data: dict
    ) -> Axes:
        """
        Convenience function to plot the PCA trajectory of the smoothed rates.

        :param axes: Axes to plot the PCA onto.
        :type axes: plt.Axes
        :param data: A dictionary containing the neccessary data. 
                Required Keys: smoothed_rates
        :type data: dict
                
        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """
        # shape: [neurons, time]
        X = data["smoothed_rates"].cpu().numpy().T

        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(X)

        axes.scatter(X_pca[:, 0], X_pca[:, 1])

        axes.set_xlabel("PC1")
        axes.set_ylabel("PC2")
        axes.set_title("Neural population trajectory")

        return axes