from imports import (
    DEFAULT_CONFIG_PATH,
    DEVICE,
    NOW,
    PCA,
    ROOT,
    Axes,
    Figure,
    Literal,
    Path,
    functional,
    matplotlib,
    os,
    pickle,
    plt,
    shutil,
    torch,
    yaml,
)
from imports import numpy as np
from imports import seaborn as sns
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
        path = DEFAULT_CONFIG_PATH

    with open(path, "r") as file:
        configs = yaml.safe_load(file)

    return configs["data"], configs["model"]

def save_model(
    model: SynthModel,
    data_path: str | None = None,
    config_path: str | None = None,
    identifier: str = NOW,
) -> None:
    """
    Helper function to save the model.

    :param model: An instance of the SynthModel class
    :type model: SynthModel, required
    :param identifier: Filename the model should be saved to.
                    If not provided, a timestamp is being used.
    :type identifier: str or None, optional (Default: None)
    """

    # check if all the parameters are set, and set them if not

    if Path(identifier).suffix == '':
        identifier = identifier + ".pt"

    if data_path is None:
        data_path = os.path.join(
            ROOT,
            "data",
            identifier,
        )
    data_path = os.path.join(data_path, "model")

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # create a directory and save the model to it
    os.makedirs(
        data_path,
        exist_ok = True
    )
    model_path = os.path.join(
        data_path,
        "synthmodel-" + identifier
    )
    torch.save(
        model.state_dict(),
        model_path
    )

    # and also save a copy of the config
    shutil.copy(
        src = config_path,
        dst = os.path.join(
            data_path,
            "synthmodel-" + Path(identifier).stem + ".yml"
        )
    )

    print("Saved model to: " + model_path)

def request_model_save(
    model: SynthModel,
    data_path: str | None = None,
    config_path: str | None = None,
    identifier: str = NOW
) -> None:
    valid = False
    while not valid:
        # request User input
        inp = input(
            "The model has not been saved during this last epoch, "
            "because it's worse than at its best.\n"
            "Do you still want to save it? [Y/n]"
        )
        # do some cleaning
        inp = inp.lower().strip()

        # and call the save function and exit
        if inp in ["yes", "y", ""]:
            save_model(
                model = model,
                data_path = data_path,
                config_path = config_path,
                identifier = identifier
            )
            return

        if inp in ["no", "n"]:
            return

        print("Invalid Input!")


def load_model(
    data_path: str | None = None,
    config_path: str | None = None,
    identifier: str  = NOW,
) -> SynthModel:
    """
    A helper function to nicely load a Model previously saved to Disk.

    :param identifier: A string to load a specific model from Disk.
                If not given, will use the same timestamp as save_model.
    :type identifier: str or None, optional (Default: None)

    :returns: An Instance of SynthModel, fully inferred from the according file
    :rtype: SynthModel
    """

    # check if all the parameters are set, and set them if not

    if Path(identifier).suffix == '':
        identifier = identifier + ".pt"

    if data_path is None:
        data_path = os.path.join(
            ROOT,
            "data",
            identifier,
        )
    data_path = os.path.join(data_path, "model")


    # try loading config that's probably been saved alongside the model:
    if config_path is None:
        try:
            cfg_name = Path(identifier).stem + ".yml"
            _, cfg = load_config(os.path.join(
                data_path,
                "synthmodel-" + cfg_name
            ))
        except FileNotFoundError:
            # if that didn't work, fall back to default choice
            _, cfg = load_config(config_path)    
    else:
        _, cfg = load_config(config_path)    

    # load model
    model_path = os.path.join(
        data_path,
        "synthmodel-" + identifier
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SynthModel(cfg)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location = DEVICE
        ),
        strict = False
    )

    model.eval()
    print(f"Successfully loaded Model from {model_path}!")
    return model

#####################################################
######### DataHandler Class Definition here #########
#####################################################

class DataHandler:
    """
    A Class for neatly wrapping an OutputMonitor.
    Capable of measuring various metrics, and neatly visualising them.
    """
    def __init__(
        self,
        recorder: functional.probe.OutputMonitor,
        time_steps: int = 1000,
        data_path: str | None = None,
        identifier: str = NOW
    ) -> None:
        """
        A Class for neatly wrapping an OutputMonitor.
        Capable of measuring various metrics, and neatly visualising them.

        :param recorder: An instance of the snntorch OutputMonitor, already wrapping a model.
        :type recorder: snntorch.functional.probe.OutputMonitor
        :param time_steps: The time-length of the samples passed through the network.
        :type time_steps: int
        :param data_path: Path to a folder where the output should be saved to.
        :type data_path: str, optional
        """

        self.recorder = recorder
        self.id = identifier

        # path stuff
        if data_path is None:
            data_path = os.path.join(
                ROOT,
                "data",
                identifier
            )
        self.img_path = os.path.join(
            data_path,
            "img"
        )
        self.bin_path = os.path.join(
            data_path,
            "bin"
        )

        self.time_steps = time_steps
        self._tendencies = {}

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

    def get_network_response(
        self,
        model: SynthModel,
        data: torch.utils.data.DataLoader,
        # split: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts network responses from a trained model and splits the resulting dataset
        into train/validation/test subsets according to the specified split ratios.

        This function runs the model on the provided data loader in evaluation mode,
        records the spike outputs from the final layer of the network (assuming a spiking
        neural network), and constructs a tensor dataset of (spike activations, labels).


        :param model: The trained model (e.g., SynthModel) to probe.
        :type model: SynthModel
        :param data: DataLoader providing batches of input data (x, label).
        :type data: torch.utils.data.DataLoader
        :param split: List of fractions for train/validation/test splits. Must sum to 1.0.
                    Default is [0.7, 0.15, 0.15].
        :type split: list[float] | None
        :return: A list of torch.utils.data.Subset objects corresponding to the split
                datasets (train, val, test).
        :rtype: list[torch.utils.data.Subset]
        :raises AssertionError: If the sum of split fractions is not approximately 1.0.
        :raises ValueError: If the data loader is empty or the model produces no output.
        """

        model.eval()
        # new output monitor to make sure we're  not accidentally deleting data from another one
        rec = functional.probe.OutputMonitor(model)
        rec.enable()
        last_layer_key = rec.monitored_layers[-1]

        outputs = []
        labels = []
        time_steps = 0
        with torch.no_grad():
            for x, label in data: # add tqdm
                time_steps = x.shape[0]
                model(x, batch_first = False)
                labels.append(label)                    # [batch_size]

        # gets list[tuple(t0)[spk, mem], tuple(t1)[spk, mem], ...]
        recordings = rec[last_layer_key]                # [T*B, N]
        # filters for spikes; list[spk(t0), spk(t1), ...]
        recordings = [x for x, _ in recordings]         # [T*B, N]
        # since recordings is now over the whole dataset, it needs splitting
        # each sample should have only time_steps time steps
        # the whole recordings list should be no_batches * time_steps
        assert len(recordings) == len(data) * time_steps
        chunks = []
        for n in range(len(data)):
            chunk = recordings[
                n * time_steps : (n+1) * time_steps     # [T, N]
            ]
            chunks.append(torch.stack(chunk))
        # chunks should be now [len(data)] with items of [T, B, N]

        # make them to giant tensors
        outputs = torch.cat(chunks, dim = 1)            # [T, all_samples, N]
        outputs = outputs.permute(1, 0, -1)             # [all_samples, T, N]
        labels  = torch.cat(labels)                     # [all_samples]

        self.raw_outputs = outputs
        self.raw_labels = labels
        return outputs, labels

    def get_output_repr(
        self,
        type: Literal["count", "smoothed"],  # noqa: F821
        window_width: int = 20,
        step: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "raw_outputs"):
            raise RuntimeError("Please run get_network_response() beforehand.")

        starts = torch.arange(
            0,
            self.time_steps - window_width + 1,
            step
        )
        outputs = []
        centers = starts + window_width / 2

        if type == "count":
            for start in starts:
                out = self.raw_outputs[
                    :,
                    :,
                    start: start + window_width
                ]
                outputs.append(out.sum(dim = 1)) # sum window over time

        elif type == "smoothed":
            rates = self.measure_rate(
                self.raw_outputs,
                dt = 1,
                tau = 5,
            )
            for start in starts:
                out = rates[
                    ...,
                    start: start + window_width
                ]
                outputs.append(out.flatten(start_dim=1))

        return centers, torch.stack(outputs, dim = 1)


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
        :param loss: A list of losses that got calculated while recording the model.
        :type loss: list
        :param acc: A list of accuracies that got calculated while recording the model.
        :type acc: list

        :returns: A Dictionary, with keys `sample-#`, which itself is a dictionary
                    containing the measurements (another dictionary) and the label.
                    Additionally the keys "loss" and "accuracy" are saved, if they are passed.
        :rtype: dict
        """

        all_measure = {}

        # assuming only one batch, since that's the cleanest way
        inputs, labels = next(iter(data))
        # inputs should be a tensor of shape [time, batch, neuron]
        # and we want [batch, neuron, time] here
        inputs = torch.permute(inputs, (1,2,0))

        for n, cls in enumerate(labels):
            this_input = inputs[n]
            all_measure[f"sample-{n}"] = {}
            all_measure[f"sample-{n}"]["measurements"] = {}
            all_measure[f"sample-{n}"]["class"] = cls
            all_measure[f"sample-{n}"]["input"] = this_input

            # do all the analysis first for the inputs
            all_measure[f"sample-{n}"]["measurements"]["neurons.-1"] = {
                "spikes": this_input,
                "membrane": None,
                "neurons": this_input.shape[0],
                "time_steps": this_input.shape[1],
                "rates": this_input.sum(1),
                "smoothed_rates": self.measure_rate(this_input),
                "rsync": self.measure_rsync(this_input),
                "isis": self.measure_isis(this_input)
            }

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
        # over spikes.shape[0], because not every neuron necessarily spikes
        for neuron in range(spikes.shape[0]):
            # only select relevant neurons, still works if mask is empty
            mask = torch.where(spk_neuron == neuron)[0]
            # calculates the "forward difference", so N+1 - N
            isis.append(torch.diff(spk_times[mask]))

        out = torch.nested.nested_tensor(isis, layout = torch.jagged)
        return out

    def measure_rate(
        self,
        spikes: torch.Tensor,
        dt: float = 1.,
        tau: float = 20.,
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
        t = t.flip(0)

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

        if not self._tendencies:
            raise UnboundLocalError(
                "Tendencies have not been calculated before. " \
                "Please call measure_tendencies() before you call this function."
            )

        if save:
            os.makedirs(
                self.img_path,
                exist_ok = True
            )
            os.environ["MPLBACKEND"] = "pdf"

        if name_ext is None:
            name_ext = self.id

        FIG_SIZE = (23.4, 16.5) # A2 papersize
        DPI = 300
        NROWS = 5
        HEIGHT_RATIOS = [23,23,1,12,23]

        fig = plt.figure(
            num = "tendency-vis",
            clear = True,
            figsize = FIG_SIZE,
            dpi = DPI
        )

        for key in self._tendencies:
            measurements = self._tendencies[key]["measurements"]

            if save and not blocking:
                axes = fig.subplots(
                    nrows = NROWS,
                    # raster plot, heatmap of smoothed rates, rsync, pca trajectory of rates? 
                    # (idk about last one, slopmachine suggested that)
                    ncols = len(measurements),  # layers as cols
                    squeeze = True,
                    height_ratios = HEIGHT_RATIOS
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
                    height_ratios = HEIGHT_RATIOS
                )

            self.ylabel = True
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

                if self.ylabel:
                    self.ylabel = False

            fig.suptitle(
                f"Spike Analysis of Class {self._tendencies[key]["class"].item()}"
            )
            fig.tight_layout()

            if save:
                fig.savefig(
                    os.path.join(
                        self.img_path,
                        f"tendencies-{name_ext}-{key}.pdf"
                    ),
                    format = "pdf"
                )
                fig.clear()

        if save:
            # # copy config
            # config_path = os.path.join(
            #     self.path,
            #     "config.yml"
            # )
            # # but only if it hasn't been copied yet
            # if not os.path.exists(config_path):
            #     shutil.copy(
            #         src = os.path.join(
            #             ROOT,
            #             "config.yml"
            #         ),
            #         dst = config_path
            #     )

            # and pickle the measurements
            with open(
                os.path.join(self.bin_path, f"measurements-{name_ext}.pkl"),
                "wb+"
            ) as file:
                pickle.dump(
                    self._tendencies,
                    file
                )

        self.plot_rsync(
            save = save,
            name_ext = name_ext
        )

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
        :param data: A dictionary containing the necessary data.
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

        # axes.set_xlabel("Time (ms)")
        axes.set_xlim(0, data["time_steps"])
        axes.set_ylim(-0.5, data["neurons"])
        axes.set_title("Spikes - RSync of " + "{:.2f}".format(data["rsync"]))
        if self.ylabel:
            axes.set_ylabel("Neurons")

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
        :param data: A dictionary containing the necessary data.
                Required Keys: membrane, neurons, time_steps.
        :type data: dict
        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """

        mem = data["membrane"].cpu().numpy()
        # Ensure mem is float32/float64 and not clipped
        mem = mem.astype(float)

        # Define offset: one unit apart (e.g., 1.0) for clarity
        offset = 1.0
        num_neurons = mem.shape[0]

        # Create offset membrane potentials
        # We'll add an offset per neuron: neuron * offset
        # But we'll keep the original voltage values intact
        offset_mem = mem + np.arange(num_neurons)[:, np.newaxis] * offset

        # Create a secondary y-axis
        secax = axes.twinx()

        # Plot the offset membrane potentials
        secax.plot(
            offset_mem.T,
            c="orange",
            alpha=0.7,
            linewidth=1.0
        )

        # Set y-limits based on the actual offset range
        buffer = 0.5
        y_min = offset_mem.min() - buffer  # Slight buffer below first neuron
        y_max = offset_mem.max() + buffer  # Slight buffer above last neuron
        secax.set_ylim(y_min, y_max)

        # Label the y-axis with neuron indices (optional)
        secax.set_ylabel("Neuron Membrane Potential (offset)")
        secax.set_yticks(np.arange(num_neurons) * offset + offset / 2)
        secax.set_yticklabels([f"Neuron {i}" for i in range(num_neurons)])

        # Optional: add horizontal lines for threshold/reset if needed
        # e.g., secax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

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
        :param data: A dictionary containing the necessary data. 
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

        cmap = matplotlib.colormaps["viridis"]
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
        axes.set_xlim(0, data["time_steps"])
        axes.set_title('Instantaneous firing rates')
        axes.set_xlabel('Time (ms)')
        if self.ylabel:
            axes.set_ylabel('Neuron')

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
        :param data: A dictionary containing the necessary data. 
                Required Keys: isis
        :type data: dict

        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """

        data_clean = torch.nested.to_padded_tensor(
            data["isis"],
            padding = 0
        ).to(torch.float)

        if data_clean.numel() == 0:
            return axes
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
        axes.set_yscale("log")
        axes.set_title("Histogram of ISIs")
        axes.set_xlabel("Bins")
        if self.ylabel:
            axes.set_ylabel("Counts")

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
        :param data: A dictionary containing the necessary data. 
                Required Keys: smoothed_rates
        :type data: dict

        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """
        # data shape: [neurons, times_steps]
        X = data["smoothed_rates"].cpu().numpy()

        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(X)

        # # Create a label-to-color mapping
        # labels = np.array([f"Neuron_{i:03d}" for i in range(X.shape[0])])
        # unique = np.unique(labels)
        # label_to_idx = {label: idx for idx, label in enumerate(unique)}
        # cvec = np.array([label_to_idx[label] for label in labels])

        # # Use a colormap to assign colors
        # # this is fucked up
        # cmap = cm.viridis  # or any other colormap: 'plasma', 'inferno', 'Set1', etc.
        # norm = plt.Normalize(vmin=0, vmax=len(unique) - 1)
        # colors = cmap(norm(cvec))

        axes.scatter(
            X_pca[:, 0], 
            X_pca[:, 1],
            # c = colors
        )

        axes.set_title("Neural population trajectory")
        axes.set_xlabel("PC1 (Explained Var:\n"
                        f"X:{pca.explained_variance_[0] * 100}%, "
                        f"Y:{pca.explained_variance_[1] * 100})")
        if self.ylabel:
            axes.set_ylabel("PC2")

        return axes

    def plot_rsync(
        self,
        save: bool = True,
        name_ext: str | None = None,
    ) -> Figure:

        # pre-execution checks
        if not self._tendencies:
            raise UnboundLocalError(
                "Tendencies have not been calculated before. " \
                "Please call measure_tendencies() before you call this function."
            )

        if save:
            os.makedirs(
                self.img_path,
                exist_ok = True
            )
            os.environ["MPLBACKEND"] = "pdf"
        if name_ext is None:
            name_ext = self.id

        # init empty lists
        rsyncs = []
        cls = []

        # and create lists that extract rsync and classes 
        for sample in self._tendencies.values():
            rsyncs.append([
                neuron["rsync"].item()
                for neuron in sample["measurements"].values()
            ])
            cls.append(sample["class"].item())

        rsyncs = np.array(rsyncs)
        cls = np.array(cls)
        unique_classes = np.unique(cls)

        fig = plt.figure(
            num = "rsync",
            figsize = (2.5 * unique_classes.size, 5),
            dpi = 300,
            clear = True
        )
        axes = fig.subplot_mosaic(
            mosaic = [
                [f"class{i}" for i in unique_classes],
                ["legend"    for _ in unique_classes]
            ],
            width_ratios = [1 for _ in unique_classes],
            height_ratios = (13,1)
        )

        vmin = rsyncs.min()
        vmax = rsyncs.max()

        for c in unique_classes:
            axes[f"class{c}"] = sns.heatmap(
                rsyncs[cls == c],
                annot = True,
                cmap = "viridis",
                vmin = vmin,
                vmax = vmax,
                ax = axes[f"class{c}"],
                cbar = True,
                cbar_ax = axes["legend"],
                cbar_kws = {"location": "bottom"}
            )
            axes[f"class{c}"].set_title(f"Class {c}")
            axes[f"class{c}"].set_xlabel("Layers")
            axes[f"class{c}"].set_ylabel("Samples")

        fig.tight_layout()
        fig.suptitle("RSync per Class and Layer")

        if save:
            fig.savefig(
                os.path.join(
                    self.img_path,
                    f"rsyncs-{name_ext}.pdf"
                ),
                format = "pdf"
            )
            fig.clear()

        return fig

    def save_metrics(
        self,
        loss: list | None,
        acc: list | None,
        name_ext: str | None = None
    ) -> None:

        if name_ext is None:
            name_ext = self.id

        self.metrics = {}
        if loss:
            self.metrics["loss"] = loss
        if acc:
            self.metrics["acc"] = acc

        # and pickle the measurements
        with open(
            os.path.join(self.bin_path, f"metrics-{name_ext}.pkl"),
            "wb+"
        ) as file:
            pickle.dump(
                self.metrics,
                file
            )