from imports import (
    DEFAULT_CONFIG_PATH,
    DEVICE,
    NOW,
    PCA,
    ROOT,
    Axes,
    AxesImage,
    Figure,
    Literal,
    MaxNLocator,
    Path,
    # ScalarMappable,
    cm,
    colors,
    functional,
    matplotlib,
    os,
    pickle,
    plt,
    shutil,
    signal,
    torch,
    tqdm,
    yaml,
)
from imports import numpy as np
from imports import (
    pandas as pd,
)
from imports import seaborn as sns
from imports import snntorch as snn
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

    if Path(identifier).suffix == "":
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

    if Path(identifier).suffix == "":
        identifier = identifier + ".pt"

    if data_path is None:
        data_path = os.path.join(
            ROOT,
            "data",
            identifier,
        )
    data_path = os.path.join(data_path, "model")


    # try loading config that"s probably been saved alongside the model:
    if config_path is None:
        try:
            cfg_name = Path(identifier).stem + ".yml"
            _, cfg = load_config(os.path.join(
                data_path,
                "synthmodel-" + cfg_name
            ))
        except FileNotFoundError:
            # if that didn"t work, fall back to default choice
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
        # recorder: functional.probe.OutputMonitor,
        model: SynthModel,
        time_steps: int = 1000,
        dt_ms: float = 1.0,
        rate_tau_ms: float = 20.0,
        rsync_tau_ms: float = 3.0,
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

        self.recorder = functional.probe.OutputMonitor(
            model, snn.Leaky
        )
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
        self.dt_ms = dt_ms
        self.rate_tau_ms = rate_tau_ms
        self.rsync_tau_ms = rsync_tau_ms
        self._tendencies = {}
        self.y_title = False

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
        layer: int = -1
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
        # new output monitor to make sure we"re  not accidentally deleting data from another one
        # rec = functional.probe.OutputMonitor(model, snn.Leaky)
        # rec.enable()
        self.recorder.clear_recorded_data()
        self.recorder.enable()
        layer_key = self.recorder.monitored_layers[layer]


        outputs = []
        labels = []
        time_steps = 0
        with torch.no_grad():
            for x, label in tqdm.tqdm(
                data,
                total = len(data),
                desc = "Collecting Model Responses"
            ): # add tqdm
                x = x.to(DEVICE)
                time_steps = x.shape[0]
                model(x, batch_first = False)
                labels.append(label)                    # [batch_size]

        # gets list[tuple(t0)[spk, mem], tuple(t1)[spk, mem], ...]
        recordings = self.recorder[layer_key]                # [T*B, B, N]
        # filters for spikes; list[spk(t0), spk(t1), ...]
        recordings = [x for x, _ in recordings]         # [T*B, B, N]
        # since recordings is now over the whole dataset, it needs splitting
        # each sample should have only time_steps time steps
        # the whole recordings list should be no_batches * time_steps
        assert len(recordings) == len(data) * time_steps
        chunks = []
        for n in range(len(data)):
            chunk = recordings[
                n * time_steps : (n+1) * time_steps     # [T, B, N]
            ]
            chunks.append(torch.stack(chunk))
        # chunks should be now [len(data)] with items of [T, B, N]

        # make them to giant tensors
        outputs = torch.cat(chunks, dim = 1)            # [T, all_samples, N]
        outputs = outputs.permute(1, 0, 2)              # [all_samples, T, N]
        labels  = torch.cat(labels)                     # [all_samples]

        self.raw_outputs = outputs
        self.raw_labels = labels
        return outputs, labels

    def get_output_repr(
        self,
        repr: Literal["count", "smoothed"],  # noqa: F821
        window_width: int = 20,
        step: int = 5,
        filtered = True
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

        if repr == "count":
            for start in starts:
                out = self.raw_outputs[
                    :,
                    start: start + window_width,
                    :
                ]
                outputs.append(out.sum(dim = 1)) # sum window over time

        elif repr == "smoothed":
            rates = self.measure_rate(
                self.raw_outputs,
                dt_ms = 1,
                tau_ms = 5,
            )

            if filtered:
                max_rates = rates.max() / 2
                # filter out neurons with rate > 150
                mask = (rates < max_rates).any(dim=1)
                rates = rates[mask] # this "deletes" neurons

            for start in starts:
                out = rates[
                    :,
                    start: start + window_width,
                    :
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

        # assuming only one batch, since that"s the cleanest way
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
    ) -> list[torch.Tensor]:
        """
        Function to calculate the time-difference between the spikes of each neuron.
        Since not every neuron spikes the same amount of times, the returning tensor is jagged in the second dimension.

        :param spikes: The spike-train of a layer.
        :type spikes: torch.Tensor
        :returns: A Tensor with the same first dim, but jagged in the second.
        :rtype: torch.Tensor
        """
        if spikes.device != "cpu":
            spikes = spikes.to(torch.device("cpu"))

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

        # out = torch.nested.nested_tensor(isis, layout = torch.jagged)
        return isis

    def measure_rate(
        self,
        spikes: torch.Tensor,
        dt_ms: float = 1.,
        tau_ms: float = 20.,
    ) -> torch.Tensor:
        """
        Estimate instantaneous firing rate from spike trains using
        exponential kernel smoothing.

        :param spikes:  Binary spike tensor of shape [neurons, time_steps].
                    Values should be 0/1 (or spike counts per bin).
        :type spikes: torch.Tensor
        :param dt: Time step size in milliseconds.
        :type dt: float
        :param tau: Exponential kernel time constant in milliseconds.
        :type tau: float

        :returns: Smoothed firing rates in Hz.
                    Shape: [neurons, time_steps]
        :rtype: torch.Tensor
        """

        if dt_ms is None:
            dt_ms = self.dt_ms

        if tau_ms is None:
            tau_ms = self.rate_tau_ms

        dt_s = dt_ms / 1000.0
        tau_s = tau_ms / 1000.0

        # kernel length: ~5 tau captures most of exponential decay
        kernel_length = max(
            1,
            int(np.ceil(5 * tau_s / dt_s))
        )

        t = torch.arange(
            kernel_length,
            device = spikes.device,
            dtype = spikes.dtype
        ) * dt_s
        t = t.flip(0)

        # causal exponential kernel
        kernel = torch.exp(-t / tau_s)

        # normalize kernel so output is in spikes/sec (Hz)
        kernel = kernel / kernel.sum() / dt_s

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
        layer_titles: list | None = None,
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

        if layer_titles is None:
            layer_titles = ["Input", "Hidden", "Output"]

        FIG_SIZE = (18, 15) # A2 papersize
        DPI = 300
        NROWS = 4
        HEIGHT_RATIOS = [1, 1, 0.5, 1.35]

        fig = plt.figure(
            num = "tendency-vis",
            clear = True,
            figsize = FIG_SIZE,
            layout = "constrained",
            dpi = DPI
        )

        for key in tqdm.tqdm(
            self._tendencies,
            total = len(self._tendencies),
            desc = "Plotting Spike Analysis"
        ):
            # the sample analysed in the thesis used these functions
            # to create the subplots.
            if key == "sample-12":
                # measurements for this sample
                measurements = self._tendencies[key]["measurements"]
                # call the new per-layer PDF creator
                self._plot_layerwise_pdf(
                    measurements=measurements,
                    name_ext=name_ext,
                    sample_key=key,
                )
                self._plot_hidden_raster_and_rate(
                    measurements=measurements,
                    name_ext=name_ext,
                    sample_key=key,
                    hidden_layer_name="neurons.0",   # adjust if your hidden layer has a different key
                )

            measurements = self._tendencies[key]["measurements"]
            num_layers = len(measurements)

            if save and not blocking:
                axes = fig.subplots(
                    nrows = NROWS,
                    ncols = num_layers,  # layers as cols
                    squeeze = False,
                    gridspec_kw = {
                        "height_ratios": HEIGHT_RATIOS,
                        "hspace": 0.,
                        "wspace": 0.
                    }
                )
            else:
                fig, axes = plt.subplots(
                    nrows = NROWS,
                    ncols = num_layers,  # layers as cols
                    squeeze = False,
                    figsize = FIG_SIZE,
                    dpi = DPI,
                    gridspec_kw = {
                        "height_ratios": HEIGHT_RATIOS,
                        "hspace": 0.12,
                        "wspace": 0.08
                    }
                )

            # -------------------------------------------------
            # Column labels
            # -------------------------------------------------
            # make sure layer titles are correct
            if num_layers != len(layer_titles):
                layer_titles = [
                    f"Layer {i}"
                    for i in range(num_layers)
                ]
            # and set them
            for i, title in enumerate(layer_titles):
                axes[0, i].set_title(
                    title,
                    fontsize = 13,
                    fontweight = "bold",
                    pad = 8,
                )

            # -------------------------------------------------
            # Shared firing-rate normalization
            # -------------------------------------------------

            # all_rates = np.concatenate([
            #     data["smoothed_rates"]
            #         .cpu()
            #         .numpy()
            #         .ravel()
            #     for data in measurements.values()
            # ])

            # rate_norm = colors.Normalize(
            #     vmin = 0,
            #     vmax = np.max(all_rates),
            # )
            # -------------------------------------------------
            # Shared time normalization
            # -------------------------------------------------
            time_norm = colors.Normalize(
                vmin=0,
                vmax=self.time_steps * self.dt_ms,
            )

            # -------------------------------------------------
            # Plot each layer
            # -------------------------------------------------
            images = []
            all_stats = []

            for i, (layer, data) in enumerate(
                measurements.items()
            ):
                show_ylabel = i == 0

                # plot spikes
                self._plot_spikes(
                    axes = axes[0, i],
                    data = data,
                    title = layer_titles[i],
                    show_ylabel = show_ylabel
                )

                # and membrane if it's the last layer
                if i == num_layers - 1:
                    self._plot_membrane(
                        axes = axes[0, i],
                        data = data
                    )

                # then the rate heatmap
                im = self._plot_rate_heatmap(
                    fig = fig,
                    axes = axes[1, i],
                    data = data,
                    show_ylabel = show_ylabel
                )
                images.append(im)

                # plot the isi histogram
                self._plot_isis(
                    axes = axes[2, i],
                    data = data,
                    show_ylabel = show_ylabel
                )

                # and finally pca plots
                _, stats = self._plot_pca_trajectory(
                    axes = axes[3, i],
                    data = data,
                    time_norm = time_norm
                )

                # add layer key to stats and append
                stats["layer"] = layer
                all_stats.append(stats)

            # -------------------------------------------------
            # Row labels
            # -------------------------------------------------

            row_titles = [
                "Spikes",
                "Instantaneous Firing Rate",
                "Inter-Spike Interval",
                "Population Trajectory",
            ]

            for row, title in enumerate(row_titles):
                axes[row, 0].annotate(
                    title,
                    xy = (-0.15, 0.5),
                    xycoords = "axes fraction",
                    rotation = 90,
                    ha = "center",
                    va = "center",
                    fontsize = 11,
                    fontweight = "bold",
                )

            # Figure Title
            cls = self._tendencies[key]["class"].item()
            fig.suptitle(
                f"Network activity — Class {cls}",
                fontsize=16,
                fontweight="bold",
            )

            # -------------------------------------------------
            # Shared firing-rate colorbar
            # -------------------------------------------------

            # fig.colorbar(
            #     images[0],
            #     ax = axes[1, :],
            #     location = "bottom",
            #     orientation = "horizontal",
            #     label = "Instantaneous firing rate (Hz)",
            #     fraction = 0.05,
            #     pad = 0.08,
            # )

            # -------------------------------------------------
            # Shared PCA time colorbar
            # -------------------------------------------------

            sm = cm.ScalarMappable(
                norm = time_norm,
                cmap = "viridis",
            )

            fig.colorbar(
                sm,
                ax = axes[3, :],
                location = "right",
                label = "Time (ms)",
                shrink = 0.8,
                pad = 0
            )

            # -------------------------------------------------
            # Save statistics
            # -------------------------------------------------

            df_stats = pd.DataFrame(all_stats)

            if save:
                # save figure
                fig.savefig(
                    os.path.join(
                        self.img_path,
                        f"tendencies-{name_ext}-{key}.pdf"
                    ),
                    format = "pdf",
                    bbox_inches = "tight",
                )
                fig.clear()

                # and pca csv
                df_stats.to_csv(
                    os.path.join(
                        self.img_path,
                        f"pca-{name_ext}-{key}.csv"
                    ),
                    index = False
                )

                # print a clean table
                print("\nPCA Analysis Summary:")
                print(
                    df_stats[[
                        "layer",
                        "pc1_var",
                        "pc2_var",
                        "d_eff",
                        "n_pc_90"
                    ]].to_string(index = False)
                )
            else:
                # print a clean table
                print("\nPCA Analysis Summary:")
                print(
                    df_stats[[
                        "layer",
                        "pc1_var",
                        "pc2_var",
                        "d_eff",
                        "n_pc_90"
                    ]].to_string(index = False)
                )


        if save:
            # and pickle the measurements
            with open(
                os.path.join(
                    self.bin_path,
                    f"measurements-{name_ext}.pkl"
                ),
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
        self.plot_filtered_rsync(
            save = save,
            name_ext = name_ext
        )

        if blocking:
            plt.show()


    def _plot_spikes(
        self,
        axes: Axes,
        data: dict,
        title: str,
        show_ylabel: bool = False
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

        spikes = data["spikes"].detach().cpu().numpy()

        spike_times = [
            np.flatnonzero(spikes[neuron])
            for neuron in range(spikes.shape[0])
        ]

        axes.eventplot(
            spike_times,
            orientation = "horizontal",
            lineoffsets = np.arange(spikes.shape[0]),
            linelengths = 0.1 if data["neurons"] == 2 else 0.7,
            linewidths = 0.8 if spikes.shape[0] > 50 else 1.0,
            colors = "black",
            rasterized = True,
        )

        # y axis tick setup
        axes.yaxis.set_major_locator(MaxNLocator(
            nbins = min(10, data["neurons"]),
            integer = True
        ))
        axes.set_ylim(-0.5, data["neurons"] - 0.5)
        axes.set_xlim(0, data["time_steps"])

        # labels
        axes.set_title(
            title,
            fontsize = 12,
            fontweight = "bold"
        )
        if show_ylabel:
            axes.set_ylabel("Neurons")
        else:
            axes.set_ylabel("")

        axes.set_xlabel("Time (ms)")
        axes.grid(
            True,
            axis = "x",
            alpha = 0.2,
            linewidth = 0.7
        )

        return axes

    def _plot_membrane(
        self,
        axes: Axes,
        data: dict,
        offset: float = 1.0,
        threshold: float | None = None,
        baseline: float | None = None
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

        mem = data["membrane"].cpu().numpy().astype(float)

        # Define offset: one unit apart (e.g., 1.0) for clarity
        num_neurons = mem.shape[0]


        # create offset membrane potentials
        # we'll add an offset per neuron: neuron * offset
        # but we"ll keep the original voltage values intact
        neuron_offsets = np.arange(num_neurons) * offset
        offset_mem = mem + neuron_offsets[:, np.newaxis]

        # make membrane potentials look different
        cmap = matplotlib.colormaps["spring"]
        colors = cmap(np.linspace(0, 1, num_neurons))

        # create a secondary y-axis
        secax = axes.twinx()

        # plot the offset membrane potentials
        for neuron in range(num_neurons):
            secax.plot(
                np.arange(data["time_steps"]),
                offset_mem[neuron],
                color = colors[neuron],
                alpha = 0.7,
                linewidth = 0.9,
                label = f"Neuron {neuron}" if num_neurons <= 10 else None
            )

            # Optional reference lines
            if baseline is not None:
                secax.axhline(
                    y = baseline + neuron_offsets[neuron],
                    color = "blue",
                    alpha = 0.5,
                    linestyle = ":",
                    linewidth = 0.7
                )
            if threshold is not None:
                secax.axhline(
                    y = threshold + neuron_offsets[neuron],
                    color = "red",
                    alpha = 0.5,
                    linestyle = "--",
                    linewidth = 0.7
                )

        # Set y-limits based on the actual offset range
        buffer = 0.5
        y_min = offset_mem.min() - buffer  # Slight buffer below first neuron
        y_max = offset_mem.max() + buffer  # Slight buffer above last neuron
        secax.set_ylim(y_min, y_max)

        # Label the y-axis with neuron indices (optional)
        secax.set_ylabel("Neuron Membrane Potential (offset)")
        secax.set_yticks(neuron_offsets)
        secax.set_yticklabels([str(i) for i in range(num_neurons)])
        axes.grid(True, alpha = 0.3, axis = "x")

        # Optional legend
        if num_neurons <= 10:
            secax.legend(loc="upper right", fontsize=8)

        return secax


    def _plot_rate_heatmap(
        self,
        fig: Figure,
        axes: Axes,
        data: dict,
        show_ylabel: bool = False
    ) -> AxesImage:
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
            aspect = "auto",
            origin = "lower",
            cmap = cmap,
            extent = (
                0,
                data["time_steps"] * self.dt_ms,
                -0.5,
                data["neurons"] - 0.5,
            ),
            interpolation = "nearest",
            rasterized = True,
        )

        fig.colorbar(
            im,
            ax = axes,
            orientation = "horizontal",
            location = "bottom",
            fraction = 0.04,
            aspect = 40,
            label = "Firing rate (Hz)",
            # pad = 0.03,
        )

        if show_ylabel:
            axes.set_ylabel("Neurons")
        else:
            axes.set_ylabel("")

        axes.yaxis.set_major_locator(MaxNLocator(
            nbins = min(10, data["neurons"]),
            integer = True
        ))
        axes.set_xlim(0, data["time_steps"] * self.dt_ms)
        axes.set_xlabel("Time (ms)")

        return im

    def _plot_isis(
        self,
        axes: Axes,
        data: dict,
        show_ylabel: bool = False
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

        isis = [
            x.cpu().numpy()
            for x in data["isis"]
            if x.numel() > 0
        ]

        if not isis:
            axes.text(
                0.5,
                0.5,
                "No ISIs",
                ha="center",
                va="center",
                transform=axes.transAxes,
            )
            return axes

        values = np.concatenate(isis)

        # Remove invalid/non-positive values.
        values = values[
            np.isfinite(values) &
            (values >= 0)
        ]

        if values.size == 0:
            return axes

        # Use sensible bin width while keeping the number of bins manageable.
        # Explicitly include zero.
        if values.max() == 0:
            bins = np.array([
                -0.5 * self.dt_ms,
                0.5 * self.dt_ms,
            ])
        else:
            n_bins = min(
                50,
                max(10, int(np.sqrt(values.size)))
            )

            bins = np.linspace(
                0,
                values.max(),
                n_bins + 1,
            )

            # Make sure zero has its own visible bin.
            bins[0] = -0.5 * self.dt_ms

        # and plot
        axes.hist(
            values,
            bins = bins,                # type:ignore
            color = "tab:blue",
            alpha = 0.85,
            edgecolor = "none",
        )

        axes.set_yscale("log")
        axes.set_xlabel("Inter-Spike Interval (ms)")
        if show_ylabel:
            axes.set_ylabel("Count")
        else:
            axes.set_ylabel("")

        return axes

    def _plot_pca_trajectory(
        self,
        axes: Axes,
        data: dict,
        time_norm: colors.Normalize | None = None,
    ) -> tuple[Axes, dict]:
        """
        Convenience function to plot the PCA trajectory of the smoothed rates.
        Includes calculation of effective dimensionality and cumulative variance.

        :param axes: Axes to plot the PCA onto.
        :param data: A dictionary containing the necessary data. 
                Required Keys: smoothed_rates
        :type data: dict

        :returns: The Axes now containing a plot.
        :rtype: plt.Axes
        """
        # Shape: [neurons, time_steps]
        X = data["smoothed_rates"].cpu().numpy()

        # Remove linear trend along the time axis (axis 1)
        X = signal.detrend(X, axis = 1)

        # PCA to extract population dynamics
        # We want to visualize how the population state evolves over time - i.e., the "trajectory" of neural activity.
        # sklearn"s PCA expects data in the format [Samples, Features], where:
        #   - Samples: individual observations (here, time points)
        #   - Features: variables measured at each sample (here, neuron activities)
        #
        # The original data has shape [Neurons, Time], meaning:
        #   - Each row = one neuron (feature)
        #   - Each column = one time point (sample)
        #
        # To analyze population dynamics, we need to treat each **time point** as a sample,
        # and each **neuron** as a feature. This means we must transpose the data:
        #   X_detrended.T -> shape [Time, Neurons]
        #
        # Now:
        #   - Each row is a time point (sample): the population state at that moment
        #   - Each column is a neuron (feature): its activity level at that time
        #
        # PCA will now find the dominant patterns of co-variation across neurons at each time point.
        # The principal components (PCs) represent spatial population modes (e.g., global up/down states),
        # and their scores (projection of data onto PCs) represent how those modes evolve over time.
        X_T = X.T
        pca = PCA()
        X_plot = pca.fit_transform(X_T)

        # Calculate Metrics
        # Explained Variance Ratio (Percent)
        explained_var_ratio = pca.explained_variance_ratio_

        # Cumulative Variance
        cumulative_var = np.cumsum(explained_var_ratio)
        n_pc_90 = (
            np.argmax(cumulative_var >= 0.9) + 1
            if np.any(cumulative_var >= 0.9)
            else len(cumulative_var)
        )

        # Effective Dimensionality (D_eff)
        # Using the formula: (sum(eigvals))^2 / sum(eigvals^2)
        eigvals = pca.explained_variance_
        d_eff = (eigvals.sum()**2) / (eigvals**2).sum()

        stats = {
            "shape_pca": X_T.shape,
            "pc1_var": explained_var_ratio[0],
            "pc2_var": explained_var_ratio[1],
            "d_eff": d_eff,
            "n_pc_90": n_pc_90,
            "centering": True,
            "detrending": True
        }
        time = (
            np.arange(X_plot.shape[0]) *
            self.dt_ms
        )

        if time_norm is None:
            time_norm = colors.Normalize(
                vmin = time.min(),
                vmax = time.max(),
            )

        # Create a color map based on time to visualize trajectory direction
        axes.scatter(
            X_plot[:, 0],
            X_plot[:, 1],
            c = time,
            cmap = "viridis",
            alpha = 0.8,
            s = 5 # smaller points for lines
        )

        # Add a line connecting the points to emphasize trajectory
        axes.plot(
            X_plot[:, 0],
            X_plot[:, 1],
            color = "gray",
            alpha = 0.3,
            linewidth = 0.7,
            zorder = 0
        )

        # Explicit start/end markers.
        axes.scatter(
            X_plot[0, 0],
            X_plot[0, 1],
            s = 40,
            facecolor = "white",
            edgecolor = "black",
            linewidth = 1,
            zorder = 5,
            label = "Start",
        )

        axes.scatter(
            X_plot[-1, 0],
            X_plot[-1, 1],
            s = 100,
            marker = "X",
            facecolor = "black",
            edgecolor = "white",
            linewidth = 0.5,
            zorder = 5,
            label = "End",
        )

        # labels with explained var percentages
        axes.set_xlabel(f"PC1 ({explained_var_ratio[0]*100:.1f}%)")
        axes.set_ylabel(f"PC2 ({explained_var_ratio[1]*100:.1f}%)")
        axes.legend()

        # Add colorbar for time
        return axes, stats
 
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
            layout = "constrained",
            clear = True
        )
        axes = fig.subplot_mosaic(
            mosaic = [
                [f"class{i}" for i in unique_classes],
                ["legend"    for _ in unique_classes]
            ],
            # squeeze = False,
            width_ratios = [1 for _ in unique_classes],
            height_ratios = (1,0.01),
        )


        norm = colors.Normalize(
            vmin = rsyncs.min(),
            vmax = rsyncs.max(),
        )
        cmap = matplotlib.colormaps["viridis"]

        for c in unique_classes:
            axes[f"class{c}"] = sns.heatmap(
                rsyncs[cls == c],
                ax = axes[f"class{c}"],
                norm = norm,
                cmap = cmap,
                annot = True,
                cbar = True,
                cbar_ax = axes["legend"],
                cbar_kws = {"location": "bottom"}
            )
            axes[f"class{c}"].set_title(f"Class {c}")
            axes[f"class{c}"].set_xlabel("Layers")
            axes[f"class{c}"].set_ylabel("Samples")

        fig.suptitle(
            "Population synchrony across samples",
            fontsize=14,
            fontweight="bold",
        )

        if save:
            fig.savefig(
                os.path.join(
                    self.img_path,
                    f"rsyncs-{name_ext}.pdf"
                ),
                format = "pdf",
                bbox_inches = "tight",
            )
            plt.close(fig)

        return fig

    def plot_filtered_rsync(
        self,
        save: bool = True,
        name_ext: str | None = None,
    ):

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
        cls = []

        # and create lists that extract rsync and classes 
        for sample in self._tendencies.values():
            cls.append(sample["class"].item())

        cls = np.array(cls)
        unique_classes = np.unique(cls)

        fig = plt.figure(
            num = "rsync-filtered",
            figsize = (2.5 * unique_classes.size, 5),
            dpi = 300,
            layout = "constrained",
            clear = True
        )
        axes = fig.subplot_mosaic(
            mosaic = [
                [f"class{i}" for i in unique_classes],
                ["legend"    for _ in unique_classes]
            ],
            # squeeze = False,
            width_ratios = [1 for _ in unique_classes],
            height_ratios = (1,0.01),
        )

        # MAX_RATES = 150
        filtered_rsyncs = []
        for sample in self._tendencies.values():
            sample_filtered_rsyncs = []
            for layer, data in sample["measurements"].items():
                spikes = data["spikes"] # shape: [neurons, time_steps]
                rates = data["smoothed_rates"]
                max_rates = rates.max() / 2
                # filter out neurons with rate > 150
                if layer == "neurons.0":
                    # values = rates.mean(dim=1)
                    # print(values)
                    # mask = values < 100
                    mask = (rates < max_rates).any(dim=1)
                    print(mask.sum())
                    spikes = spikes[mask] # this "deletes" neurons, but we don't want that?
                    # rates = rates.masked_fill(~mask.unsqueeze(1), 0) # doesn't change the result
                rsync = self.measure_rsync(spikes)
                sample_filtered_rsyncs.append(rsync.cpu().numpy())
            filtered_rsyncs.append(sample_filtered_rsyncs)

        rsyncs = np.array(filtered_rsyncs)

        norm = colors.Normalize(
            vmin = rsyncs.min(),
            vmax = rsyncs.max(),
        )
        cmap = matplotlib.colormaps["viridis"]

        for c in unique_classes:
            axes[f"class{c}"] = sns.heatmap(
                rsyncs[cls == c],
                ax = axes[f"class{c}"],
                norm = norm,
                cmap = cmap,
                annot = True,
                cbar = True,
                cbar_ax = axes["legend"],
                cbar_kws = {"location": "bottom"}
            )
            axes[f"class{c}"].set_title(f"Class {c}")
            axes[f"class{c}"].set_xlabel("Layers")
            axes[f"class{c}"].set_ylabel("Samples")

        fig.suptitle(
            "Population synchrony across samples",
            fontsize=14,
            fontweight="bold",
        )

        if save:
            fig.savefig(
                os.path.join(
                    self.img_path,
                    f"filtered-rsyncs-{name_ext}.pdf"
                ),
                format = "pdf",
                bbox_inches = "tight",
            )
            plt.close(fig)

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

    # the following two functions are purely vibe-coded

    def _plot_layerwise_pdf(
        self,
        measurements: dict,
        name_ext: str,
        sample_key: str,
    ) -> None:
        """
        Create three PDFs (one per layer) that contain only:
            * instantaneous firing-rate heat-map,
            * ISI histogram,
            * PCA trajectory (time-coloured).

        The function is meant to be called from ``visualise_tendencies`` when the
        current sample is ``sample-12``.  All figures are saved in ``self.img_path``.
        The layout is **vertical** (rate → ISI → PCA) and the three PDFs are named

            tendencies-{name_ext}-{sample_key}-{layer}.pdf

        Parameters
        ----------
        measurements : dict
            ``self._tendencies[sample_key]["measurements"]`` - a mapping
            ``layer_name → layer_dict``.
        name_ext : str
            Identifier that is appended to the file name (usually the run-id).
        sample_key : str
            The key of the sample being plotted (e.g. ``"sample-12"``).
        """
        # ------------------------------------------------------------------
        # 1️⃣  Global normalisations (shared across all layers)
        # ------------------------------------------------------------------
        #   - rate colour scale (max over *all* layers)
        # all_rates = np.concatenate([
        #     data["smoothed_rates"].cpu().numpy().ravel()
        #     for data in measurements.values()
        # ])
        # rate_norm = colors.Normalize(vmin=0, vmax=np.max(all_rates))

        #   - time normalisation for the PCA colour-bar
        time_norm = colors.Normalize(
            vmin=0,
            vmax=self.time_steps * self.dt_ms,
        )

        # ------------------------------------------------------------------
        # 2️⃣  Loop over layers → one figure per layer
        # ------------------------------------------------------------------
        for layer_name, layer_data in measurements.items():
            # --------------------------------------------------------------
            #   Figure set-up (3 rows, 1 column)
            # --------------------------------------------------------------
            FIG_SIZE = (6, 11)          # a comfortable portrait size
            DPI      = 300
            NROWS    = 3
            HEIGHT_RATIOS = [1.2, 0.4, 1]   # rate, ISI, PCA

            fig, axes = plt.subplots(
                nrows=NROWS,
                ncols=1,
                figsize=FIG_SIZE,
                dpi=DPI,
                gridspec_kw={
                    "height_ratios": HEIGHT_RATIOS,
                    # hspace=0.25,
                    # wspace=0.0,
                },
                # constrained_layout=True,
            )
            # ``axes`` is a 1-D array because we have a single column
            axes = np.atleast_1d(axes)

            # --------------------------------------------------------------
            #   2️⃣  Rate heat-map (row 0)
            # --------------------------------------------------------------
            self._plot_rate_heatmap(
                fig=fig,
                axes=axes[0],
                data=layer_data,
                show_ylabel=True,
            )
            # enforce the shared colour-scale
            # im.set_norm(rate_norm)

            # --------------------------------------------------------------
            #   3️⃣  ISI histogram (row 1)
            # --------------------------------------------------------------
            self._plot_isis(
                axes=axes[1],
                data=layer_data,
                show_ylabel=True,
            )

            # --------------------------------------------------------------
            #   4️⃣  PCA trajectory (row 2)
            # --------------------------------------------------------------
            self._plot_pca_trajectory(
                axes=axes[2],
                data=layer_data,
                time_norm=time_norm,
            )
            # add a colour-bar for the time axis (right side of the PCA panel)
            sm = cm.ScalarMappable(norm=time_norm, cmap="viridis")
            fig.colorbar(
                sm,
                ax=axes[2],
                location="right",
                label="Time (ms)",
                shrink=0.8,
                pad=0.02,
            )

            # --------------------------------------------------------------
            #   5️⃣  Global titles / labels
            # --------------------------------------------------------------
            layer_map = {
                "neurons.-1": "Input",
                "neurons.0": "Hidden",
                "neurons.1": "Output"
            }

            fig.suptitle(
                f"{layer_map[layer_name]} Layer - Sample {sample_key.split('-')[-1]} - "
                f"Class {self._tendencies[sample_key]['class'].item()}",
                fontsize=14,
                fontweight="bold",
            )
            axes[0].set_title("Instantaneous Firing Rate", fontsize=12)
            axes[1].set_title("Inter-Spike-Interval Histogram", fontsize=12)
            axes[2].set_title("Population Trajectory (PCA)", fontsize=12)

            # --------------------------------------------------------------
            #   6️⃣  Save the PDF
            # --------------------------------------------------------------
            out_path = os.path.join(
                self.img_path,
                f"tendencies-{name_ext}-{sample_key}-{layer_name}.pdf",
            )
            fig.tight_layout(pad=1.2, h_pad=0.8, w_pad=0.8)
            fig.savefig(out_path, format="pdf", bbox_inches="tight")
            plt.close(fig)   # free memory

                # ----------------------------------------------------------------------
    # NEW HELPER: raster + heat-map of the hidden layer (side-by-side)
    # ----------------------------------------------------------------------
    def _plot_hidden_raster_and_rate(
        self,
        measurements: dict,
        name_ext: str,
        sample_key: str,
        hidden_layer_name: str = "neurons.0",   # default name used in the repo
    ) -> None:
        """
        Produce a **single PDF** that contains

            ┌───────────────┬───────────────────────┐
            │  Spike raster │  Instantaneous rate    │
            │   (layer)     │   heat-map (layer)     │
            └───────────────┴───────────────────────┘

        The raster is plotted with the existing ``_plot_spikes`` routine,
        the heat-map with ``_plot_rate_heatmap``.  Both panels share the same
        time-axis (ms) and are saved as

            tendencies-{name_ext}-{sample_key}-hidden-raster-heat.pdf

        Parameters
        ----------
        measurements : dict
            ``self._tendencies[sample_key]["measurements"]`` - a mapping
            ``layer_name → layer_dict``.
        name_ext : str
            Identifier that is appended to the file name (usually the run-id).
        sample_key : str
            The key of the sample being plotted (e.g. ``"sample-12"``).
        hidden_layer_name : str, optional
            The exact key that identifies the hidden layer in ``measurements``.
            The default matches the naming used in the original code.
        """
        # ------------------------------------------------------------------
        # 1️⃣  Grab the hidden-layer data
        # ------------------------------------------------------------------
        if hidden_layer_name not in measurements:
            raise KeyError(
                f"Hidden layer '{hidden_layer_name}' not found in measurements. "
                f"Available layers: {list(measurements)}"
            )
        hidden_data = measurements[hidden_layer_name]

        # ------------------------------------------------------------------
        # 2️⃣  Figure set-up - 1 row, 2 columns
        # ------------------------------------------------------------------
        FIG_SIZE = (12,4)          # wide enough for side-by-side view
        DPI      = 300
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=FIG_SIZE,
            dpi=DPI,
            gridspec_kw={
                "wspace": 0.15, "hspace":0.0
            },
            # constrained_layout=True,
        )
        # axes[0] → raster, axes[1] → heat-map
        # ------------------------------------------------------------------
        # 3️⃣  Plot the raster (left)
        # ------------------------------------------------------------------
        layer_map = {
            "neurons.-1": "Input",
            "neurons.0": "Hidden",
            "neurons.1": "Output"
        }
        self._plot_spikes(
            axes=axes[0],
            data=hidden_data,
            title=f"{layer_map[hidden_layer_name]} - Spike raster",
            show_ylabel=True,
        )
        # remove the x-label on the raster (we’ll put it on the heat-map)
        # axes[0].set_xlabel("")
        axes[0].set_title(f"{layer_map[hidden_layer_name]} - Spike raster", fontsize=12)


        # ------------------------------------------------------------------
        # 4️⃣  Plot the rate heat-map (right)
        # ------------------------------------------------------------------
        self._plot_rate_heatmap(
            fig=fig,
            axes=axes[1],
            data=hidden_data,
            show_ylabel=True,
        )
        # give the heat-map its own title
        axes[1].set_title(f"{layer_map[hidden_layer_name]} - Instantaneous rate", fontsize=12)
        axes[1].set_ylabel("")

        # ------------------------------------------------------------------
        # 5️⃣  Global figure title & colour-bar handling
        # ------------------------------------------------------------------
        class_id = self._tendencies[sample_key]["class"].item()
        fig.suptitle(
            f"Sample {sample_key.split('-')[-1]} - Class {class_id} - Hidden layer overview",
            fontsize=14,
            fontweight="bold",
        )
        # The heat-map already adds its own colour-bar (bottom).  No extra work needed.

        # ------------------------------------------------------------------
        # 6️⃣  Save the PDF
        # ------------------------------------------------------------------
        out_path = os.path.join(
            self.img_path,
            f"tendencies-{name_ext}-{sample_key}-hidden-raster-heat.pdf",
        )
        fig.tight_layout()
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)   # free memory
    # ----------------------------------------------------------------------
    # END OF NEW HELPER
    # ----------------------------------------------------------------------