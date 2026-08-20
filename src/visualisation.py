from imports import MaxNLocator, Path, argparse, os, pickle, plt, torch
from imports import numpy as np
from imports import snntorch as snn
from imports import spikeplot as splt


def plot_lif_voltage() -> None:
    """
    Adapted from https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html
    """
    neuron = snn.Leaky(0.9)

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10, 1), torch.ones(50, 1)*0.15, torch.zeros(20, 1)), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec = [mem]
    spk_rec = [spk_out]

    # Simulation run across 100 time steps.
    for step in range(cur_in.shape[0]):
        spk_out, mem = neuron(cur_in[step], mem)
        mem_rec.append(mem)
        spk_rec.append(spk_out)

    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)


    # def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max2=1.25):
    # Generate Plots
    fig, ax = plt.subplots(
        4,
        figsize = (10,8),
        sharex = False,
        gridspec_kw = {"height_ratios": [1, 1, 0.4, 0.7]}
    )

    # Plot input current
    ax[0].plot(cur_in, c = "tab:orange")
    ax[0].set_ylim([0, 0.4])
    ax[0].set_xlim([0, cur_in.shape[0]])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    ax[0].set_title("Leaky Integrate-and-Fire Neuron with step input")

    # Plot membrane potential
    ax[1].plot(mem_rec)
    ax[1].set_ylim([0, 1.3]) 
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[1].axhline(
        y = 1, 
        alpha = 0.25, 
        linestyle = "dashed", 
        c = "black", 
        linewidth = 2
    )
    ax[1].sharex(ax[0])

    # Plot output spike using spikeplot
    splt.raster(
        spk_rec,
        ax[2],
        s = 400,
        c = "black",
        marker = "|"
    )
    ax[2].sharex(ax[0])

    ax[2].set_xlabel("Time step")
    ax[2].set_ylabel("Output spikes")
    ax[2].set_yticks([]) 

    # ------------------------------------------------------------------
    # Zoomed spike train with binary representation
    # ------------------------------------------------------------------
    start, end = 20, 40

    zoom_spikes = spk_rec[start:end].squeeze().int().numpy()
    times = np.arange(start, end, dtype = np.int32)

    # Show spikes
    ax[3].vlines(
        times[zoom_spikes == 1],
        0,
        1,
        color="black",
        linewidth=2,
    )

    ax[3].set_xlim(start - 0.5, end - 0.5)
    ax[3].set_ylim(-0.8, 1.2)
    ax[3].set_yticks([])
    ax[3].set_xlabel("Time step")
    ax[3].set_title(f"Spike train ({start}–{end}) Array representation")

    # Binary values underneath each timestep
    for t, val in zip(times, zoom_spikes):
        ax[3].text(
            t,
            -0.35,
            str(val),
            ha = "center",
            va = "center",
            fontsize = 11,
            family = "monospace",
        )
    ax[3].xaxis.set_major_locator(
        MaxNLocator(integer = True)
    )

    fig.tight_layout()
    plt.savefig("./img/lif-neuron.pdf", format = "pdf")
    plt.show()


def plot_info_data(
    *paths: str
):
    """
    Loads pickle files from three different paths and plots the data.

    Each pickle file is expected to be at: <path>/estim/info.pkl
    The pickle file should contain a tuple[dict, Tensor]: (all_info, centres).
    all_info should have the key "information"

    :param pathX: Path towards the three data directories
    :type pathX: str, required
    """
    if not paths:
        raise ValueError("At least one path must be provided.")

    # Define default colors and labels (will be extended if needed)
    default_colors = ["b", "g", "r", "c", "m", "y", "k", "tab:blue", "tab:orange", "tab:green"]
    labels = [Path(path).name for path in paths]

    # Use as many colors and labels as needed
    colors = default_colors[:len(paths)]

    fig, ax = plt.subplots(
        1,
        figsize = (10,6),
    )

    for i, base_path in enumerate(paths):
        # Construct the full file path
        file_path = os.path.join(base_path, "estim", "info.pkl")

        try:
            # Load the pickle file
            with open(file_path, "rb") as f:
                all_info, centres = pickle.load(f)

            # Ensure data is on CPU and converted to numpy for plotting
            if isinstance(all_info, torch.Tensor):
                y_vals = all_info.cpu().numpy()
            else:
                y_vals = all_info

            if isinstance(centres, torch.Tensor):
                x_vals = centres.cpu().numpy()
            else:
                x_vals = centres

            # because I"m stupid, I have to unpack the mofos in y_vals
            y_vals = [dic["information"] for dic in y_vals]

            # Plot the data
            ax.plot(
                x_vals,
                y_vals,
                color = colors[i],
                label = labels[i],
                marker = "o",
                # s = 2,
                linestyle = "-"
            )

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")


    fig.suptitle("Comparison of Information over Surrogate Gradients")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Information (Bits)")
    ax.legend()
    ax.grid(True)
    plt.savefig("./img/info-surrogates.pdf", format = "pdf")


def plot_info_decoder_loss(
    *paths: str
):
    """
    Loads pickle files from three different paths and plots the data.

    Each pickle file is expected to be at: <path>/estim/info.pkl
    The pickle file should contain a tuple[dict, Tensor]: (all_info, centres).
    all_info should have the key "information"

    :param pathX: Path towards the three data directories
    :type pathX: str, required
    """
    if not paths:
        raise ValueError("At least one path must be provided.")

    # Define default colors and labels (will be extended if needed)
    default_colors = ["b", "g", "r", "c", "m", "y", "k", "tab:blue", "tab:orange", "tab:green"]

    # Use as many colors and labels as needed
    colors = default_colors[:len(paths)]
    labels = [Path(path).name for path in paths]

    fig, ax = plt.subplots(
        1,
        figsize = (10,6),
    )

    for i, base_path in enumerate(paths):
        # Construct the full file path
        file_path = os.path.join(base_path, "estim", "info.pkl")

        try:
            # Load the pickle file
            with open(file_path, "rb") as f:
                all_info, centres = pickle.load(f)

            # Ensure data is on CPU and converted to numpy for plotting
            if isinstance(all_info, torch.Tensor):
                y_vals = all_info.cpu().numpy()
            else:
                y_vals = all_info

            if isinstance(centres, torch.Tensor):
                x_vals = centres.cpu().numpy()
            else:
                x_vals = centres

            # because I"m stupid, I have to unpack the mofos in y_vals
            y_vals = [dic["decoder_accuracy"] for dic in y_vals]

            # Plot the data
            ax.plot(
                x_vals,
                y_vals,
                color = colors[i],
                label = labels[i],
                marker = "o",
                # s = 2,
                linestyle = "-"
            )
            print(f"MaxVal for data: {np.max(y_vals)}")

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")

    fig.suptitle("Comparison of Decoder Accuracy over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    plt.savefig("./img/info-acc-surrogates.pdf", format = "pdf")

def plot_loss_data(
    *paths: str,
    plot_loss: bool = True
):
    """
    Loads pickle files from three different paths and plots the data.

    Each pickle file is expected to be at: <path>/bin/test-metrics.pkl
    The pickle file should contain a tuple[list, list]: (loss, acc).
    both lists should contain #epochs lists with #batches of entries each.

    :param paths: Path towards the three data directories
    :type paths: str, required
    :param plot_loss: Whether to plot the loss or accuracy (Default True)
    :type plot_loss: bool, optional
    """
    if not paths:
        raise ValueError("At least one path must be provided.")

    # Define default colors and labels (will be extended if needed)
    default_colors = ["b", "g", "r", "c", "m", "y", "k", "tab:blue", "tab:orange", "tab:green"]

    # Use as many colors and labels as needed
    colors = default_colors[:len(paths)]
    labels = [Path(path).name for path in paths]

    fig, ax = plt.subplots(
        1,
        figsize = (10,6),
    )

    for i, base_path in enumerate(paths):
        # Construct the full file path
        file_path = os.path.join(base_path, "bin", "test-metrics.pkl")

        try:
            # Load the pickle file
            with open(file_path, "rb") as f:
                loss, acc = pickle.load(f)

            loss = [np.mean(ep) for ep in loss]
            acc  = [np.mean(ep) for ep in acc]
            if i == 4:
                big = np.argmax(loss)
                loss = loss[:big]
                acc = acc[:big]

            # what to plot
            if plot_loss:
                data = loss
            else:
                data = acc

            x_vals = np.arange(len(loss))

            # Plot the loss
            ax.plot(
                x_vals,
                data,
                color = colors[i],
                label = labels[i],
                marker = "o",
                # s = 2,
                linestyle = "-"
            )
            print(f"MaxVal for {file_path}: {np.max(data)}")


        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")

    y_title = "Loss" if plot_loss else "Accuracy"
    fig.suptitle("Comparison of Accuracy for Surrogate Gradients")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(y_title)
    ax.legend()
    ax.grid(True)
    print("Saved")
    plt.savefig(f"./img/{y_title.lower()[:4]}-surrogates-full.pdf", format = "pdf")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot data from three pickle files.")

    # Add arguments for the three paths
    parser.add_argument(
        "paths",
        type = str,
        nargs = "+",
        help = "Directory paths for the data."
            "Each should contain bin/test-metrics.pkl and estim/info.pkl"
    )

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    plot_lif_voltage()

    plot_info_data(*args.paths)
    plot_info_decoder_loss(*args.paths)
    plot_loss_data(*args.paths, plot_loss = False)