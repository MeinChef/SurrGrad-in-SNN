from imports import numpy as np
from imports import plt
from imports import torch
from imports import snntorch as snn
from imports import spikeplot as splt
from imports import Figure
from misc import make_path# , load_spk_rec


def plot_loss_acc(config:dict) -> Figure:
    # load values from files
    loss = np.loadtxt(
        make_path(config["data_path"] + "/loss.txt"),
    )

    acc = np.loadtxt(
        make_path(config["data_path"] + "/acc.txt"),
    )

    assert len(loss) == len(acc), print(f"Loss ain't acc, off by {len(loss)-len(acc)}")
    
    epochs = np.arange(1, len(loss) + 1)

    fig, ax1 = plt.subplots()

    # Plot loss on the left y-axis
    ax1.set_xlabel('Batches')
    ax1.set_ylabel('Loss', color='orange')
    ax1.plot(epochs, loss, color='orange', label='Loss')
    ax1.tick_params(axis='y', labelcolor='orange')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='blue')
    ax2.plot(epochs, acc, color='blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title('Loss and Accuracy during Training')
    fig.tight_layout()
    plt.show()

    return fig

def plot_epoch_losses(
    epoch_losses,
    steps_per_epoch=None,
    smooth=False,
    window=5,
    figsize=(10, 5),
):
    """
    Plot training loss over steps for multiple epochs.

    Parameters
    ----------
    epoch_losses : list[list[float]]
        A list where each sublist contains loss values for one epoch.
    steps_per_epoch : int | None
        Optional fixed number of steps per epoch. If None, inferred from sublist lengths.
    smooth : bool
        Whether to apply moving average smoothing.
    window : int
        Moving average window size if smooth=True.
    figsize : tuple
        Figure size for matplotlib.
    """

    plt.figure(figsize=figsize)

    global_step = 0

    for epoch_idx, losses in enumerate(epoch_losses):
        losses = np.array(losses)

        # Optional smoothing
        if smooth and len(losses) >= window:
            kernel = np.ones(window) / window
            losses = np.convolve(losses, kernel, mode="valid")

        # X-axis steps
        if steps_per_epoch is None:
            steps = np.arange(global_step, global_step + len(losses))
            global_step += len(losses)
        else:
            steps = np.arange(
                epoch_idx * steps_per_epoch,
                epoch_idx * steps_per_epoch + len(losses),
            )

        plt.plot(steps, losses, label=f"Epoch {epoch_idx + 1}")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
        3, 
        figsize = (8,6), 
        sharex = True, 
        gridspec_kw = {'height_ratios': [1, 1, 0.4]}
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
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(
        spk_rec, 
        ax[2], 
        s = 400, 
        c = "black", 
        marker = "|"
    )

    plt.ylabel("Output spikes")
    plt.yticks([]) 

    plt.savefig("./img/lif-neuron.png")
    plt.show()

if __name__ == "__main__":
    plot_lif_voltage()