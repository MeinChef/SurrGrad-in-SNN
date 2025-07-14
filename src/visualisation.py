from imports import numpy as np
from imports import plt
from imports import torch
from imports import spikeplot as splt
from misc import make_path, load_spk_rec


def plot_loss_acc(config:dict) -> plt.Figure:
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

def plot_recordings(
        config:dict,
        identifier:str = "test-ep0"
) -> plt.Figure:
    """
    Nicely plot the spike-recordings of the hidden layers
    
    :param config: the config dictionary
    :type config: dict
    :return: the figure object
    :rtype: plt.Figure
    """

    # load the recordings
    rec = load_spk_rec(
        config = config,
        identifier = identifier
    )

    fig, axs = plt.subplots(
        nrows = 1,
        ncols = 1,
        figsize = (10, 6)
    )

    tmp = torch.tensor(rec[0]["arr_1"])
    #  s: size of scatter points; c: color of scatter points
    splt.raster(tmp, axs)

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    # Example usage
    config = {
        "data_path": "data",
    }
    plot_recordings(config, identifier="test-ep0")