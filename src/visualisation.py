from imports import numpy as np
from imports import plt
from imports import torch
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
