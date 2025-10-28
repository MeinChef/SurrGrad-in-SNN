from imports import os
from imports import datetime
from imports import warnings
from imports import numpy as np
from imports import torch
from imports import yaml
from imports import plt
from misc import make_path


# load the config.yml
def load_config(path: str = "config.yml") -> tuple[dict, dict]:
    with open(path, "r") as file:
        configs = yaml.safe_load(file)

    return configs["data"], configs["model"]

class DataHandler:
    def __init__(
        self,
        path: str = "data/",
    ) -> None:
        """
        Class that handles the recording and loading of the data that is being created during training.
        """
        self.path = path
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def spk_rec_to_file(
        self,
        data: dict = None,
        identifier: str | list[str] = None,
    ) -> None:
        """
        Function for saving the spike recording of the hidden layers into a file on disk.
        
        :param data: Dictionary, with each key containing a list: [[recording_of_layer1], [recording_of_layer2], [recording_of_layer3]]
        :type data: dict, required
        
        :param identifier: Useful for saving the data with a custom filename
        :type identifier: str or list[str], optional

        :param path: Path where to save the data. Default data/rec/
        :type path: str, optional
        """

        # make path os-independent
        path = make_path(self.path + "/rec/")

        # TODO: change path resolving with str.split() and os.path.join()
        if isinstance(identifier, list):
            assert len(identifier) == 3
        elif isinstance(identifier, str):
            identifier = [identifier + "-layer1.npz", identifier + "-layer2.npz", identifier + "-layer3.npz"]
        elif identifier is None:
            identifier = [self.now + "-layer1.npz", self.now + "-layer2.npz", self.now + "-layer3.npz"]

        for key in data.keys():
            for i, layer in enumerate(data[key]):
                # check if there are even any recordings left to save
                if not layer:
                    continue

                # try concatenating the layers along batch dimension
                try:
                    array = torch.cat(layer, dim = 1)
                    if array.get_device() >= 0:
                        array = array.cpu()
                    array = array.numpy().astype(np.int8)

                    np.save(
                        os.path.join(
                            path, 
                            key + "_" + identifier[i]
                        ),
                        array
                    )

                # if that didn't work, save each recording individually
                except Exception as e:
                    # warn that it didn't happen as expected
                    warnings.warn(
                        "Failed saving the recordings as single Tensor. " +
                        "Proceeding with saving the recordings individually.\n" +
                        f"Actual Exception: {e}"
                    )

                    for j, rec in enumerate(layer):

                        # if recording is in GPU memory
                        if rec.get_device() >= 0:
                            rec = rec.cpu().numpy()
                            # we shouldn"t loose any expressiveness, since spikes are usually 0s or 1s
                            layer[j] = rec.astype(np.int8) 
                        # or it's on cpu
                        elif rec.get_device() == -1:
                            layer[j] = rec.numpy().astype(np.int8)
                    
                    np.savez_compressed(
                        os.path.join(
                            path,
                            key + identifier[i]
                        ),
                        layer
                    )

    def load_spk_rec(
        self,
        identifier: str = "test-ep0"
    ) -> list[np.ndarray]:
        """
        Function for loading the spike recordings of the hidden layers from a file on disk.
        
        :param config: config dictionary
        :type config: dict
        
        :param identifier: Useful for loading the data with a custom filename
        :type identifier: str, optional

        :return: list of the structure [[recording_of_layer1], [recording_of_layer2], [recording_of_layer3]]
        :rtype: list[np.ndarray]
        """

        paths = [make_path(self.path + "/rec/" + identifier + f"-layer{layer}.npz") for layer in range(1,4)]

        recordings = []
        for path in paths:
            try:
                data = np.load(path) # old: allow_pickle = True
                # data is an open file - needs to be closed before returning
                # data is a dictionary-ish containing keys arr_0, arr_1.....
                
                layer = []
                for i in range(len(data)):
                    layer.append(
                        pad_along_axis(
                            data.get(f"arr_{i}"),
                            pad_val = 0
                        )
                    )

                recordings.append(
                    np.concatenate(
                        layer,
                        axis = 1,
                        dtype = np.int8
                    )
                )
                data.close()

            except FileNotFoundError:
                print(f"File {path} not found. Skipping...")
                continue

        return recordings

    def flush_to_file(
        self,
        loss: list,
        loss_ident: str | None = None,
        acc: list = None, 
        acc_ident: str | None = None,
        spk_rec: dict[list[list,list,list]] = None,
        spk_ident: str = None
    ) -> None:
        """
        Saves the output from the model to a file, human-readable.
        
        :param loss: list of loss values
        :type loss: list
        
        :param acc: list of accuracy values
        :type acc: list

        :param spk_rec: spk_rec of all layers 
        :type spk_rec: dict, optional
        """

        if len(loss) != 0:
            try:
                np.savetxt(
                    make_path(self.path + f"/loss_{loss_ident}.txt"),
                    loss,
                    fmt = "%.8f"
                )
            except Exception as e:
                print(f"An error occurred: {e}")
                breakpoint()
        if acc:
            try:
                if len(acc) != 0:
                    np.savetxt(
                        make_path(self.path + f"/acc_{acc_ident}.txt"),
                        acc,
                        fmt = "%.8f"
                    )
            except Exception as e:
                print(f"An error occurred: {e}")
                breakpoint()

        if spk_rec:
            try:
                self.spk_rec_to_file(
                    data = spk_rec,
                    identifier = spk_ident,
                )
            except Exception as e:
                print(f"An error occurred: {e}")
                breakpoint()

    def plot_spikes(
        self,
        identifier: str = "",
        infer_epochs: bool = False,
        **kwargs
    ) -> plt.Axes:

        """
        Plots the activations of the network, as it has been recorded during training.
        For the function to properly infer the epochs, the files have to be named with the structure 
        [identifier]-ep[#epoch]-layer[#].npz and identifier has to be provided
        """

        # Subject to Re-Write
        rec = []
        if infer_epochs:
            cont = os.listdir(make_path(self.path + "/rec/"))
            cont = [file for file in cont if identifier in file]

            epochs = 0
            while len([file for file in cont if f"ep{epochs}" in file]) >= 3:
                epochs +=1

            for epoch in range(epochs):
                rec.append(
                    self.load_spk_rec(
                        identifier = identifier + f"-ep{epoch}"
                    )
                )

        else:
            if identifier != "":
                rec = self.load_spk_rec(identifier = identifier)
            else:
                rec = self.load_spk_rec()
        # breakpoint()
        # make as many plots as there are epochs
        # rows of 10 classes
        # columns of 3, for the layers


        # for i in range(max(epochs, 1)): 
        #     pass
        #     # fig, axes = plt.subplots


        # fig, axes = plt.subplots(
        #     nrows = len(rec),
        #     ncols = 3,
        #     **kwargs
        # )
        
        # for i in range(len(axes)):
        #     for j in range(len(rec[i])):
        #         try:
        #             # plot only one sample for now
        #             spikeplot.raster(
        #                 torch.tensor(rec[i][j][:,0]),
        #                 ax = axes[i][j]
        #             )
        #         except:
        #             breakpoint()

        axes = []
        for i in range(len(rec[0])):
            if i < len(rec[0]) - 1:
                _, ax = plt.subplots(
                    nrows = 1, 
                    ncols = 1, 
                    subplot_kw = {"projection": "3d"},
                )
                ax.set_box_aspect((6,1,1))
                ax.set_xlabel("Step")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                axes.append(ax)

            else: 
                # if this is the last layer, we only want to plot 2d data
                _, ax = plt.subplots(
                    nrows = 1,
                    ncols = 1,
                    figsize = (20,4)
                )
                ax.set_xlabel("Step")
                ax.set_ylabel("Y")
                ax.set_ylim(0, 10)
                ax.set_xlim(0, 314)
                ax.grid(True)
                axes.append(ax)

        for i, ax in enumerate(axes):
            if ax == axes[-1]:
                arr = rec[0][i][:,0].squeeze()
                ax.scatter(
                    arr[:,0],
                    arr[:,1],
                    color = "black",
                )
                ax.set_title(f"No. Spikes: {arr.sum()}.")

            else:
                # take one sample of ith layer
                arr = rec[0][i][:,0].squeeze()
                arr = np.sum(arr, axis = 1)
                
                ax.set_xlim(0, arr.shape[0])
                ax.set_ylim(0, arr.shape[1])
                ax.set_zlim(0, arr.shape[2])

                for step in range(arr.shape[0]):
                    # Scatter points where arr[step, y, z] == 1
                    if arr[step].sum() == 0:
                        continue

                    ys, zs = np.where(arr[step] >= 1)
                    xs = np.full_like(ys, step)
                    ax.scatter(
                        xs, 
                        ys, 
                        zs, 
                        color = "black", 
                        s = arr[step, arr[step] >= 1] ** 2
                    )

        plt.tight_layout()
        plt.show()

    def plot_loss_accuracy(
        self,
        loss: list,
        accuracy: list,
        training: bool = False,
        epoch: int = -1,
        filename: str | None = None,
        show: bool = False
    ) -> None:
        


        fig, ax1 = plt.subplots()

        # plot loss on the left y-axis
        ax1.set_xlabel("Batches")
        ax1.set_ylabel("Loss", color = "orange")
        ax1.plot(
            range(len(loss)), 
            loss, 
            color = "orange", 
            label = "Loss"
        )
        ax1.tick_params(axis = "y", labelcolor = "orange")

        # create a second y-axis for accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color = "blue")
        ax2.plot(
            range(len(loss)), 
            accuracy, 
            color = "blue", 
            label = "Accuracy"
        )
        ax2.tick_params(axis = "y", labelcolor = "blue")

        plt.title(f"Loss and Accuracy. Epoch {epoch}{' of Training' if training else ' of Testing'}")
        fig.tight_layout()
        if filename:
            plt.savefig(
                os.path.join(self.path, self.now + "_" + filename + ".png")
            )
        if show:
            plt.show()

        
def pad_along_axis(
        array: np.ndarray, 
        target_length: int = 314, # as calculated from misc.get_longest_observation 
        axis: int = 0,
        pad_val: int = -1
    ) -> np.ndarray:

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(
        array, 
        pad_width = npad, 
        mode = "constant",
        constant_values = pad_val
    )