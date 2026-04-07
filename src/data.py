from imports import yaml
from imports import datetime
from imports import functional
from imports import torch
from imports import plt


# load the config.yml
def load_config(path: str = "config.yml") -> tuple[dict, dict]:
    with open(path, "r") as file:
        configs = yaml.safe_load(file)

    return configs["data"], configs["model"]

class DataHandler():
    def __init__(
        self,
        time_steps: int,
        datapath: str = "data/",
    ) -> None:
        
        self.path = datapath
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.time_steps = time_steps

    def visualise(
        self,
        recorder: functional.probe.OutputMonitor,
        blocking: bool = True
    ) -> plt.figure:
        
        # instantiate plot
        fig, axes = plt.subplots(
            nrows = 2,
            ncols = len(recorder.monitored_layers),
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
        for i, layer in enumerate(recorder.monitored_layers):
            # i'm only interested in the first sample for now
            layerlist = recorder[layer][:self.time_steps]
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
        plt.tight_layout(pad=2.0)
        plt.show()