from imports import yaml
from imports import datetime
from imports import functional
from imports import snntorch as snn
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
        recorder: functional.probe.OutputMonitor,
        time_steps: int,
        datapath: str = "data/",
    ) -> None:
        
        self.recorder = recorder
        self.path = datapath
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.time_steps = time_steps


    def measure_tendency(
        self
    ) -> dict:
        
        #
        measurements = {}
        rate = []
        for layer in self.recorder.monitored_layers:
            # create tensor from recording
            layerlist = self.recorder[layer][:self.time_steps]
            spikes = torch.stack([x[0,:] for x, _ in layerlist]).T
            
            # calculate the average isi, synchrony
            tmp_rate = []
            for neuron in range(spikes.shape[0]):
                returnvalue = self.measure_rate(spikes[neuron])
                if returnvalue:
                    tmp_rate.append(returnvalue["no_spk"])
                else:
                    tmp_rate.append(torch.tensor([0]))
                # self.measure_latency()
            rate.append(torch.tensor(tmp_rate))
            return rate

    def measure_rate(
        self,
        spike_train: torch.Tensor
    ) -> dict | bool:
        
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
        
        # measure isis, get variance
        # since spike train is a vector, the tuple only has one entry.
        spike_loc = spike_train.nonzero(as_tuple = True)[0]

        if len(spike_loc) <= 1:
            return False
        
        spike_loc, _ = torch.sort(spike_loc)

        # there will be one less isi than the length of the spiketrain
        isi = torch.zeros((len(spike_loc - 1)))

        # starting at index 1, calculate the isi for [0-1] and [1-2].
        # then jump to index 3, calculate [2-3] and [3-4]
        # keep in mind that if a list is might not have the 1+ith entry
        for i in range(1,len(spike_loc), 2):
            isi[i-1] = spike_loc[i] - spike_loc[i-1]
            if i+1 < len(spike_loc):
                isi[i] = spike_loc[i+1] - spike_loc[i]
        
        return {
            "no_spk": spike_train.sum(),
            "rate": spike_train.sum() / 1e3,
            "avg": isi.mean(),
            "med": isi.median(),
            "var": isi.var(),  
            "stdev": isi.std()
        }

    def measure_latency(
        self,
    ):
        raise NotImplementedError
    
    def visualise(
        self,
        blocking: bool = False
    ) -> plt.figure:
        
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
        output
    ) -> plt.figure:
        fig, axes = plt.subplots(
            nrows = 1, 
            ncols = 3,
            squeeze = True,
            figsize = (14,5),
            dpi = 200
        )

        for i, layer in enumerate(output):
            layer = torch.where(layer == 0, torch.tensor(float('nan')), layer)
            axes[i].scatter(torch.arange(len(layer)), layer)
        
        
        plt.tight_layout()
        plt.show()