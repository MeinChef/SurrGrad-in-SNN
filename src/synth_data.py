import torch
import numpy as np
import timeit

# TODO: subclass generator for realsies, or return a torch dataloader somewhere
class DataGenerator():
    def __init__(
            self,
            time_steps: int,
            inter_spike_intervals: np.ndarray,
            rate: float = 15.,
            neurons: int = 10,
            no_samples: int = 3e4,
            move_spikes: float = 0.,
        ) -> None:
        
        """
        isi should be 1d array, and len = classes
        """


        if rate < 5 or rate > 20:
            raise ValueError(f"Rate is in an invalid range. Valid range is [5;20]. Actual value: {rate}")
        if move_spikes < 0 or move_spikes > 1:
            raise ValueError(f"Move_spikes is in an invalid range. Valid range is [0;1]. Actual value: {move_spikes}")

        self.rng = np.random.default_rng()

        self.time_steps = time_steps
        self.neurons = neurons
        self.inter_spike_intervals = inter_spike_intervals
        self.no_spike_pairs_per_neuron = np.floor(rate * (self.time_steps / 1000))
        self.total_samples = no_samples


    def gen_data(
        self,
        isi: int = 10,
    ) -> np.ndarray:
        """
        generate one sample 
        """
        if isi < 1 or isi > 10:
            raise ValueError(f"ISI is in an invalid range. Valid range is [1;10]. Actual value: {isi}")
        
        full_data = []
        for i in range(self.neurons):
            valid_starts = np.arange(
                self.time_steps - isi, 
                dtype = np.int32, 
            )

            cur_no = 0
            while cur_no < self.no_spike_pairs_per_neuron:
                sample = self.rng.choice(
                    valid_starts,
                    size = (1,),
                    replace = True,
                    shuffle = False # not needed, but makes it faster
                )[0]
                full_data.extend(((sample, i), (sample + isi, i)))
                valid_starts = valid_starts[abs(valid_starts - sample) > isi]
                cur_no += 1
        
        spikes = np.array(full_data)
        out = np.zeros(
            shape = (self.time_steps, self.neurons),
            dtype = np.float32,
        )
        out[spikes[:,0], spikes[:,1]] = 1
        return out
    
    def gen_data_torch_cpu(
        self,
        isi: int = 10
    ) -> torch.Tensor:
        """
        generate one sample 
        """
        if isi < 1 or isi > 10:
            raise ValueError(f"ISI is in an invalid range. Valid range is [1;10]. Actual value: {isi}")
        
        full_data = []
        for i in range(self.neurons):
            valid_starts = torch.arange(
                self.time_steps - isi, 
                dtype = torch.int32,
            )

            cur_no = 0
            while cur_no < self.no_spike_pairs_per_neuron:
                sample = torch.randint(
                    low = 0,
                    high = len(valid_starts),
                    size = (1,),
                ).item()
                sample = valid_starts[sample]
                full_data.extend(((sample, i), (sample + isi, i)))
                valid_starts = valid_starts[abs(valid_starts - sample) > isi]
                cur_no += 1

        spikes = torch.tensor(full_data)
        out = torch.zeros(
            size = (self.time_steps, self.neurons),
            dtype = torch.float32,
        )
        out[spikes[:,0], spikes[:,1]] = 1
        return out

    def gen_data_torch_cuda(
        self,
        isi: int = 10
    ) -> torch.Tensor:
        """
        generate one sample 
        """
        if isi < 1 or isi > 10:
            raise ValueError(f"ISI is in an invalid range. Valid range is [1;10]. Actual value: {isi}")
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        full_data = []
        for i in range(self.neurons):
            valid_starts = torch.arange(
                self.time_steps - isi, 
                dtype = torch.int32,
                device = device
            )

            cur_no = 0
            while cur_no < self.no_spike_pairs_per_neuron:
                sample = torch.randint(
                    low = 0,
                    high = len(valid_starts),
                    size = (1,),
                    device = device
                ).item()
                sample = valid_starts[sample]
                full_data.extend(((sample, i), (sample + isi, i)))
                valid_starts = valid_starts[abs(valid_starts - sample) > isi]
                cur_no += 1
        
        spikes = torch.tensor(full_data)
        out = torch.zeros(
            size = (self.time_steps, self.neurons),
            dtype = torch.float32,
            device = device
        )
        out[spikes[:,0], spikes[:,1]] = 1
        return out


if __name__ == "__main__":

    data = DataGenerator(
        time_steps = 1000,
        inter_spike_intervals = np.arange(1)
    )


    print(timeit.timeit(
        data.gen_data,
        number = 10000
    ))
    # 22.460367957000017
    # 22.72771648700018
    # 22.611762075000115

    print(timeit.timeit(
        data.gen_data_torch_cpu,
        number = 10000
    ))
    # 48.99222566199933
    # 48.467646289000186

    print(timeit.timeit(
        data.gen_data_torch_cuda,
        number = 10000
    ))
    # 157.93103715899997