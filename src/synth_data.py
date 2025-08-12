import torch
import numpy as np
import timeit

# https://stackoverflow.com/questions/14446128/append-vs-extend-efficiency

# TODO: subclass generator for realsies, or return a torch dataloader somewhere
class DataGenerator():
    def __init__(
            self,
            time_steps: int,
            inter_spike_intervals: np.ndarray,
            rate: float = 15.,
            neurons: int = 10,
            no_samples: int = 3e3,
            move_spikes: float = 0.,
        ) -> None:
        
        """
        isi should be 1d array, and len = classes
        """


        if rate < 5 or rate > 20:
            raise ValueError(f"Rate is in an invalid range. Valid range is [5;20]. Actual value: {rate}")
        if move_spikes < 0 or move_spikes > 1:
            raise ValueError(f"Move_spikes is in an invalid range. Valid range is [0;1]. Actual value: {move_spikes}")
        if time_steps > np.iinfo(np.uint16).max:
            raise ValueError(f"Maximum amount of time steps exceeded. Max is {np.iinfo(np.uint16).max}, got {time_steps}")


        self.rng = np.random.default_rng()

        self.time_steps = time_steps
        self.neurons = neurons
        self.inter_spike_intervals = inter_spike_intervals
        self.no_spike_pairs_per_neuron = int(rate * (self.time_steps / 1000))
        self.total_samples = int(no_samples)


    def gen_sample(
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
                dtype = np.uint16, 
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
        out[spikes[:,0], spikes[:,1]] += 1
        return out
    
    def generate_dataset(
        self
    ) -> torch.utils.data.DataLoader:
        samples_per_class = int(self.total_samples / len(self.inter_spike_intervals))
        missing = self.total_samples - samples_per_class * len(self.inter_spike_intervals) 

        samples = [samples_per_class] * len(self.inter_spike_intervals)

        if missing != 0:
            samples[:missing] += 1
        
        data = []
        target = []
        for i, no in enumerate(samples):
            data.extend([
                self.gen_sample(
                    isi = self.inter_spike_intervals[i]
                ) for _ in range(no)
            ])
            
            target.append(np.full(
                shape = (no,),
                fill_value = i,
                dtype = np.float32
            ))

        data = np.stack(data)
        target = np.concatenate(target)

        data = torch.from_numpy(data)
        target = torch.from_numpy(target)

        dataset = torch.utils.data.TensorDataset(data, target)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 64,
            shuffle = True,
            num_workers = 6
        )
        return dataloader

# def unison_shuffled_copies(
#         arr_0: np.ndarray, 
#         arr_1: np.ndarray
#     ) -> tuple[np.ndarray, np.ndarray]:
#     """
#         Convenience function to shuffle two arrays in unison along their first dimension.
#         :param arr_0: First array to shuffle.
#         :type arr_0: np.ndarray, required
#         :param arr_1: Second array to shuffle.
#         :type arr_1: np.ndarray, required
#         :return: Tuple of shuffled arrays.
#         :rtype: tuple[np.ndarray, np.ndarray]
#     """
#     # set random seed for reproducibility
#     np.random.seed(42)
#     assert len(arr_0) == len(arr_1)

#     p = np.random.permutation(len(arr_0))
#     return arr_0[p], arr_1[p]



if __name__ == "__main__":

    data = DataGenerator(
        time_steps = 1000,
        inter_spike_intervals = np.arange(1,11)
    )
    # data.generate_dataset()

    # print(timeit.timeit(
    #     data.gen_data,
    #     number = 10000
    # ))
    # # 22.460367957000017
    # # 22.72771648700018
    # # 22.611762075000115

    # print(timeit.timeit(
    #     data.gen_data_torch_cpu,
    #     number = 10000
    # ))
    # # 48.99222566199933
    # # 48.467646289000186

    # print(timeit.timeit(
    #     data.gen_data_torch_cuda,
    #     number = 10000
    # ))
    # 157.93103715899997


    print(timeit.timeit(
        data.gen_sample_lst,
        number = 10000
    ))
    # 24.029359733998717
    # 23.42586822999874
    print(timeit.timeit(
        data.gen_sample_np,
        number = 10000
    ))
    # 24.91761143199983
    # 24.41251021700009