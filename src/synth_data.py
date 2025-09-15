import torch
import numpy as np
import timeit
import math

# https://stackoverflow.com/questions/14446128/append-vs-extend-efficiency

class NewGenerator:
    def __init__(
        self,
        time_steps: int,
        jitter: float = 0.0,
        neurons: int = 10,
        min_isi: int = 1,
        max_isi: int = 10,
        min_rate: int = 5,
        max_rate: int = 20,
    ):
        self.time_steps = time_steps
        self.jitter = jitter
        self.neurons = neurons
        self.min_isi  = min_isi
        self.max_isi  = max_isi
        self.min_rate = min_rate
        self.max_rate = max_rate

        # create a grid of all possible (isi, rate) pairs
        value_grid = np.meshgrid(
            np.arange(self.min_isi, self.max_isi + 1),
            np.arange(self.min_rate, self.max_rate + 1)
        )
        # create class assignment matrix (half of the values are class 0, half class 1)
        class_assignment = self._class_assign_matrix(
            np.zeros_like(
                value_grid[0],
                dtype = np.uint8
            )
        )
        self.isis    = value_grid[0].flatten()
        self.rates   = value_grid[1].flatten()
        self.classes = class_assignment.flatten()

        self.rng = np.random.default_rng()

    def _class_assign_matrix(
            self, 
            arr: np.ndarray
    ) -> np.ndarray:
        slope = arr.shape[0]/arr.shape[1]

        # fill the "upper triangle" from [0,0] to [n_rows, n_cols]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                # everything that is above or on the slope is 1
                if j >= i * slope:
                    arr[i, j] = 1
        return arr
    
    def _generate_sample(
        self,
        isi: int,
        rate: int
    ) -> np.ndarray:
        
        # calculate number of spikepairs, given in [source]
        spk_pairs = int(rate * (self.time_steps / 1000))

        # preallocate the resulting array
        out = np.zeros(
            shape = (self.time_steps, self.neurons),
            dtype = np.float32,
        )

        for i in range(self.neurons):
            mask = np.ones(
                (self.time_steps - isi,), 
                dtype = bool
            )

            cur_no = 0
            while cur_no < spk_pairs:
                sample = self.rng.choice(
                    np.flatnonzero(mask),
                    size = (1,),
                    replace = True,
                    shuffle = False # not needed, but makes it faster
                )[0]

                # set value directly in the array
                out[sample, i] += 1
                out[sample + isi, i] += 1
                
                # check boundaries and handle all cases
                left = max(sample - isi, 0)
                right = min(sample + isi + 1, len(mask))
                mask[left:right] = False
                cur_no += 1
                
                # break if no valid positions are left
                if mask.sum() == 0:
                    break

        return out
    
    # TODO: does not work, optimize
    def _jitter(
        self,
        sample: np.ndarray,
    ) -> np.ndarray:
        
        spikes = np.nonzero(sample)
        valid_to = np.nonzero(sample == 0)

        for neuron in range(sample.shape[1]):
            mask = spikes[1] == neuron
            to_move = math.ceil(len(spikes[0][mask]) * self.jitter)
            selected_spikes = self.rng.choice(
                spikes[0][mask],
                size = to_move,
                replace = False,
                shuffle = False
            )
            new_positions = self.rng.choice(
                valid_to[0][mask],
                size = to_move,
                replace = False,
                shuffle = False
            )
            sample[selected_spikes, neuron] -= 1
            sample[new_positions, neuron] += 1
        
        return sample


    def generate_samples(
        self,
        no_samples: int = 3000,
    ) -> tuple[np.ndarray, np.ndarray]:
        samples_per_class = math.ceil(no_samples / 2)
        samples = []
        labels  = []

        for i in range(2):
            indices = self.rng.choice(
                np.nonzero(self.classes == i)[0],
                size = samples_per_class,
                replace = True
            )
            for idx in indices:
                # generate sample with self.isis[idx] and self.rates[idx]
                sample = self._generate_sample(
                    isi = self.isis[idx],
                    rate = self.rates[idx]
                )
                if self.jitter > 0:
                    sample = self._jitter(
                        sample = sample,
                    )
                # append to list of samples
                # append class label to list of labels
                samples.append(sample)
                labels.append(i)
        samples = np.stack(samples)
        labels = np.array(labels)
        return samples, labels

    def generate_dataset(
        self
    ) -> torch.utils.data.DataLoader:
        pass
    




if __name__ == "__main__":

    data = NewGenerator(
        time_steps = 100,
        jitter = 0.1,
        min_isi = 1,
        max_isi = 10,
        min_rate = 5,
        max_rate = 20,
    )
    data.generate_samples()

    # TODO: benchmark
    # print(timeit.timeit(
    #     data.generate_samples,
    #     number = 10000
    # ))