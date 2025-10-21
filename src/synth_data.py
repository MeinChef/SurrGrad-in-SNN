from imports import numpy as np
from imports import math
from imports import os
from imports import torch
from imports import plt

DEBUG = False

class DataGenerator:
    def __init__(
        self,
        time_steps: int,
        jitter: float = 0.0,
        neurons: int = 10,
        min_isi: int = 1,
        max_isi: int = 10,
        min_rate: int = 5,
        max_rate: int = 20,
        precision: np.dtype = np.float32
    ):
        self.time_steps = time_steps
        self.jitter = jitter
        self.neurons = neurons
        self._min_isi  = min_isi
        self._max_isi  = max_isi
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._precision = precision

        # create a grid of all possible (isi, rate) pairs
        value_grid = np.meshgrid(
            np.arange(self._min_isi, self._max_isi + 1),
            np.arange(self._min_rate, self._max_rate + 1)
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
        self.torch_rng = torch.default_generator

    def _class_assign_matrix(
            self, 
            arr: np.ndarray
    ) -> np.ndarray:
        slope = arr.shape[0]/arr.shape[1]

        # fill the "upper triangle" from [0,0] to [n_rows, n_cols]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                # everything that is above or on the slope is class 1
                # everything below stays class 0.
                if j >= i * slope:
                    arr[i, j] = 1
        return arr
    
    def _generate_sample(
        self,
        isi: int,
        rate: int
    ) -> np.ndarray:
        
        # calculate number of spikepairs, given in [source]
        spk_pairs = math.floor(rate * (self.time_steps / 1000))

        # preallocate the resulting array
        out = np.zeros(
            shape = (self.time_steps, self.neurons),
            dtype = self._precision,
        )

        for i in range(self.neurons):
            # array with valid start positions of a spike pair
            mask = np.ones(
                (self.time_steps - isi,), 
                dtype = bool
            )

            cur_no = 0
            while cur_no < spk_pairs:
                # select a random starting position
                sample = np.flatnonzero(mask)
                self.rng.shuffle(
                    sample
                )
                sample = sample[0]

                # set value directly in the array
                out[sample, i] += 1
                out[sample + isi, i] += 1
                
                # check boundaries and handle all cases
                left = max(sample - isi, 0)
                right = min(sample + isi + 1, len(mask))
                # and invalidate the space around the just set pair
                # to make sure there are no overlaps
                mask[left:right] = False
                cur_no += 1
                
                # break if no valid positions are left
                if mask.sum() == 0:
                    print("Warning: Did not generate all spike pairs, no valid positions left")
                    break

        return out
    
    def _jitter(
        self,
        sample: np.ndarray,
    ) -> np.ndarray:
        
        # get indices where spikes are, and where they aren't 
        spikes = np.nonzero(sample)
        valid_to = np.nonzero(sample == 0)


        for neuron in range(sample.shape[1]):
            # create a mask for each neuron
            # that is because spikes should not change their associated neuron
            spk_mask = spikes[1] == neuron
            move_mask = valid_to[1] == neuron
            to_move = math.ceil(spk_mask.sum() * self.jitter)

            # jumble spikes and select the first to_move ones 
            # (same as random choice without replacement, but faster)
            selected_spikes = self.rng.permutation(
                spikes[0][spk_mask]
            )[:to_move]

            # generate new position from valid ones
            new_positions = self.rng.permutation(
                valid_to[0][move_mask]
            )[:to_move]

            # move spikes (or remove the original positions of the spikes moved)
            # (and add spikes where they got moved to)
            sample[selected_spikes, neuron] -= 1
            sample[new_positions, neuron] += 1
        
        return sample


    def generate_samples(
        self,
        no_samples: int = 3000,
    ) -> tuple[np.ndarray, np.ndarray]:
        samples_per_class = math.ceil(no_samples / 2)

        if DEBUG:
            print(f"Samples:\n Type: {type(no_samples)}\n Actual: {no_samples}")
            print(f"Neurons:\n Type: {type(self.neurons)}\n Actual: {self.neurons}")
            print(f"TimeSteps:\n Type: {type(self.time_steps)}\n Actual: {self.time_steps}")

        # preallocate arrays
        samples = np.empty(
            shape = (no_samples, self.time_steps, self.neurons),
            dtype = self._precision
        )
        labels  = np.empty(
            shape = (no_samples,),
            dtype = np.uint8
        )

        for i in range(2):
            # generate samples by drawing an index from the class assignment array
            # where the class is the specified one 
            indices = self.rng.choice(
                np.nonzero(self.classes == i)[0],
                size = samples_per_class,
                replace = True
            )
            for j, idx in enumerate(indices):
                # generate sample with self.isis[idx] and self.rates[idx]
                # this works, because all arrays got flattened
                sample = self._generate_sample(
                    isi = self.isis[idx],
                    rate = self.rates[idx]
                )

                # some debug print statements for roughly checking whether the jittering worked
                if DEBUG:
                    sample_sum = [sample[:,i].sum() for i in range(self.neurons)]
                    print(f"Rate: {self.rates[idx]}, ISI: {self.isis[idx]}, Class: {self.classes[idx]}")
                    print(f"Sample sum per neuron: {sample_sum}.\n" +
                          f"Expected sum per neuron: {self.rates[idx] * self.neurons * (self.time_steps / 1000) * 2}")
                    
                if self.jitter > 0:
                    sample = self._jitter(
                        sample = sample,
                    )
                    
                    if DEBUG:
                        print("After jittering:\n" +
                              f"Sample sum per neuron: {[sample[:,i].sum() for i in range(self.neurons)]}.\n" +
                              "Expected to match sample sum per neuron.\n" +
                              f"{[sample[:,i].sum() for i in range(self.neurons)] == sample_sum}"
                              )
                        
                # store samples and labels
                # i * len(indices) specifies the initial offset from the array start
                # as in: is it class one (saved from 0 - len(indices)) or is it class two
                # which is saved from len(indices) to -1
                # the +j is the offset from the start of each class.
                samples[i * len(indices) + j] = sample
            labels[i * len(indices) : (i + 1) * len(indices)] = i

        return samples, labels

    def generate_dataset(
        self,
        no_samples: int = 3000,
        batch_size: int = 256,
        shuffle: bool = True,
        train_split: float = 0.8,
        prefetch: int = 16,
        workers: int = max(2, os.cpu_count() - 4)
    ) -> torch.utils.data.DataLoader:
        
        if train_split > 1:
            raise ValueError(f"train_split is too large. Can be at most 1. Actual: {train_split}")

        # generate the samples
        data, labels = self.generate_samples(no_samples = no_samples)
        
        # put them in a dataset with the correct dtype
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(data),
            torch.from_numpy(labels.astype(np.int64))
        )

        if train_split == 0:
            # and make a nice dataloader out of it
            loader = torch.utils.data.DataLoader(
                dataset = ds,
                batch_size = batch_size,
                shuffle = shuffle,
                drop_last = True,
                num_workers = workers,
                pin_memory = True,
                prefetch_factor = prefetch
            )
        else:
            # split the dataset if specified
            train, test = torch.utils.data.random_split(
                ds,
                [train_split, 1 - train_split],
                self.torch_rng
            )

            # and again Dataloader
            train_loader = torch.utils.data.DataLoader(
                dataset = train,
                batch_size = batch_size,
                shuffle = shuffle,
                drop_last = True,
                num_workers = workers,
                pin_memory = True,
                prefetch_factor = prefetch
            )
            test_loader = torch.utils.data.DataLoader(
                dataset = test,
                batch_size = batch_size,
                shuffle = shuffle,
                drop_last = True,
                num_workers = workers,
                pin_memory = True,
                prefetch_factor = prefetch
            )
            loader = (train_loader, test_loader)


        return loader

def vis(
        sample: np.ndarray, 
        label: int = 0
    ) -> None:
    plt.spy(
        sample.T,
        marker = ".",
        markersize = 10,
        aspect = "auto"
    )
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.title("Label: " + str(label))