import numpy as np
import matplotlib.pyplot as plt
from typing import Any

import snntorch
import snntorch.functional
import snntorch.spikegen
import snntorch.spikeplot as splt
import snntorch.surrogate
import snntorch.utils

import torch
import torch.utils
import torch.utils.data

from torchvision import datasets, transforms

import snn_model


def get_emnist_letters(
        transform: Any = transforms.ToTensor(), 
        target_transform: Any = transforms.ToTensor(),
        subset: int = None, 
        batch_size: int = 128
    ) -> object:
    '''Function to get the letters of the EMNIST dataset - like MNIST, just with letters'''
    
    # get train subset of data
    train = datasets.EMNIST(
        root = 'data',
        split = 'letters',
        train = True,
        download = True,
        transform = transform,
        target_transform = target_transform
    )

    # get test subset of data
    test = datasets.EMNIST(
        root = 'data',
        split = 'letters',
        train = False,
        download = True,
        transform = transform,
        target_transform = target_transform
    )
    
    # if subset, specify which
    if subset:
        train = snntorch.utils.data_subset(train, subset)
        test  = snntorch.utils.data_subset(test, subset)

    # load train subset of data
    train = torch.utils.data.DataLoader(
        train, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
        prefetch_factor = 5, 
        num_workers = torch.get_num_threads() - 1 # num_workers should be threads (-1, to leave some thread for other programs)
    ) 

    # load test subset of data
    test = torch.utils.data.DataLoader(
        test, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
        prefetch_factor = 5, 
        num_workers = torch.get_num_threads() - 1 # num_workers should be threads (-1, to leave some thread for other programs)
    ) 

    return train, test



if __name__ == "__main__":
    
    subset = None # if not "None" will train with less than the whole dataset, useful for testing the code, but training/testing done on whole dataset
    batch_size = 2048
    epochs = 5

    steps = 100 # simulation time steps
    tau = 5 # time constant in ms
    threshold = 0.01 # membrane threshold at which the neurons fire
    delta_t = torch.tensor(1)
    beta = torch.exp(-delta_t / torch.tensor(tau)) # no idea why this is the correct beta current, but the documentation said so


    num_neuro_in = 784 # input features, 784 = 28*28, pixels of letter images
    num_neuro_hid = 1024 # hidden layer number of neurons
    num_classes = 26 # also neurons out, number of letters in the alphabet
    
    # preparation for outsourcing to GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # basic preprocessing
    transf = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            transforms.Lambda(lambda x: x.reshape(num_neuro_in))
    ])

    # rescales targets from indices 1-26 to indices 0-25 (otherwise error)
    target_transf = transforms.Compose([
        transforms.Lambda(lambda x: x -1 )
    ])
    
    # get data
    train, test = get_emnist_letters(
        transform = transf, 
        target_transform = target_transf,
        subset = subset, 
        batch_size = batch_size
    )
    
    # declare model from class
    model = snn_model.SNN(
        layers = [num_neuro_in, num_neuro_hid, num_classes],
        beta = beta,
        num_steps = steps,
        threshold = threshold,
        tau = tau,
        batch_size = batch_size
    )

    # set model optimiser and loss
    # standard optimiser of this function is Adam
    # temporal loss used for latency coding: the class with the first spike is being chosen
    model.set_optimiser()
    model.set_loss(snntorch.functional.loss.ce_temporal_loss())
    
    # train/test model and return accuracy and loss
    loss, acc = model.train_test_loop(train, test, epochs)
