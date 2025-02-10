import torch
import torch.utils.data
import snntorch
import snntorch.spikegen
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# subclassing torch.nn.Module
class SNN(torch.nn.Module):
    def __init__(
            self, 
            layers: list = [784,1024,26], 
            beta: torch.Tensor = torch.tensor(0.8), 
            num_steps: int = 100,
            threshold: float = 0.01,
            tau: int = 5,
            batch_size: int = 1024
        ) -> None:

        super().__init__()

        self.num_steps = num_steps # number of timesteps within which spikes can happen, for latency coding one per sequence
        self.threshold = threshold # membrane threshold at which neurons spike
        self.tau = tau # time constant
        self.num_layer = len(layers) - 1 # input layer not counted for further processes
        self.out = layers[-1] # declare output layer
        self.batch = batch_size
        
        # preparation for outsourcing to GPU
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # raise error if layers are not exactly three
        assert len(layers) == 3, 'currently this supports only 1 hidden and one output layer'

        # initialise connections and neurons
        self.fc1 = torch.nn.Linear(layers[0], layers[1])
        self.lif1 = snntorch.Leaky(beta = beta)
        self.fc2 = torch.nn.Linear(layers[1], layers[2])
        self.lif2 = snntorch.Leaky(beta = beta)
        
        # send to GPU
        self.to(self.device)
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # initialise hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # record the final layer
        mem2_recl = []
        spk2_recl = []
        
        # forward and integrate spikes throughout network
        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # record membrane potential and spikes of last layer
            mem2_recl.append(mem2)
            spk2_recl.append(spk2)
      
        
        # return spk2_rec, mem2_rec
        return torch.stack(spk2_recl, dim = 0).to(self.device), torch.stack(mem2_recl, dim = 0).to(self.device)


    def train_test_loop(
            self, 
            train: Any = None, 
            test: Any = None, 
            epochs: int = 25
        ) -> tuple[tuple[list, list], tuple[list, list]]:

        # initialse lists for recording accuracy and loss per each forward pass
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []

        # initialse lists for averagig accuracy and loss per epoch
        train_accs = []
        train_losss = []
        test_accs = []
        test_losss = []

        for epoch in range(epochs):

            # TRAIN LOOP
            for x, target in tqdm(train):
                target = target.to(self.device)

                # latency encoding of inputs
                x = snntorch.spikegen.latency(
                    data = x, 
                    num_steps = self.num_steps, 
                    threshold = self.threshold,
                    tau = self.tau, 
                    clip = True, 
                    normalize = True, 
                    linear = True
                ).to(self.device)
                
                # forward pass
                self.train()
                spk_rec, _ = self(x)

                # calculate loss
                loss_val = self.loss(spk_rec, target)

                # metrics
                train_loss.append(loss_val.item())
                train_acc.append(snntorch.functional.acc.accuracy_temporal(spk_rec, target))

                # clear prev gradients, calculate gradients, weight update
                self.optimiser.zero_grad()
                loss_val.backward()
                self.optimiser.step()
            
        
            # TEST LOOP
            with torch.no_grad():
                self.eval()

                for x, target in test:
                    target = target.to(self.device)

                    # latency encoding of inputs
                    x = snntorch.spikegen.latency(
                        data = x, 
                        num_steps = self.num_steps, 
                        threshold = self.threshold,
                        tau = self.tau, 
                        clip = True, 
                        normalize = True, 
                        linear = True
                    ).to(self.device)
                    
                    # forward pass
                    spk_rec, _ = self(x)

                    # calculate loss
                    loss_val = self.loss(spk_rec, target)

                    # metrics
                    test_loss.append(loss_val.item())
                    test_acc.append(snntorch.functional.acc.accuracy_temporal(spk_rec, target))


            #average over accuracies and losses per epoch
            train_accs.append(np.mean(train_acc))
            train_losss.append(np.mean(train_loss))
            test_accs.append(np.mean(test_acc))
            test_losss.append(np.mean(test_loss))

            # print accuracies and losses per epoch
            print(f'Train Losses after Epoch {epoch}: {train_losss[epoch]}')
            print(f'Test Losses after Epoch {epoch}: {test_losss[epoch]}')
            print(f'Train Accs after Epoch {epoch}: {train_accs[epoch]}')
            print(f'Test Accs after Epoch {epoch}: {test_accs[epoch]}')

        # plot accuracies and losses
        self.plot_metrics(
            epochs = epochs,
            train_loss = train_losss, 
            test_loss = test_losss,
            train_acc = train_accs,
            test_acc = test_accs
        )
        
        # return metrics
        return (train_losss, test_losss), (train_accs, test_accs)
                    

    def set_optimiser(self, optim: Any = torch.optim.Adam, learning_rate: float = 0.001) -> None:
        self.optimiser = optim(self.parameters(), lr = learning_rate)


    def set_loss(self, loss: Any = torch.nn.CrossEntropyLoss()) -> None:
        self.loss = loss


    def plot_metrics(self, epochs: int, train_loss: list, test_loss: list, train_acc: list, test_acc: list) -> None:
        
        fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True)
        
        fig.set_size_inches(10, 5)
        
        # subplot accuracies
        ax[0].plot(range(epochs), train_acc, label = "Training")
        ax[0].plot(range(epochs), test_acc, label = "Testing")
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy in %')
        ax[0].set_title('Accuracy')
        ax[0].legend()
        
        # subplot losses
        ax[1].plot(range(epochs), train_loss, label = "Training")
        ax[1].plot(range(epochs), test_loss, label = "Testing")
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Loss')
        ax[1].legend()

        fig.suptitle(f'Metrics: tau={self.tau}, thresh={self.threshold}')
        plt.show()