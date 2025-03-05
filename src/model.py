from imports import torch
from imports import snntorch as snn
import misc

class SNN(torch.nn.Module):
    def __init__(self, config: dict) -> None:

        super().__init__()
        
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        self.surrogate = misc.resolve_gradient(config = self.config)



        # the actual network
        self.connect1 = torch.nn.Conv2d(2, 12, 5)
        self.connect2 = torch.nn.MaxPool2d(2)
        self.neuron1 = snn.Leaky(
                beta = config['beta'], 
                spike_grad = self.surrogate, 
                init_hidden = True
            )
        
        self.connect3 = torch.nn.Conv2d(12, 32, 5)
        self.connect4 = torch.nn.MaxPool2d(2)
        self.neuron2 = snn.Leaky(
                beta = config['beta'], 
                spike_grad = self.surrogate, 
                init_hidden = True
            )
        
        self.connect5 = torch.nn.Flatten()
        self.connect6 = torch.nn.Linear(32*5*5, 10)
        self.neuron3 = snn.Leaky(
                beta = config['beta'], 
                spike_grad = self.surrogate, 
                init_hidden = True
            )
        
        # send to gpu
        self.to(device = self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        pass