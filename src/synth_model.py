from imports import torch
from imports import snntorch as snn
from imports import functional
from misc import resolve_gradient, resolve_acc, resolve_loss, resolve_optim

class SynthModel:
    def __init__(
        self,
        config: dict
    ) -> None:
        
        super().__init__()
        self.config = config

        # check backend
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else: 
            self.device = torch.device("cpu")

        # resolve gradient
        surrogate = resolve_gradient(config = self.config["surrogate"])

        ###########################
        ### DEFINITION OF MODEL ###
        ###########################

        self.con1 = torch.nn.Linear(
            in_features = self.config["neurons_in"],
            out_features = self.config["neurons_hidden_1"],
            device = self.device
        )
        self.neuron1 = snn.Leaky(
            beta = config["neuron"]["beta"], 
            spike_grad = surrogate, 
            init_hidden = False
        )

        self.con2 = torch.nn.Linear(
            in_features = self.config["neurons_hidden_1"],
            out_features = self.config["neurons_hidden_2"],
            device = self.device
        )
        self.neuron2 = snn.Leaky(
            beta = config["neuron"]["beta"], 
            spike_grad = surrogate,
            init_hidden = False
        )

        self.con3 = torch.nn.Linear(
            in_features = self.config["neurons_hidden_2"],
            out_features = self.config["neurons_out"],
            device = self.device
        )
        self.neuron3 = snn.Leaky(
            beta = config["neuron"]["beta"], 
            spike_grad = surrogate,
            init_hidden = False
        )


        # resolve additional bits
        self.lossfn = resolve_loss(config = self.config["loss"])
        self.acc    = resolve_acc(config = self.config["accuracy"])
        self.optim  = resolve_optim(
            config  = self.config["optimiser"], 
            params  = self.parameters()
        )



        # send to gpu
        self.to(device = self.device)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        mem3 = self.neuron3.reset_mem()

        for step in range(self.config["time_steps"]):  
            # layer 1
            cur1 = self.con1(x[step])
            spk1, mem1 = self.neuron1(cur1, mem1)

            # layer 2
            cur2 = self.con2(spk1)
            spk2, mem2 = self.neuron2(cur2, mem2)

            # layer 3
            cur3 = self.con3(spk2)
            spk3, mem3 = self.neuron3(cur3, mem3)

        return spk3, mem3