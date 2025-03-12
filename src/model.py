import torch.utils.data.dataloader
from imports import torch
from imports import snntorch as snn
from imports import numpy as np
import misc

class Model(torch.nn.Module):
    def __init__(
            self, 
            config: dict
        ) -> None:

        super().__init__()
        
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else self.device
        
        self.surrogate = misc.resolve_gradient(config = self.config)

        # the actual network
        self.connect1 = torch.nn.Conv2d(2, 12, 5)
        self.connect2 = torch.nn.MaxPool2d(2)
        self.neuron1 = snn.Leaky(
                beta = config['beta'], 
                spike_grad = self.surrogate, 
                init_hidden = False
            )
        
        self.connect3 = torch.nn.Conv2d(12, 32, 5)
        self.connect4 = torch.nn.MaxPool2d(2)
        self.neuron2 = snn.Leaky(
                beta = config['beta'], 
                spike_grad = self.surrogate, 
                init_hidden = False
            )
        
        self.connect5 = torch.nn.Flatten()
        self.connect6 = torch.nn.Linear(32*5*5, 10)
        self.neuron3 = snn.Leaky(
                beta = config['beta'], 
                spike_grad = self.surrogate, 
                init_hidden = False
            )
        
        # send to gpu
        self.to(device = self.device)

    def forward(
            self, 
            x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        
        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        mem3 = self.neuron3.reset_mem()

        # x.shape[0] is the number of time_steps within the minibatch. varies.
        time_steps = x.shape[0]

        # x.shape[1] is the minibatch-size
        # 10, because there are 10 classes to predict
        out = torch.empty(
            [time_steps, x.shape[1], 10], 
            device = self.device
        )

        if self.config["DEBUG"]:
            print("Size of x:", x.shape)
            print("Size of mem1:", mem1.shape)
            print("Size of mem2:", mem2.shape)
            print("Size of mem3:", mem3.shape)

        if self.config["record_hidden"]:
            
            self.config["layer1"][0] = time_steps
            self.config["layer2"][0] = time_steps
            self.config["layer3"][0] = time_steps

            rec_spk1 = torch.empty(
                self.config["layer1"],
                dtype = torch.float32,
                # device = self.device
                device = torch.device("cpu")
            )

            rec_spk2 = torch.empty(
                self.config["layer2"],
                dtype = torch.float32,
                # device = self.device
                device = torch.device("cpu")
            )

            rec_spk3 = torch.empty(
                self.config["layer3"],
                dtype = torch.float32,
                # device = self.device
                device = torch.device("cpu")
            )


        for step in range(time_steps):
            cur1 = self.connect1(x[step])
            cur2 = self.connect2(cur1)
            spk1, mem1 = self.neuron1(cur2, mem1)
            cur3 = self.connect3(spk1)
            cur4 = self.connect4(cur3)
            spk2, mem2 = self.neuron2(cur4, mem2)
            cur5 = self.connect5(spk2)
            cur6 = self.connect6(cur5)
            spk3, mem3 = self.neuron3(cur6, mem3)

            out[step] = spk3

            if self.config["record_hidden"]:
                rec_spk1[step] = spk1
                rec_spk2[step] = spk2
                rec_spk3[step] = spk3
                

        if self.config["DEBUG"]:
            print("Size of mem1:", mem1.shape)
            print("Size of mem2:", mem2.shape)
            print("Size of mem3:", mem3.shape)

            print("Size of spk1:", spk1.shape)
            print("Size of spk2:", spk2.shape)
            print("Size of spk3:", spk3.shape)

            print("Type of spk:", spk1.dtype)
            print("Type of mem:", mem1.dtype)

            print("Location of spk:", spk1.get_device())
            print("location of mem:", mem1.get_device())

        # return the hidden recording, if so desired
        if self.config["record_hidden"]:
            return out, (rec_spk1, rec_spk2, rec_spk3)
        
        return out


    def train_loop(
            self, 
            data: torch.utils.data.DataLoader
        ) -> int:

        self.expand_config(data)

        for epoch in range(self.config["epochs"]):
            for x, target in data:
                x = x.to(self.device)
                
                if self.config["DEBUG"]:
                    print("Type of Data:", x.dtype, "\nGPU:", x.get_device())

                self.forward(x)
                return 0
            

    def expand_config(
            self, 
            data: torch.utils.data.DataLoader
        ) -> None:
        '''
        Generates the config-values for the output shape of the individual layers in the network. Necessary for spike recording.
        ### Args:
        data: Dataloader
        '''

        x, _ =  next(iter(data))
        x = x.to(self.device)

        # make one forward pass, since the shapes are only obvious 
        # once the forward pass completed
        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        mem3 = self.neuron3.reset_mem()

        # spk and mem have exactly the same shape after each layer
        cur = self.connect1(x[0])
        cur = self.connect2(cur)
        spk, mem1 = self.neuron1(cur, mem1)
        cur = self.connect3(spk)
        cur = self.connect4(cur)
        spk, mem2 = self.neuron2(cur, mem2)
        cur = self.connect5(spk)
        cur = self.connect6(cur)
        spk, mem3 = self.neuron3(cur, mem3)

        # read out the shape
        self.config["layer1"] = list(mem1.shape)
        self.config["layer2"] = list(mem2.shape)
        self.config["layer3"] = list(mem3.shape)

        # insert value 0 at index 0, for overwriting with time_steps in forward
        self.config["layer1"].insert(0, 0)
        self.config["layer2"].insert(0, 0)
        self.config["layer3"].insert(0, 0)