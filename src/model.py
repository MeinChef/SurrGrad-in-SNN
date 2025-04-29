from imports import Callable
from imports import torch
from imports import snntorch as snn
from imports import functional # snntorch.functional
from imports import surrogate # snntorch.surrogate
import misc

class Model(torch.nn.Module):
    def __init__(
        self, 
        config: dict
    ) -> None:
        '''
        Constructor of the Model class. Config is specified in config.yml

        :param config: Dictionary containing model configuration
        :type config: dictionary, required
        '''
        super().__init__()
        
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else: 
            self.device = torch.device("cpu")

        # the actual network
        self.connect1 = torch.nn.Conv2d(2, 12, 5)
        self.connect2 = torch.nn.MaxPool2d(2)
        self.neuron1 = snn.Leaky(
            beta = config["neuron"]["beta"], 
            spike_grad = self.surrogate, 
            init_hidden = False
        )
        
        self.connect3 = torch.nn.Conv2d(12, 32, 5)
        self.connect4 = torch.nn.MaxPool2d(2)
        self.neuron2 = snn.Leaky(
            beta = config["neuron"]["beta"], 
            spike_grad = self.surrogate, 
            init_hidden = False
        )
        
        self.connect5 = torch.nn.Flatten()
        self.connect6 = torch.nn.Linear(32*5*5, 10)
        self.neuron3 = snn.Leaky(
            beta = config["neuron"]["beta"], 
            spike_grad = self.surrogate, 
            init_hidden = False
        )
        
        # send to gpu
        self.to(device = self.device)

        # use config to set essentials
        self.surrogate = misc.resolve_gradient(config = self.config["surrogate"])
        self.loss = misc.resolve_loss(config = self.config["loss"])
        self.acc = misc.resolve_acc(config = self.config["accuracy"])
        self.optim = misc.resolve_optim(config = self.config["optimiser"], params = self.parameters())



    ##########################
    #### Setter functions ####
    ##########################
    
    def set_surrogate(
        self,
        surrogate: Callable = surrogate.fast_sigmoid(slope = 25)
    ): 
        '''
        Function to set the surrogate gradient to be then used internally.
        
        :param surrogate: A Class, whose caller takes (spk_out, targets)
        :type surrogate: Callable, required 
        '''
        self.surrogate = surrogate
    
    def set_loss(
        self,
        loss: Callable = functional.loss.mse_temporal_loss()
    ) -> None:
        '''
        Function to set the loss function to be then used internally.
        
        :param loss: A Class, whose caller takes (spk_out, targets)
        :type loss: Callable, required 
        '''
        self.loss = loss


    def set_acc(
        self,
        acc: Callable = functional.acc.accuracy_temporal
    ) -> None:
        '''
        Function to set the accuarcy function to be then used internally.
        
        :param acc: A class or function, whose caller receives (spk_out, targets)
        :type acc: Callable, required
        '''
        self.acc = acc

    def set_optim(
        self,
        optim: Callable
    ) -> None:
        '''
        Function to set the Optimiser to be then used internally

        :param optim: an already initalised Optimiser
        :type optim: Callable, required
        '''
        self.optim = optim


    ########################
    #### Main functions ####
    ########################

    def forward(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the Model. Passes x through all layers, returns either a tuple[tensor, tensor] or a tuple[tensor].
        Depending on config["record_hidden"], set in config.yml
        
        :param x: tensor - a minibatch with the dimensions [time_steps, minibatch_size, 2, 34, 34]
        :type x: torch.Tensor, required
        :return: tuple[tensor, tensor] or tuple[tensor] - depending on config["record_hidden"]
        :rtype: tuple
        '''
        
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

        # initialise arrays for recording the hidden layers
        if self.config["record_hidden"]:

            # on gpu might be a tiny bit faster, but idk if that's worth it
            rec_spk1 = torch.empty(
                [time_steps, *self.config["layer1"]],
                dtype = torch.float32,
                # device = self.device,
                device = torch.device("cpu"),
                requires_grad = False
            )

            rec_spk2 = torch.empty(
                [time_steps, *self.config["layer2"]],
                dtype = torch.float32,
                # device = self.device,
                device = torch.device("cpu"),
                requires_grad = False
            )

            rec_spk3 = torch.empty(
                [time_steps, *self.config["layer3"]],
                dtype = torch.float32,
                # device = self.device,
                device = torch.device("cpu"),
                requires_grad = False
            )

        # the actual forward pass
        for step in range(time_steps):
            cur = self.connect1(x[step])
            cur = self.connect2(cur)
            spk1, mem1 = self.neuron1(cur, mem1)
            cur = self.connect3(spk1)
            cur = self.connect4(cur)
            spk2, mem2 = self.neuron2(cur, mem2)
            cur = self.connect5(spk2)
            cur = self.connect6(cur)
            spk3, mem3 = self.neuron3(cur, mem3)

            out[step] = spk3

            # recording
            if self.config["record_hidden"]:
                rec_spk1[step] = spk1.detach().cpu()
                rec_spk2[step] = spk2.detach().cpu()
                rec_spk3[step] = spk3.detach().cpu()


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
        
        return (out,)


    def train_loop(
        self, 
        data: torch.utils.data.DataLoader
    ) -> tuple[list, list, list]:

        '''
        Function for the training loop over the entire dataset.

        :param data: DataLoader - data for the training
        :type data: torch.utils.data.DataLoader, required
        :return: tuple[list, list, list] - loss history, accuracy history, recording of the hidden layers
        :rtype: tuple
        '''

        self.expand_config(data)
        self.train()

        if self.config["record_train"]:
            self.config["record_hidden"] = True
        elif not self.config["record_train"]:
            self.config["record_hidden"] = False

        loss_hist = []
        acc_hist  = []
        if self.config["record_hidden"]:
            rec_list  = [[],[],[]]

        # training loop
        for i, (x, target) in enumerate(data):
            x = x.to(self.device)
            target = target.to(self.device)
            
            if self.config["DEBUG"]:
                print("Type of Data:", x.dtype, "\non GPU:", x.get_device(), "\nShape of Data:", x.shape)
            
            # differentiate between recording hidden states or not
            if self.config["record_hidden"]:
                x, rec = self.forward(x)

                # separate the recordings to the individual layers
                rec_list[0].append(rec[0])
                rec_list[1].append(rec[1])
                rec_list[2].append(rec[2])

            else:
                x, = self.forward(x)

            # loss and accuracy calculations
            loss = self.loss(x, target)
            acc = self.acc(x, target)

            # weight update
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # TODO: record loss/accuracy during training
            # TODO: dump list regularly to file
            loss_hist.append(loss.item())
            acc_hist.append(acc)

            if self.config["PROGRESS"]:
                print(f"Batch: {i}")
                print("Current Loss during Training:", loss.item())
                print("Current Accuracy during Training:", acc)


            # interrupt training after specified amount of minibatches
            if i == self.config["partial_training"]:
                break

        # TODO: gc.collect()?
        torch.cuda.empty_cache()
        return loss_hist, acc_hist, rec_list
    

    def test_loop(
        self,
        data: torch.utils.data.DataLoader
    ) -> None:
        
        '''
        Function for evaluating the network on the test-part of a dataset.

        :param data: DataLoader - data for the testing
        :type data: torch.utils.data.DataLoader, required
        :return: tuple[list, list, list] - loss history, accuracy history, recording of the hidden layers
        :rtype: tuple
        '''


        if self.config["record_test"]:
            self.config["record_hidden"] = True
        elif not self.config["record_test"]:
            self.config["record_hidden"] = False

        loss_hist = []
        acc_hist  = []
        rec_list  = [[],[],[]]

        # test loop
        with torch.no_grad():
            self.train(False)

            for x, target in data:
                x = x.to(self.device)
                target = target.to(self.device)
                
                if self.config["DEBUG"]:
                    print("Type of Data:", x.dtype, "\non GPU:", x.get_device(), "\nShape of Data:", x.shape)
                
                # differentiate between recording hidden states or not
                if self.config["record_hidden"]:
                    x, rec = self.forward(x)

                    # separate the recordings to the individual layers
                    rec_list[0].append(rec[0])
                    rec_list[1].append(rec[1])
                    rec_list[2].append(rec[2])
        
                else:
                    x, = self.forward(x)

                # loss and accuracy calculations
                loss = self.loss(x, target)
                acc = self.acc(x, target)

                loss_hist.append(loss)
                acc_hist.append(acc)

        return loss_hist, acc_hist, rec_list


    ##########################
    #### Helper functions ####
    ##########################

    def expand_config(
        self, 
        data: torch.utils.data.DataLoader
    ) -> None:
        '''
        Generates the config-values for the output shape of the individual layers in the network. Necessary for spike recording.
       
        :param data: Dataloader - data for the training
        :type data: torch.utils.data.DataLoader, required
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