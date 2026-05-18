from imports import torch
from imports import snntorch as snn
from imports import tqdm
from imports import Literal, Callable
from misc import resolve_gradient, resolve_acc, resolve_loss, resolve_optim

class SynthModel(torch.nn.Module):
    def __init__(
        self,
        config: dict,
    ) -> None:

        """
        A model for learning the spike time and rate encoding of data.
        Expects data to be presented in the form 

        :param config: A dictionary with keys for setting up the SNN.
        :type config: dict, required

        :param record: Whether to record the hidden layers during testing. Can be bool or None. Default is None
        :type record: bool | None, optional
        """
        
        super().__init__()

        # check backend
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else: 
            self.device = torch.device("cpu")

        # resolve gradient
        surrogate = resolve_gradient(config = config["surrogate"])

        ###########################
        ### DEFINITION OF MODEL ###
        ###########################
        # layer 1
        self.con1 = torch.nn.Linear(
            in_features = config["features"]["val"],
            out_features = config["neurons_hidden_1"],
            device = self.device
        )
        torch.nn.init.xavier_uniform_(self.con1.weight)
        self.neuron1 = snn.Leaky(
            beta = config["neuron_beta"], 
            spike_grad = surrogate, 
            init_hidden = False
        )

        # layer 2
        self.con2 = torch.nn.Linear(
            in_features = config["neurons_hidden_1"],
            out_features = config["neurons_hidden_2"],
            device = self.device
        )
        torch.nn.init.xavier_uniform_(self.con2.weight)
        self.neuron2 = snn.Leaky(
            beta = config["neuron_beta"], 
            spike_grad = surrogate,
            init_hidden = False
        )

        # layer 3 / output
        self.con3 = torch.nn.Linear(
            in_features = config["neurons_hidden_2"],
            out_features = config["neurons_out"],
            device = self.device
        )
        torch.nn.init.xavier_uniform_(self.con3.weight)
        self.neuron3 = snn.Leaky(
            beta = config["neuron_beta"], 
            spike_grad = surrogate,
            init_hidden = False
        )


        # resolve additional bits
        self.lossfn = resolve_loss(config = config["loss"])
        self.acc    = resolve_acc(config = config["accuracy"])
        self.optim  = resolve_optim(
            config  = config["optimiser"], 
            params  = self.parameters()
        )

        # save config to class
        self._time_steps = config["time_steps"]["val"]
        self._epochs = config["epochs"]
        self._partial_train = config["partial_training"]
        self._partial_test  = config["partial_testing"]
        self._move_fraction = config["move_fraction"]

        # neuron features
        self._in_first = config["features"]["val"]
        self._out_first = config["neurons_hidden_1"]
        self._in_second = self._out_first
        self._out_second = config["neurons_hidden_2"]
        self._in_third = self._out_second
        self._out_third = config["neurons_out"]
        self._neurons_out = self._out_third
        self._return_spk = config["return_spk"]

        # predefine output tensor for forward
        self._forward_output_buffer = None
        self._samples = config["samples"]
        self._build = False

        # send to gpu
        self.to(device = self.device)

        # Let cuDNN find optimal algorithms
        torch.backends.cudnn.benchmark = True  

    ####################################
    ### DEFINITION OF MAIN FUNCTIONS ###
    ####################################


    def forward(
        self,
        x: torch.Tensor,
        batch_first: bool = True
    ) -> torch.Tensor:

        """
        The forward function of the network. Passes a single batch through the network. 

        :param x: Data to pass through the network.
        :type x: Tensor, required

        :param batch_first: Whether the first dimension of the input tensor is the batch dimension (True) or time steps (False).
        :type batch_first: bool, optional

        :return: Output of the last layer
        :rtype: Tensor
        """

        # setup
        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        mem3 = self.neuron3.reset_mem()

        if batch_first:
            # reshape to actually have the time_steps first again
            # that makes the for loop later cleaner
            x = x.permute(1, 0, -1).contiguous()

        # pre-allocate the output-tensor
        out = torch.empty(
            [
                self._time_steps,
                x.shape[1],
                self._neurons_out
            ], 
            device = self.device,
            dtype = x.dtype
        )


        for step in range(self._time_steps):  
            # layer 1
            cur1 = self.con1(x[step])
            spk1, mem1 = self.neuron1(cur1, mem1)

            # layer 2
            cur2 = self.con2(spk1)
            spk2, mem2 = self.neuron2(cur2, mem2)

            # layer 3
            cur3 = self.con3(spk2)
            spk3, mem3 = self.neuron3(cur3, mem3)

            if self._return_spk:
                out[step] = spk3
            else:
                out[step] = mem3

        return out

    def fit(
        self,
        data: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list, list]:
        
        """
        Function for fitting (training) the network. 
        The Dataloader should contain the data and the labels.

        :param data: Dataloader. Should contain a tuple with (data, label).
        :type data: torch DataLoader, required

        :return: Lists containing the accuracy and loss during the training
        :rtype: tuple[list, list]
        """
        
        # # check if model has been build already
        # if not self._build:
        #     self.build_vaules(next(iter(data))[0])

        # pre-define variables
        loss_hist = []
        acc_hist  = []

        # set model in training mode
        self.train()

        # just try and see what happens
        # x, train = next(iter(data))

        # training loop
        for i, (x, target) in tqdm.tqdm(enumerate(data)):
            # check if the training has been already done to the specified amount
            if i == self._partial_train:
                break

            # move tensors to device
            if x.device != self.device:
                x = x.to(self.device)
            if target.device != self.device:
                target = target.to(self.device)

            # make prediction
            pred = self.forward(x, batch_first = False)

            # loss and accuracy calculations
            loss = self.lossfn(pred, target)
            if loss.isnan().any():
                print("something's fishy")
            acc = self.acc(pred, target)

            # weight update
            self.optim.zero_grad()
            # loss.backward(retain_graph = True)
            loss.backward()
            self.optim.step()

            # TODO: dump list regularly to file
            loss_hist.append(loss.item())
            acc_hist.append(acc)
            
        return loss_hist, acc_hist


    def evaluate(
        self,
        data: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list, list]:
        """
        Function for evaluating (testing) the network. 
        The Dataloader should contain the data and the labels.

        :param data: Dataloader. Should contain a tuple with (data, label).
        :type data: torch DataLoader, required

        :return: Lists with accuracy and lost during evaluation.\n
                If hidden layers are recorded, also return a dictionary with the recordings.\n
                If record_per_class is true, the dictionary contains more than one key.
                If record_per_class is false, the dictionary only contains the key 'class_0'
        :rtype: tuple[list, list, dict | None]
        """
        
        # # check if model has been build already 
        # if not self._build:
        #     self.build_vaules(next(iter(data))[0])
            
        # pre-define variables
        loss_hist = []
        acc_hist  = []
        # set model in evaulating mode
        self.eval()


        # test loop
        with torch.no_grad():
            for i, (x, target) in tqdm.tqdm(enumerate(data)):
                # check if the training has been already done to the specified amount
                if i == self._partial_test:
                    break

                # move tensors to device
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.forward(x, batch_first = False)

                # loss and accuracy calculations
                loss = self.lossfn(pred, target)
                acc = self.acc(pred, target)

                # TODO: record loss/accuracy during training
                # TODO: dump list regularly to file
                loss_hist.append(loss.item())
                acc_hist.append(acc)
            
        torch.cuda.empty_cache()
        
        return loss_hist, acc_hist

    #########################
    ### Augmented Forward ###
    #########################

    def _forward_first_layer(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        # setup
        mem = self.neuron1.reset_mem()

        # pre-allocate the output-tensor
        out = torch.zeros(
            [
                self._time_steps,
                x.shape[1],             # batch size
                self._out_first
            ],
            device = self.device
        )

        # loop over time
        for step in range(self._time_steps):
            cur = self.con1(x[step])
            out[step], mem = self.neuron1(cur, mem)

        return out
    
    def _forward_second_layer(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        # setup
        mem = self.neuron2.reset_mem()

        # pre-allocate the output-tensor
        out = torch.zeros(
            [
                self._time_steps,
                x.shape[1],             # batch size
                self._out_second
            ],
            device = self.device
        )

        # loop over time
        for step in range(self._time_steps):
            cur = self.con2(x[step])
            out[step], mem = self.neuron2(cur, mem)

        return out
    
    def _forward_third_layer(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        # setup
        mem = self.neuron1.reset_mem()

        # pre-allocate the output-tensor
        out = torch.empty(
            [
                self._time_steps,
                x.shape[1],             # batch size
                self._out_third
            ],
            device = self.device
        )

        # loop over time
        for step in range(self._time_steps):
            cur = self.con3(x[step])
            if self._return_spk:
                _, out[step] = self.neuron3(cur, mem)
            else:
                out[step], _ = self.neuron3(cur, mem)

        return out
    
    def _jitter_layer_out(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()
    
    def _shuffle_layer_out(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        out = x.clone()
        # get indices where spikes are, and where they aren't
        spikes = x.nonzero() # shape: spikes, 3 (one col for time, sample in batch, neuron)
        valid_to = torch.nonzero(x == 0)

        for sample in range(x.shape[1]):
            to_spk = valid_to[torch.where(valid_to[:,1] == sample)]
            to_rm = spikes[torch.where(spikes[:,1] == sample)]

            for neuron in range(x.shape[2]):
                to_move = torch.ceil(x[:, sample, neuron].sum() * self._move_fraction)
                
                # get valid positions for neuron, shuffle and select the first couple
                # to simulate random picking
                new_spk = to_spk[torch.where(to_spk[:,2] == neuron)]
                new_spk = new_spk[torch.randperm(new_spk.shape[0])]
                new_spk = new_spk[:to_move]
                out[new_spk] = 1

                # and remove the old spikes
                old_spk = to_rm[torch.where(to_rm[:,2] == neuron)]
                old_spk = old_spk[torch.randperm(old_spk.shape[0])]
                old_spk = old_spk[:to_move]
                out[old_spk] = 0
        
        return out


    def augmented_forward(
        self,
        data: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        augment: Literal["shuffle", "jitter"] | Callable = "jitter",                    # noqa: F821
        only_nth_layer: int | None = None
    ) -> tuple[list, list]:
        
        if augment != "jitter" and augment != "shuffle" and callable(augment):
            raise ValueError(f"Expected 'shuffle' or 'jitter'. Got '{augment}' (Type: {type(augment)}) instead.")
        
        if augment == "jitter":
            augment_fn = self._jitter_layer_out
        elif augment == "shuffle":
            augment_fn = self._shuffle_layer_out
        else:
            augment_fn = augment
        
        loss = []
        acc = []

        self.eval()
        with torch.no_grad():
            for i, (x, target) in tqdm.tqdm(enumerate(data)):

                # move tensors to device
                if x.device != self.device:
                    x = x.to(self.device)
                if target.device != self.device:
                    target = target.to(self.device)

                x = self._forward_first_layer(x)
                if only_nth_layer == 1:
                    x = augment_fn(x)
                
                x = self._forward_second_layer(x)
                if only_nth_layer == 2:
                    x = augment_fn(x)
                
                x = self._forward_third_layer(x)
                if only_nth_layer == 3:
                    x = augment_fn(x)
                
                loss.append(self.lossfn(x, target).item)
                acc.append(self.acc(x, target))

        return loss, acc