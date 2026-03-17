from imports import torch
from imports import snntorch as snn
from imports import tqdm
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
        self._neurons_out = config["neurons_out"]

        self._samples = config["samples"]
        self._build = False

        # send to gpu
        self.to(device = self.device)

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
            x = x.permute(1, 0, -1)

        # pre-allocate the output-tensor
        out = torch.empty(
            [
                self._time_steps,
                x.shape[1],
                self._neurons_out
            ], 
            device = self.device
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

            out[step] = spk3

        return out

    def fit(
        self,
        data: torch.utils.data.DataLoader
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

        # training loop
        for i, (x, target) in tqdm.tqdm(enumerate(data)):
            # check if the training has been already done to the specified amount
            if i == self._partial_train:
                break

            # move tensors to device
            x = x.to(self.device)
            target = target.to(self.device)

            # make prediction
            pred = self.forward(x)

            # loss and accuracy calculations
            loss = self.lossfn(pred, target)
            acc = self.acc(pred, target)

            # weight update
            self.optim.zero_grad()
            # loss.backward(retain_graph = True)
            loss.backward()
            self.optim.step()

            # TODO: dump list regularly to file
            loss_hist.append(loss.item())
            acc_hist.append(acc)
        
        torch.cuda.empty_cache()
        
        return loss_hist, acc_hist


    def evaluate(
        self,
        data: torch.utils.data.DataLoader,
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

                pred = self.forward(x)

                # loss and accuracy calculations
                loss = self.lossfn(pred, target)
                acc = self.acc(pred, target)

                # TODO: record loss/accuracy during training
                # TODO: dump list regularly to file
                loss_hist.append(loss.item())
                acc_hist.append(acc)
            
        torch.cuda.empty_cache()
        
        return loss_hist, acc_hist

    # def build_vaules(
    #     self,
    #     x: torch.Tensor,
    #     batch_first: bool = True
    # ) -> None:
    #     """
    #     Function needs to be called before starting to train the model.
    #     It sets and infers values needed for training.

    #     :param x: A batch, as it would be usually passed through the network
    #     :type x: Tensor

    #     :param batch_first: Whether the first dimension is batch_size (True) or time_steps (False). Default True
    #     :type batch_first: bool, optional

    #     :returns:
    #     :rtype: None
    #     """

    #     mem1 = self.neuron1.reset_mem()
    #     mem2 = self.neuron2.reset_mem()
    #     mem3 = self.neuron3.reset_mem()

    #     x = x.to(self.device)

    #     if batch_first:
    #         # reshape to actually have the time_steps first again
    #         x = x.permute(1, 0, -1)

    #     # layer 1
    #     cur1 = self.con1(x[0])
    #     spk1, mem1 = self.neuron1(cur1, mem1)

    #     # layer 2
    #     cur2 = self.con2(spk1)
    #     spk2, mem2 = self.neuron2(cur2, mem2)

    #     # layer 3
    #     cur3 = self.con3(spk2)
    #     spk3, mem3 = self.neuron3(cur3, mem3)

    #     self._build = True