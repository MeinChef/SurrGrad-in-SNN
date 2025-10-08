from imports import torch
from imports import snntorch as snn
from imports import functional
from imports import tqdm
from misc import resolve_gradient, resolve_acc, resolve_loss, resolve_optim

class SynthModel(torch.nn.Module):
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
            in_features = self.config["features"],
            out_features = self.config["neurons_hidden_1"],
            device = self.device
        )
        self.neuron1 = snn.Leaky(
            beta = config["neuron_beta"], 
            spike_grad = surrogate, 
            init_hidden = False
        )

        self.con2 = torch.nn.Linear(
            in_features = self.config["neurons_hidden_1"],
            out_features = self.config["neurons_hidden_2"],
            device = self.device
        )
        self.neuron2 = snn.Leaky(
            beta = config["neuron_beta"], 
            spike_grad = surrogate,
            init_hidden = False
        )

        self.con3 = torch.nn.Linear(
            in_features = self.config["neurons_hidden_2"],
            out_features = self.config["neurons_out"],
            device = self.device
        )
        self.neuron3 = snn.Leaky(
            beta = config["neuron_beta"], 
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

        self._time_steps = config["time_steps"]
        self._epochs = config["epochs"]
        self._partial_train = config["partial_training"]
        self._partial_test  = config["partial_testing"]

        self._record = False
        self._samples = config["samples"]

        # send to gpu
        self.to(device = self.device)

    ####################################
    ### DEFINITION OF MAIN FUNCTIONS ###
    ####################################


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        mem3 = self.neuron3.reset_mem()

        out = torch.empty(
            [
                self._time_steps,
                self.config["neurons_out"]
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

            if self._record:
                self.rec_spk1[step] = spk1
                self.rec_spk2[step] = spk2
                self.rec_spk3[step] = spk3

        return out

    def fit(
        self,
        data: torch.utils.data.DataLoader
    ) -> tuple[torch.Tensor]:
        
        if not self._build:
            self.build_vaules(next(iter(data)))

        # pre-define variables
        loss_hist = []
        acc_hist  = []

        # set model in training mode
        self.train()

        # training loop
        for i, (x, target) in tqdm.tqdm(enumerate(data)):
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
            loss.backward(retain_graph = True)
            self.optim.step()

            # TODO: record loss/accuracy during training
            # TODO: dump list regularly to file
            loss_hist.append(loss.item())
            acc_hist.append(acc)

            if i == self.config["partial_training"]:
                break
        
        torch.cuda.empty_cache()
        
        return loss_hist, acc_hist


    def evaluate(
        self,
        data: torch.utils.data.DataLoader
    ) -> tuple[torch.Tensor]:
        
        # check if model has been build already 
        if not self._build:
            self.build_vaules(next(iter(data)))
            
        # pre-define variables
        loss_hist = []
        acc_hist  = []
        rec_list  = [[], [], []]

        # set model in evaulating mode
        self.eval()


        # test loop
        with torch.no_grad():
            for i, (x, target) in tqdm.tqdm(enumerate(data)):
                # move tensors to device
                x = x.to(self.device)
                target = target.to(self.device)

                # make prediction
                if self._record:
                    # create a mask for the hidden layer recordings
                    mask = self.create_mask(target)
                    if torch.is_tensor(mask):
                        # separate the recordings to the individual layers
                        rec_list[0].append(self.rec_spk1[:, mask])
                        rec_list[1].append(self.rec_spk2[:, mask])
                        rec_list[2].append(self.rec_spk3[:, mask])
                else:
                    pred = self.forward(x)

                # loss and accuracy calculations
                loss = self.lossfn(pred, target)
                acc = self.acc(pred, target)

                # TODO: record loss/accuracy during training
                # TODO: dump list regularly to file
                loss_hist.append(loss.item())
                acc_hist.append(acc)

                if i == self.config["partial_testing"]:
                    break
            
        torch.cuda.empty_cache()
        
        return loss_hist, acc_hist

    ######################################
    ### DEFINITION OF HELPER FUNCTIONS ###
    ######################################

    def create_mask(
            self,
            target: torch.Tensor,
    ) -> torch.Tensor | bool:

        '''
        Function to create a mask for the hidden layer recordings. Returns mask if there is still something to be recorded.
        Otherwise returns False.

        :param target: tensor - the target labels for the minibatch
        :type target: torch.Tensor, required
        :return: mask - a boolean mask for the hidden layer recordings or False if there is no value in this batch to be recorded
        :rtype: torch.Tensor
        '''

        # TODO:
        # add param cls: int = None
        # and return only the mask for the specified class - since
        # as of now there are no ways to determine in retrospective which recording
        # belongs to wich class


        mask = torch.zeros_like(
            target, 
            dtype = torch.bool, 
            device = torch.device("cpu")
        )

        if (self._counter == 0).all():
            # all classes have been recorded the specified amount of times
            # no need to create a mask
            return False

        indices = []
        for cls in range(self.config["neurons_out"]):
            if self._counter[cls] == 0:
                # class has been already recorded
                continue
            else:
                # get indices for the class
                # since target is a 1D tensor, we will get one 1D tensor with indices
                idxs = torch.nonzero(target == cls, as_tuple = True)[0]
                idxs = idxs[:self._counter[cls]]
                
                # substract the count of found samples
                self._counter[cls] -= len(idxs)
                indices.append(idxs)

        return 


    def build_vaules(
        self,
        x: torch.Tensor
    ) -> None:
        """
        Function needs to be called before starting to train the model.
        It sets and infers values needed for training.
        """

        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        mem3 = self.neuron3.reset_mem()

        # layer 1
        cur1 = self.con1(x[0])
        spk1, mem1 = self.neuron1(cur1, mem1)

        # layer 2
        cur2 = self.con2(spk1)
        spk2, mem2 = self.neuron2(cur2, mem2)

        # layer 3
        cur3 = self.con3(spk2)
        spk3, mem3 = self.neuron3(cur3, mem3)

        # record the shapes
        self._layer1_shape = spk1.shape
        self._layer2_shape = spk2.shape
        self._layer3_shape = spk3.shape

        if self._record:
            self._init_tensors__()

        self._build = True



    def _init_tensors__(
            self,
            layer1_shape: tuple,
            layer2_shape: tuple,
            layer3_shape: tuple
    ) -> None:
        
        self.rec_spk1 = torch.zeros(
            [
                self._time_steps,
                *layer1_shape
            ],
            dtype = torch.float32,
            device = "cpu",
            requires_grad = False
        )

        self.rec_spk2 = torch.zeros(
            [
                self._time_steps,
                *layer2_shape
            ],
            dtype = torch.float32,
            device = "cpu",
            requires_grad = False
        )

        self.rec_spk3 = torch.zeros(
            [
                self._time_steps,
                *layer3_shape
            ],
            dtype = torch.float32,
            device = "cpu",
            requires_grad = False
        )

        self._counter = torch.full(
            size = (self.config["neurons_out"],),
            fill_value = self._samples,
            dtype = torch.int32,
            device = torch.device("cpu")
        )