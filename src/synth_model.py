from imports import torch
from imports import snntorch as snn
from imports import functional
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
            [self.cur_steps, x.shape[1], 10], 
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
            self._build = True
        

    def eval(
        self,
        data: torch.utils.data.DataLoader
    ) -> tuple[torch.Tensor]:
        
        # check if model has been build already 
        if not self._build:
            self.build_vaules(next(iter(data)))
            self._build = True


    ######################################
    ### DEFINITION OF HELPER FUNCTIONS ###
    ######################################

    def build_vaules(
        self,
        x: torch.Tensor
    ) -> None:
        """
        Function needs to be called before starting to train the model.
        It sets and infers values needed for training.
        """

        self._time_steps = x.shape[0]


    def _init_tensors__(
            self
    ) -> None:
        self.cur_steps = -1
        
        if self._record:
            self.rec_spk1 = torch.zeros(
                [
                    self._time_steps,
                    *self.config["layer1"]
                ],
                dtype = torch.float32,
                device = "cpu",
                requires_grad = False
            )

            self.rec_spk2 = torch.zeros(
                [
                    self._time_steps,
                    *self.config["layer2"]
                ],
                dtype = torch.float32,
                device = "cpu",
                requires_grad = False
            )

            self.rec_spk3 = torch.zeros(
                [
                    self._time_steps,
                    *self.config["layer3"]
                ],
                dtype = torch.float32,
                device = "cpu",
                requires_grad = False
            )