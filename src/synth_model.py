from imports import torch
from imports import snntorch as snn
from imports import tqdm
from misc import resolve_gradient, resolve_acc, resolve_loss, resolve_optim

class SynthModel(torch.nn.Module):
    def __init__(
        self,
        config: dict,
        record: bool | None = None
    ) -> None:
        
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

        self._time_steps = config["time_steps"]["val"]
        self._epochs = config["epochs"]
        self._partial_train = config["partial_training"]
        self._partial_test  = config["partial_testing"]
        self._neurons_out = config["neurons_out"]

        self._record = record
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

        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        mem3 = self.neuron3.reset_mem()

        if batch_first:
            # reshape to actually have the time_steps first again
            # that makes the for loop later cleaner
            x = x.permute(1, 0, -1)

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
            self.build_vaules(next(iter(data))[0])

        # pre-define variables
        loss_hist = []
        acc_hist  = []

        # set model in training mode
        self.train()

        # training loop
        for i, (x, target) in tqdm.tqdm(enumerate(data)):
            # check if the training has been already done to the specified amount
            if i == self._partial_test:
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

            # TODO: record loss/accuracy during training
            # TODO: dump list regularly to file
            loss_hist.append(loss.item())
            acc_hist.append(acc)
        
        torch.cuda.empty_cache()
        
        return loss_hist, acc_hist


    def evaluate(
        self,
        data: torch.utils.data.DataLoader,
        record_per_class: bool = False
    ) -> tuple[list, list, dict | None]:
        
        
        # check if model has been build already 
        if not self._build:
            self.build_vaules(next(iter(data))[0])
            
        # pre-define variables
        loss_hist = []
        acc_hist  = []
        if self._record:
            if record_per_class:
                rec_dict  = {}
                for cls in range(self._counter.shape[0]):
                    rec_dict[f"class_{cls}"] = [[], [], []]

            else:
                rec_dict = {
                    "class_0": [[], [], []]
                }
        else:
            rec_dict = None
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

                # make prediction
                if self._record:
                    pred = self.forward(x)

                    # create a mask for the hidden layer recordings
                    mask = self.create_mask(
                        target = target,
                        per_class = record_per_class
                    )
                    if torch.is_tensor(mask):
                        if record_per_class:
                            # write the recordings per class
                            for cls in range(self._counter.shape[0]):
                                # separate the recordings to the individual layers
                                rec_dict[f"class_{cls}"][0].append(self.rec_spk1[:, mask[cls]].detach().clone().cpu())
                                rec_dict[f"class_{cls}"][1].append(self.rec_spk2[:, mask[cls]].detach().clone().cpu())
                                rec_dict[f"class_{cls}"][2].append(self.rec_spk3[:, mask[cls]].detach().clone().cpu())
                        else:
                            # and no distinction between classes
                            rec_dict["class_0"][0].append(self.rec_spk1[:, mask].detach().clone().cpu())
                            rec_dict["class_0"][1].append(self.rec_spk2[:, mask].detach().clone().cpu())
                            rec_dict["class_0"][2].append(self.rec_spk3[:, mask].detach().clone().cpu())
                else:
                    pred = self.forward(x)

                # loss and accuracy calculations
                loss = self.lossfn(pred, target)
                acc = self.acc(pred, target)

                # TODO: record loss/accuracy during training
                # TODO: dump list regularly to file
                loss_hist.append(loss.item())
                acc_hist.append(acc)
            
        torch.cuda.empty_cache()
        
        return loss_hist, acc_hist, rec_dict

    ######################################
    ### DEFINITION OF HELPER FUNCTIONS ###
    ######################################

    def create_mask(
        self,
        target: torch.Tensor,
        per_class: bool = True
    ) -> torch.Tensor | bool:

        '''
        Function to create a mask for the hidden layer recordings. Returns mask if there is still something to be recorded.
        Otherwise returns False.

        :param target: tensor - the target labels for the minibatch
        :type target: torch.Tensor, required
        :return: mask - a boolean mask for the hidden layer recordings or False if there is no value in this batch to be recorded
        :rtype: torch.Tensor | bool
        '''

        # sanity checks
        if not self._counter.shape[0] == self._neurons_out:
            raise ValueError(
                "Something went wrong. Shapes of _counter and config['neurons_out'] do not match." + 
                f"Actual:\n_counter: {self._counter.shape}\nneurons_out: {self._neurons_out}"
            )
        
        if (self._counter < 0).any():
            raise ValueError(
                "Something went wrong. Negative _counter encountered." +
                f"Values: {self._counter}"
            )
        
        # all classes have been recorded the specified amount of times
        # no need to create a mask
        if (self._counter == 0).all():
            
            return False
        
        # pre-allocate the mask
        mask = torch.zeros_like(
            target, 
            dtype = torch.bool, 
            device = target.device
        )
        mask = mask.unsqueeze(0).repeat(self._counter.shape[0], 1)

        # loop over every class and save if it is contained in the target tensor
        for cls in range(self._counter.shape[0]):
            if self._counter[cls] == 0:
                # class has been already recorded
                continue
            else:
                # get indices for the class
                # since target is a 1D tensor, we will get one 1D tensor with indices
                idxs = torch.nonzero(target == cls, as_tuple = True)[0]
                # this is legal, even on tensors that are smaller than self._counter[cls]
                # though it feels highly illegal
                idxs = idxs[:self._counter[cls]]
                
                # substract the count of found samples
                self._counter[cls] -= len(idxs)
                # create a mask for this class
                mask[cls][idxs] = True

        # collapse the masks into one tensor if wanted
        if not per_class:
            mask = mask.sum(dim = 0)

        return mask


    def build_vaules(
        self,
        x: torch.Tensor,
        batch_first: bool = True
    ) -> None:
        """
        Function needs to be called before starting to train the model.
        It sets and infers values needed for training.
        """

        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        mem3 = self.neuron3.reset_mem()

        if batch_first:
            # reshape to actually have the time_steps first again
            x = x.permute(1, 0, -1)

        # layer 1
        cur1 = self.con1(x[0])
        spk1, mem1 = self.neuron1(cur1, mem1)

        # layer 2
        cur2 = self.con2(spk1)
        spk2, mem2 = self.neuron2(cur2, mem2)

        # layer 3
        cur3 = self.con3(spk2)
        spk3, mem3 = self.neuron3(cur3, mem3)

        if self._record:
            self._init_tensors__(
                spk1.shape,
                spk2.shape,
                spk3.shape
            )

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
            size = (self._neurons_out,),
            fill_value = self._samples,
            dtype = torch.int32,
            device = torch.device("cpu")
        )