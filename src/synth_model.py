from imports import torch
from imports import snntorch as snn
from imports import tqdm
from misc import resolve_gradient, resolve_acc, resolve_loss, resolve_optim
# from grad import sigmoid

DEBUG = False

class SynthModel(torch.nn.Module):
    def __init__(
        self,
        config: dict,
        record: bool | None = None
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
        # surrogate = resolve_gradient(config = config["surrogate"])
        surrogate = resolve_gradient(config["surrogate"])

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
            threshold = config["neuron_threshold"],
            spike_grad = surrogate, 
            init_hidden = False
        )

        # # layer 2
        self.con2 = torch.nn.Linear(
            in_features = config["neurons_hidden_1"],
            out_features = config["neurons_out"],
            device = self.device
        )
        self.neuron2 = snn.Leaky(
            beta = config["neuron_beta"],
            threshold = config["neuron_threshold"],
            spike_grad = surrogate,
            init_hidden = False
        )

        # layer 3 / output
        # self.con3 = torch.nn.Linear(
        #     in_features = config["neurons_hidden_2"],
        #     out_features = config["neurons_out"],
        #     device = self.device
        # )
        # self.neuron3 = snn.Leaky(
        #     beta = config["neuron_beta"],
        #     threshold = config["neuron_threshold"],
        #     spike_grad = surrogate,
        #     init_hidden = False
        # )


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
        mem_last = self.neuron2.reset_mem()
        # mem3 = self.neuron3.reset_mem()

        if batch_first:
            # reshape to actually have the time_steps first again
            # that makes the for loop later cleaner
            x = x.permute(1, 0, -1)

        # pre-allocate the output-tensor
        out = torch.zeros(
            [
                self._time_steps,
                x.shape[1],
                self._neurons_out
            ], 
            device = self.device
        )

        out_mem = torch.zeros(
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
            # spk2, mem2 = self.neuron2(cur2, mem2)

            # layer 3
            # cur3 = self.con3(spk2)
            # spk3, mem3 = self.neuron3(cur3, mem3)
            spk_last, mem_last = self.neuron2(cur2, mem_last)


             # Store output - check what we're actually storing
            if DEBUG and step == 0:
                print(f"DEBUG step {step}: spk3 range [{spk_last.min():.3f}, {spk_last.max():.3f}]")
                print(f"DEBUG step {step}: mem3 range [{mem_last.min():.3f}, {mem_last.max():.3f}]")


            out[step] = spk_last
            out_mem[step] = mem_last

            if self._record:
                self.rec_spk1[step] = spk1
                # self.rec_spk2[step] = spk2
                self.rec_spk2[step] = spk_last

        if DEBUG:
            print(f"Final output shape: {out.shape}")
            print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    

        return out, out_mem

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
        
        # check if model has been build already
        if not self._build:
            self.build_vaules(next(iter(data))[0])

        # pre-define variables
        loss_hist = []
        acc_hist  = []

        # TEMPORARY: membrane loss
        lossfn_mem = torch.nn.CrossEntropyLoss()

        # set model in training mode
        self.train()

        # training loop
        for i, (x, target) in tqdm.tqdm(enumerate(data)):
        # for i, (x, target) in enumerate(data):

            # check if the training has been already done to the specified amount
            if i == self._partial_train:
                break

            # move tensors to device
            x = x.to(self.device)
            target = target.to(self.device)

            # make prediction
            pred, pred_mem = self.forward(x)

            # basic finite checks (loss may be finite while grads NaN)
            if not torch.isfinite(pred).all():
                print(f"DEBUG: non-finite values in predictions at batch {i}: min/max/nans = {pred.min().item()}/{pred.max().item()}/{pred.isnan().any().item()}")


            # loss and accuracy calculations
            if DEBUG:
                print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
                print(f"Pred sum: {pred.sum():.3f}, Target sum: {target.sum():.3f}")

            loss = self.lossfn(pred, target)
            acc = self.acc(pred, target)

            # TEMPORARY: membrane loss
            loss_val_mem = torch.zeros((1), device = self.device)
            for step in range(self._time_steps):
                loss_val_mem += lossfn_mem(pred_mem[step], target)

            if not torch.isfinite(loss):
                print(f"DEBUG: non-finite loss at batch {i}: {loss}")
                # break early to inspect
                breakpoint()

            if DEBUG:
                # --- DEBUG: inspect grads and parameter changes ---
                # take one parameter (first conv weight) snapshot
                p0 = None
                for p in self.parameters():
                    p0 = p.detach().clone()
                    break

            # weight update
            self.optim.zero_grad()
            # loss.backward(retain_graph = True)
            loss.backward()
            # try:
            #     loss.backward()
            # except RuntimeError as e:
            #     # autograd anomaly should report the op; print helpful diagnostics
            #     print(f"DEBUG: RuntimeError during backward at batch {i}: {e}")
            #     # print a few tensor stats to help locate the issue
            #     for name, tensor in [("x", x), ("pred", pred), ("target", target)]:
            #         print(f"DEBUG: {name} finite: {torch.isfinite(tensor).all().item()} has_nan: {tensor.isnan().any().item()} max: {tensor.max().item()} min: {tensor.min().item()}")
            #     # also check params
            #     for idx, p in enumerate(self.parameters()):
            #         if not torch.isfinite(p).all():
            #             print(f"DEBUG: param {idx} contains non-finite values")
            #     raise

            # loss_val_mem.backward()

            if DEBUG:
                # print grad norm
                total_grad_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        gnorm = p.grad.data.norm().item()
                        total_grad_norm += gnorm**2
                total_grad_norm = total_grad_norm**0.5
                print(f"batch {i} loss={loss.item():.6f} grad_norm={total_grad_norm:.6e}")

            self.optim.step()

            if DEBUG:
                if p0 is not None:
                    p1 = None
                    for p in self.parameters():
                        p1 = p.detach().clone()
                        break
                print("param change norm:", (p1 - p0).norm().item())

            # TODO: dump list regularly to file
            # loss_hist.append(loss.item())
            loss_hist.append(loss_val_mem.item())
            acc_hist.append(acc)
        
        torch.cuda.empty_cache()
        
        return loss_hist, acc_hist


    def evaluate(
        self,
        data: torch.utils.data.DataLoader,
        record_per_class: bool = False
    ) -> tuple[list, list, dict | None]:
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
                    pred, _ = self.forward(x)

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
                                # rec_dict[f"class_{cls}"][2].append(self.rec_spk3[:, mask[cls]].detach().clone().cpu())
                        else:
                            # and no distinction between classes
                            rec_dict["class_0"][0].append(self.rec_spk1[:, mask].detach().clone().cpu())
                            rec_dict["class_0"][1].append(self.rec_spk2[:, mask].detach().clone().cpu())
                            # rec_dict["class_0"][2].append(self.rec_spk3[:, mask].detach().clone().cpu())
                else:
                    pred, _ = self.forward(x)

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
        Function to create a mask for the hidden layer recordings. 
        Returns mask if there is still something to be recorded.
        If the mask would mask the whole input, False is being returned instead.

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
        # duplicate the mask for all classes
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

        :param x: A batch, as it would be usually passed through the network
        :type x: Tensor

        :param batch_first: Whether the first dimension is batch_size (True) or time_steps (False). Default True
        :type batch_first: bool, optional

        :returns:
        :rtype: None
        """

        mem1 = self.neuron1.reset_mem()
        mem2 = self.neuron2.reset_mem()
        # mem3 = self.neuron3.reset_mem()

        x = x.to(self.device)

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
        # cur3 = self.con3(spk2)
        # spk3, mem3 = self.neuron3(cur3, mem3)

        if self._record:
            self._init_tensors__(
                spk1.shape,
                spk2.shape,
                # spk3.shape
            )

        self._build = True



    def _init_tensors__(
        self,
        layer1_shape: tuple | None = None,
        layer2_shape: tuple | None = None,
        layer3_shape: tuple | None = None
    ) -> None:
        
        """
        Function that allocates the Tensors used during the recording of the hidden layers.

        :param layerX_shape: Tuple that defines the output shapes of layer X, defaults to None.
        :type layerX_shape: Tuple | None

        :returns:
        :rtype: None
        """
        if layer1_shape:
            self.rec_spk1 = torch.zeros(
                [
                    self._time_steps,
                    *layer1_shape
                ],
                dtype = torch.float32,
                device = self.device,
                requires_grad = False
            )

        if layer2_shape:
            self.rec_spk2 = torch.zeros(
                [
                    self._time_steps,
                    *layer2_shape
                ],
                dtype = torch.float32,
                device = self.device,
                requires_grad = False
            )

        if layer3_shape:
            self.rec_spk3 = torch.zeros(
                [
                    self._time_steps,
                    *layer3_shape
                ],
                dtype = torch.float32,
                device = self.device,
                requires_grad = False
            )

        self._counter = torch.full(
            size = (self._neurons_out,),
            fill_value = self._samples,
            dtype = torch.int32,
            device = self.device
        )