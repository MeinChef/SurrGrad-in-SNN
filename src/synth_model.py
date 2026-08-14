from imports import DEVICE, TORCH_RNG, Callable, Literal, math, torch, tqdm, warnings
from imports import snntorch as snn
from misc import resolve_acc, resolve_gradient, resolve_loss, resolve_optim
from pspfilter import PSPFilter

DEBUG = False

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
        """

        super().__init__()

        # resolve gradient
        surrogate = resolve_gradient(config = config.get(
            "surrogate",
            {"type": "fast_sigmoid", "slope": 100}
        ))

        ###########################
        ### DEFINITION OF MODEL ###
        ###########################
        self.layers:  torch.nn.ModuleList = torch.nn.ModuleList()
        self.neurons: torch.nn.ModuleList = torch.nn.ModuleList()
        self.filters: torch.nn.ModuleList = torch.nn.ModuleList()
        neuron_list = config.get("neurons", [100,2])

        # create list of properties with tuple[in, out]
        self.neuron_prop = [
            (config.get("features", {"val": 10})["val"], neuron_list[0]),
            *[(
                neuron_list[i-1],
                neuron_list[i]
            ) for i in range(1, len(neuron_list))]
        ]

        # and calculate the neuron beta from the tau and ts
        neuron_beta = torch.exp(
            torch.tensor(
                - config.get("ts", 1) /
                  config.get("neuron_tau", 5)
            )
        )

        # Define model structure
        for i, n in enumerate(self.neuron_prop):
            # Linear connection layer
            layer = torch.nn.Linear(
                in_features = n[0],
                out_features = n[1],
                device = DEVICE
            )
            layer = torch.nn.utils.parametrizations.weight_norm(layer, name = "weight")
            self.layers.append(layer)

            # leaky neurons
            if n == self.neuron_prop[-1]:
                reset = "none"
            else:
                reset = config.get("neuron_reset", "substract")
            neuron = snn.Leaky(
                beta = neuron_beta,
                learn_beta = config.get("neuron_learn_beta", True),
                spike_grad = surrogate,
                init_hidden = False,
                reset_mechanism = reset
            )
            self.neurons.append(neuron)

            # add alpha filters to the output of the spiking layer
            fil = PSPFilter(
                neurons = n[1],
                tau_init = config.get("filter_tau", 10.),
                ts = config.get("ts", 1.),
            )
            self.filters.append(fil)

        assert len(self.neuron_prop) == len(self.neurons)
        assert len(self.layers)      == len(self.neurons)
        assert len(self.filters)     == len(self.neurons)

        # resolve additional bits
        self.lossfn = resolve_loss(config = config.get(
            "loss", {
                "type": "mean_ce",
                "reduction": "mean",
                "intermediate_reduction": "mean"
            }
        ))
        self.acc    = resolve_acc(config = config.get(
            "accuracy",
            {"type": "rate"}
        ))
        self.optim  = resolve_optim(
            config  = config.get("optimiser", {
                "type": "adam",
                "lr": 0.002,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0001
            }),
            params  = [
                {"params": [p for p in self.parameters() if not hasattr(p, 'tau_param')]},
                {"params": [p for p in self.parameters() if hasattr(p, 'tau_param')], "lr": 0.02}
            ]
        )

        # save config to class
        self._epochs        = config.get("epochs", 100)
        self._partial_train = config.get("partial_training", -1)
        self._partial_test  = config.get("partial_testing", -1)
        self._move_fraction = config.get("move_fraction", 0.0)
        self._return_spk    = config.get("return_spk", False)
        self._samples       = config.get("samples", 20)
        self._time_steps    = config.get("time_steps", {"val": 1000})["val"]
        self._forward_output_buffer = None


        # accuracy/loss stuff
        acc_default_dict = {"population": {
            "is_pop": False,
            "num_classes": 2
        }}
        self._population    = config.get("accuracy", acc_default_dict)["population"]["is_pop"]
        self._num_classes   = config.get("accuracy", acc_default_dict)["population"]["num_classes"]
        self._best_loss = torch.inf

        # send to gpu
        self.to(device = DEVICE)

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
        if batch_first:
            # reshape to actually have the time_steps first again
            # that makes the for loop later cleaner
            x = x.permute(1, 0, -1).contiguous()

        # setup
        # the linter annotations are because it assumes reset_mem and other functions
        # to be a tensor / Module instead of what they are.
        mems = [neuron.reset_mem() for neuron in self.neurons]      # pyright: ignore[reportCallIssue]
        for fil in self.filters:                                    # pyright: ignore[reportAssignmentType]
            fil: PSPFilter
            fil.reset(x.shape[1])


        # pre-allocate the output-tensor
        out = torch.empty(
            [
                self._time_steps,
                x.shape[1],
                self.neuron_prop[-1][1] # no of neurons in last layer
            ],
            device = DEVICE,
            dtype = x.dtype
        )


        for step in range(self._time_steps):
            spk = x[step]

            # hidden layers
            for i in range(len(self.layers)):
                cur = self.layers[i](spk)
                spk, mems[i] = self.neurons[i](cur, mems[i])
                spk = self.filters[i](spk)

            # store output
            if self._return_spk:
                out[step] = spk
            else:
                out[step] = mems[-1]

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

        # pre-define variables
        loss_hist = []
        acc_hist  = []

        # set model in training mode
        self.train()

        # training loop
        for i, (x, target) in tqdm.tqdm(
            iterable = enumerate(data),
            total = len(data),
            desc = "Training Batches"
        ):
            # check if the training has been already done to the specified amount
            if i == self._partial_train:
                break

            # move tensors to device
            if x.device != DEVICE:
                x = x.to(DEVICE)
            if target.device != DEVICE:
                target = target.to(DEVICE)

            # make prediction
            pred = self.forward(x, batch_first = False)

            # loss and accuracy calculations
            loss = self.lossfn(pred, target)
            if loss.isnan().any():
                print("something's fishy")
            acc = self.acc(
                pred,
                target,
                population_code = self._population,
                num_classes = self._num_classes
            )

            # weight update
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # store loss/acc values for neat little plotting
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

        # pre-define variables
        loss_hist = []
        acc_hist  = []
        # set model in evaulating mode
        self.eval()


        # test loop
        with torch.no_grad():
            for i, (x, target) in tqdm.tqdm(
                iterable = enumerate(data),
                total = len(data),
                desc = "Training Batches"
            ):
                # check if the training has been already done to the specified amount
                if i == self._partial_test:
                    break

                # move tensors to device
                x = x.to(DEVICE)
                target = target.to(DEVICE)

                pred = self.forward(x, batch_first = False)

                # loss and accuracy calculations
                loss = self.lossfn(pred, target)
                acc = self.acc(
                    pred, 
                    target,
                    population_code = self._population,
                    num_classes = self._num_classes
                )

                # record loss/accuracy during training
                loss_hist.append(loss.item())
                acc_hist.append(acc)

        # update best loss
        self._best_loss = min(
            self._best_loss,
            torch.tensor(loss_hist).mean()
        )
        torch.cuda.empty_cache()

        return loss_hist, acc_hist

    #########################
    ### Augmented Forward ###
    #########################

    def _forward_layer(
        self,
        x: torch.Tensor,
        layer: int = 0
    ) -> torch.Tensor:

        if layer < 0 or layer > len(self.layers):
            raise ValueError(f"Expected layer in [1, {len(self.layers)}]. Got {layer}.")

        # setup
        neuron = self.neurons[layer]
        mem = neuron.reset_mem()                # pyright: ignore[reportCallIssue]
        con = self.layers[layer]
        fil = self.filters[layer]               # pyright: ignore[reportAssignmentType]
        fil: PSPFilter
        fil.reset(x.shape[1])


        # pre-allocate the output-tensor
        out = torch.zeros(
            [
                self._time_steps,
                x.shape[1],                 # batch size
                self.neuron_prop[layer][1]  # output of layer
            ],
            device = DEVICE
        )

        # loop over time
        for step in range(self._time_steps):
            cur = con(x[step])
            spk, mem = neuron(cur, mem)
            spk = fil(spk)

            if self._return_spk and layer == len(self.layers):
                out[step] = mem
            else:
                out[step] = spk

        return out

    def _jitter_layer_out(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        T, B, N = x.shape   # noqa: RUF059
        out = x.clone()
        left = 0

        def check_candidates(
            candy: torch.Tensor,
            left: int,
            valid_to: torch.Tensor
        ) -> torch.Tensor:
            # collision with existing spikes
            # the +left translates again into the valid_to coordinate system
            collide = ~torch.isin(candy + left, valid_to)

            # duplicate values
            unique, counts = candy.unique(return_counts = True)
            duplicate = unique[counts > 1]
            duplicate = torch.isin(candy, duplicate)

            return collide | duplicate


        for b in range(B):
            for n in range(N):
                # existing spikes
                spike_idx = torch.where(x[:, b, n] > 0)[0]
                # valid_to needs to be calculated from out, since out might change in size
                valid_to = torch.where(out[:, b, n] == 0)[0]

                to_move = math.ceil(
                    spike_idx.numel() * self._move_fraction
                )

                if to_move == 0:
                    continue

                # randomly choose spikes to remove
                remove_idx = spike_idx[
                    torch.randperm(
                        spike_idx.numel(),
                        generator = TORCH_RNG,
                        device = DEVICE
                    )[:to_move]
                ]
                # and add these to the valid positions
                # the +left because they need to be in the same coordinate system as valid_to
                valid_to = torch.cat(
                    [valid_to, remove_idx + left] 
                )

                jitters = torch.randint(
                    low = -self._jitter,
                    high = self._jitter + 1,
                    size = (to_move,),
                    generator = TORCH_RNG,
                    device = DEVICE
                )

                # this might lead to two or more spikes to be on the same time
                candidates = remove_idx + jitters

                # thus we check if they collide with existing spikes 
                # (excluding the ones that'll be removed)
                # definition of function above
                mask = check_candidates(candidates, left, valid_to)

                # mask is a positive mask, meaning True values are acceptable.
                # negating the mask allows for checking if any values are not acceptable
                if ~mask.any():
                    counter = 0
                    while ~mask.any() and counter < 100:
                        # update valid_to - here and not earlier, because it might not be needed
                        valid_to = valid_to[
                            ~torch.isin(valid_to, candidates[~mask] + left)
                        ]

                        # create as many new values as there are True filter values 
                        jitters = torch.randint(
                            low = -self._jitter,
                            high = self._jitter + 1,
                            size = (int(mask.sum()),),
                            generator = TORCH_RNG,
                            device = DEVICE
                        )

                        # update candidates and afterwards the filter
                        candidates[mask] = remove_idx[mask] + jitters
                        mask = check_candidates(candidates, left, valid_to)

                        # increase counter
                        counter += 1

                    # if the searching was unsuccessful, brute-force the first free spot
                    if ~mask.any():
                        for i in torch.where(mask)[0]:
                            for j in range(-self._jitter, self._jitter + 1):
                                if ((valid_to == remove_idx[i] + j + left).any()):
                                    chosen = remove_idx[i] + j

                                    # update valid_to (translate again with +left)
                                    valid_to = valid_to[
                                        valid_to != chosen +left
                                    ]

                                    # and set it
                                    candidates[i] = chosen
                                    break
                        mask = check_candidates(candidates, left, valid_to)

                    # cry if that did not work
                    if ~mask.any():
                        warnings.warn(
                            "Could not find spot to jitter spike to. "
                            "This really should not happen, but it did.\n"
                            f"{int(~mask.sum())} Spike(s) will be lost on neuron {n} at Sample {b}"
                        )
                        candidates = candidates[mask]


                # make time longer if any spikes would now be out of time
                if (candidates < -left).any():
                    needed_left = -int(candidates.min())
                    new_left = max(left, needed_left)

                    out = torch.nn.functional.pad(
                        out,
                        (
                            0, 0,                   # last dim
                            0, 0,                   # middle dim
                            new_left - left, 0,     # first dim
                        )
                    )

                    # update left
                    left = new_left

                # and update the candidates to be in the correct coordinate system
                add_idx = candidates + left

                if (add_idx >= out.shape[0]).any():
                    need = int(add_idx.max() + 1)
                    out = torch.nn.functional.pad(
                        out,
                        (
                            0, 0,
                            0, 0,
                            0, need - out.shape[0],
                        )
                    )


                # write to tensor
                out[add_idx, b, n] = 1
                out[remove_idx + left, b, n] = 0

        return out

    def _shuffle_layer_out(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        T, B, N = x.shape   # noqa: RUF059
        out = x.clone()

        for b in range(B):
            for n in range(N):
                # existing spikes
                spike_idx = torch.where(x[:, b, n] > 0)[0]

                to_move = math.ceil(
                    spike_idx.numel() * self._move_fraction
                )

                if to_move == 0:
                    continue

                # available empty positions
                empty_idx = torch.where(x[:, b, n] == 0)[0]

                to_move = min(to_move, empty_idx.numel())

                # randomly choose spikes to remove
                remove_idx = spike_idx[
                    torch.randperm(
                        spike_idx.numel(),
                        generator = TORCH_RNG,
                        device = DEVICE
                    )[:to_move]
                ]

                # randomly choose empty positions to activate
                add_idx = empty_idx[
                    torch.randperm(
                        empty_idx.numel(),
                        generator = TORCH_RNG,
                        device = DEVICE
                    )[:to_move]
                ]

                # write to tensor
                out[remove_idx, b, n] = 0
                out[add_idx, b, n] = 1

        return out

    def augmented_eval(
        self,
        data: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        augment: Literal["shuffle", "jitter"] | Callable = "jitter",                    # noqa: F821
        jitter: int | None = None,
        only_nth_layer: int | None = None
    ) -> tuple[list, list]:

        if augment != "jitter" and augment != "shuffle" and not callable(augment):
            raise ValueError("Expected 'shuffle' or 'jitter'.\n"
                             f"Got '{augment}' (Type: {type(augment)}) instead.")

        if self._move_fraction <= 0 or only_nth_layer is None:
            self._move_fraction = 0
            warnings.warn(
                "This function got called with a move_fraction of 0 or less. Or with only_nth_layer as None.\n"
                "This will result in a very inefficient forward pass. "
                "If this is not intended, change the 'move_fraction' and or 'augmented_layer' value in 'config.yml'.",
                category = RuntimeWarning
            )

        if augment == "jitter":
            augment_fn = self._jitter_layer_out
            if jitter and jitter > 0:
                self._jitter = jitter
            else:
                raise ValueError("Expected the parameter 'jitter' to be int and positive.\n"
                                 f"Got {jitter} of Type {type(jitter)} instead.")

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
                if x.device != DEVICE:
                    x = x.to(DEVICE)
                if target.device != DEVICE:
                    target = target.to(DEVICE)

                x = self._forward_layer(x, 1)
                if only_nth_layer == 1:
                    x = augment_fn(x)

                x = self._forward_layer(x, 2)
                if only_nth_layer == 2:
                    x = augment_fn(x)

                x = self._forward_layer(x, 3)
                if only_nth_layer == 3:
                    x = augment_fn(x)

                loss.append(self.lossfn(x, target).item)
                acc.append(self.acc(x, target))

        return loss, acc