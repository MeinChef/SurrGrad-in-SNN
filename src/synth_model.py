from imports import torch
from imports import math
from imports import os
from imports import Path
from imports import snntorch as snn
from imports import tqdm
from imports import SummaryWriter
from imports import Literal, Callable
from imports import warnings
from imports import NOW, DEVICE
from misc import resolve_gradient, resolve_acc, resolve_loss, resolve_optim

DEBUG = False
# torch.autograd.set_detect_anomaly(True)

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

        # resolve gradient
        surrogate = resolve_gradient(config = config["surrogate"])

        ###########################
        ### DEFINITION OF MODEL ###
        ###########################
        # layer 1
        self.con1 = torch.nn.Linear(
            in_features = config["features"]["val"],
            out_features = config["neurons_hidden_1"],
            device = DEVICE
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
            device = DEVICE
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
            device = DEVICE
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

        # save loss informations
        self._loss_reduction = config["loss"]["reduction"]

        # save config to class
        self._time_steps = config["time_steps"]["val"]
        self._epochs = config["epochs"]
        self._partial_train = config["partial_training"]
        self._partial_test  = config["partial_testing"]
        self._move_fraction = config["move_fraction"]
        self._best_loss = torch.inf

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
        self.to(device = DEVICE)

        if DEBUG:
            self.writer = SummaryWriter(
                log_dir = os.path.join(
                    Path(__file__).parent.parent,
                    "runs",
                    NOW
                )
            )

        # Let cuDNN find optimal algorithms
        torch.backends.cudnn.benchmark = True  

    ####################################
    ### DEFINITION OF MAIN FUNCTIONS ###
    ####################################

    def forward(
        self,
        x: torch.Tensor,
        # batch_first: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:

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

        # if batch_first:
        #     # reshape to actually have the time_steps first again
        #     # that makes the for loop later cleaner
        #     x = x.permute(1, 0, -1)#.contiguous()

        # pre-allocate the output-tensor
        out_spk = torch.empty(
            [
                self._time_steps,
                x.shape[1],
                self._neurons_out
            ], 
            device = DEVICE,
            dtype = x.dtype
        )
        out_mem = torch.empty_like(
            out_spk,
            device = DEVICE
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

            out_spk[step] = spk3
            out_mem[step] = mem3

        return out_spk, out_mem

    def fit(
        self,
        data: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        epoch: int = 0
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

        # prefetch the first batch and move it to the GPU
        dataiter = iter(data)
        x, target = next(dataiter)
        x = x.to(DEVICE)
        target = target.to(DEVICE)

        # # offset for Tensorboard
        # offset = epoch * len(data)

        # with torch.profiler.profile(
        #     schedule = torch.profiler.schedule(
        #         skip_first = 3, 
        #         wait = 2, 
        #         warmup = 5, 
        #         active = 1, 
        #         repeat = 1
        #     ),
        #     on_trace_ready = trace_handler,
        #     with_stack = True,
        #     record_shapes = True,
        # ) as prof:


        # training loop
        for i, (next_x, next_target) in tqdm.tqdm(enumerate(dataiter)):
            # check if the training has been already done to the specified amount
            if i == self._partial_train:
                break
            
            # move tensors to device
            if next_x.device != DEVICE:
                next_x = next_x.to(DEVICE, non_blocking = True)
            if next_target.device != DEVICE:
                next_target = next_target.to(DEVICE, non_blocking = True)

            # make prediction
            _, pred = self.forward(x)

            # loss and accuracy calculations
            loss = self.lossfn(pred, target)
            if loss.isnan().any():
                print("something's fishy")
            acc = self.acc(pred, target)


            # if self._loss_reduction == "none" and DEBUG:
            #     losscp = loss.clone().detach()
            #     mask = target == 0
            #     self.writer.add_scalar(
            #         tag = "Train/Class0-Loss",
            #         scalar_value = losscp[mask].mean(),
            #         global_step = offset + i
            #     )
            #     self.writer.add_scalar(
            #         tag = "Train/Class1-Loss",
            #         scalar_value = losscp[~mask].mean(),
            #         global_step = offset + i
            #     )

            # loss = loss.mean()

            # weight update
            self.optim.zero_grad()
            # loss.backward(retain_graph = True)
            loss.backward()
            self.optim.step()

            x = next_x
            target = next_target
            # torch.cuda.synchronize()

            # if DEBUG:
            #     plot_grad_flow(self.named_parameters())

            #     # get gradients of all learneable parameters
            #     j = 0
            #     for param in self.parameters():
            #         gradsrc = param.grad
            #         if gradsrc is not None:
            #             grad = gradsrc.clone().detach()
                        
            #             # skip if the parameters are the layers
            #             if len(grad.shape) <= 1:
            #                 if grad.isnan().any():
            #                     print(f"Found NaN values in Layer Gradients: {grad.isnan().sum()}")
            #                 continue
                        
            #             # max and min of non-nan values
            #             mask = grad.isfinite()
            #             maxval = grad[mask].max().item()
            #             minval = grad[mask].min().item()
                        
            #             # fix nans in channel 0
            #             grad[~mask] = minval
                        
            #             # channel 1: contain all the NaNs
            #             nangrads = torch.full_like(
            #                 grad,
            #                 fill_value = minval
            #             )
            #             nangrads[(~mask).nonzero()] = maxval
                        
            #             # channel 2: zeros
            #             filler = torch.full_like(
            #                 grad,
            #                 fill_value = minval
            #             )
            #             img = torch.stack(
            #                 (
            #                     grad,
            #                     nangrads,
            #                     filler
            #                 )
            #             )

            #             img = (img - img.min()) / (img.max() - img.min())
            #             # img *= 255
            #             self.writer.add_image(
            #                 tag = f"Train/Gradients-Layer{j}",
            #                 img_tensor = img,
            #                 global_step = offset + i
            #             )
                        
            #             j += 1

            #     self.writer.add_scalar(
            #         tag = "Train/Loss",
            #         scalar_value = loss.item(),
            #         global_step = offset + i
            #     )
            #     self.writer.add_scalar(
            #         tag = "Train/Accuracy",
            #         scalar_value = acc,
            #         global_step = offset + i
            #     )

            # TODO: dump list regularly to file
            loss_hist.append(loss.item())
            acc_hist.append(acc)
            # prof.step()

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
                x = x.to(DEVICE)
                target = target.to(DEVICE)

                _, pred = self.forward(x)

                # loss and accuracy calculations
                loss = self.lossfn(pred, target)
                acc = self.acc(pred, target)
                
                if self._loss_reduction == "none":
                    loss = loss.mean()

                # TODO: record loss/accuracy during training
                # TODO: dump list regularly to file
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
            device = DEVICE
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
            device = DEVICE
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
            device = DEVICE
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
        T, B, N = x.shape
        out = x.clone()
        left = 0

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
                    torch.randperm(spike_idx.numel(), device = DEVICE)[:to_move]
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
                    device = DEVICE
                )

                # this might lead to two or more spikes to be on the same time
                candidates = remove_idx + jitters
                
                # thus we check if they collide with existing spikes 
                # (excluding the ones that'll be removed)
                def check_candidates(candy: torch.Tensor) -> torch.Tensor:
                    # collision with existing spikes
                    # the +left translates again into the valid_to coordinate system
                    collide = ~torch.isin(candy + left, valid_to)   
                    
                    # duplicate values
                    unique, counts = candy.unique(return_counts = True)
                    duplicate = unique[counts > 1]
                    duplicate = torch.isin(candy, duplicate)
                    
                    return collide | duplicate

                mask = check_candidates(candidates)
                
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
                            device = DEVICE
                        )

                        # update candidates and afterwards the filter
                        candidates[mask] = remove_idx[mask] + jitters
                        mask = check_candidates(candidates)
                        
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
                        mask = check_candidates(candidates)

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
        T, B, N = x.shape
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

                if empty_idx.numel() < to_move:
                    to_move = empty_idx.numel()

                # randomly choose spikes to remove
                remove_idx = spike_idx[
                    torch.randperm(spike_idx.numel(), device = DEVICE)[:to_move]
                ]

                # randomly choose empty positions to activate
                add_idx = empty_idx[
                    torch.randperm(empty_idx.numel(), device = DEVICE)[:to_move]
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
        
        if augment != "jitter" and augment != "shuffle" and callable(augment):
            raise ValueError("Expected 'shuffle' or 'jitter'.\n"
                             f"Got '{augment}' (Type: {type(augment)}) instead.")
        
        if self._move_fraction <= 0:
            self._move_fraction = 0
            warnings.warn(
                "This function got called with a move_fraction of 0 or less.\n"
                "This will result in a very inefficient forward pass. "
                "If this is not intended, change the move_fraction value in 'config.yml'.",
                category = RuntimeWarning
            )

        if augment == "jitter":
            augment_fn = self._jitter_layer_out
            if jitter and jitter > 0:
                self._jitter = jitter
            else:
                raise ValueError("Expected the parameter 'jitter' to be int and positive.\n"
                                 f"Got {jitter} of {type(jitter)} instead.")
            
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
    


from imports import plt
from imports import Line2D
from imports import numpy as np
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    plt.savefig("gradflow.png")

def trace_handler(p):
    output = p.key_averages().table(sort_by=f"self_{str(DEVICE)}_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("./torch-prof/trace_" + str(p.step_num) + ".json")
