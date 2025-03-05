from imports import torch

# this function is not allowed to take any other hyperparmeters.
# stuff like slope has to be defined within the function!
def super_spike_21(input_, grad_input, spikes) -> torch.Tensor:
    raise NotImplementedError()