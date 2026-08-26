from imports import Callable, surrogate, torch


def stable_sigmoid(slope: int) -> Callable:
    
    def inner(input_, grad_input, spikes):
        _slope = slope

        grad = (
            grad_input
            * _slope
            * torch.sigmoid(_slope * input_)
        * (1 - torch.sigmoid(_slope * input_))
        )
        return grad

    return surrogate.custom_surrogate(inner)