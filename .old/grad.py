from imports import torch
from torch.autograd.function import once_differentiable
from imports import Callable

################################
##### THIS LEAKS AS OF NOW #####
################################


# this function is not allowed to take any other hyperparmeters.
# stuff like slope has to be defined within the function!
def super_spike_21(slope:float, alpha:float, beta:float) -> Callable:
    """Sigmoid surrogate gradient enclosed with a parameterized parameters."""
    slope = slope
    alpha = alpha
    beta = beta
    
    def inner(x):
        return SuperSpike19.apply(x, alpha, beta, slope)
    
    return inner

class SuperSpike19(torch.autograd.Function):
    
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of sigmoid function.

        .. math::

                TODO


    Adapted from:

    *F. Zenke, H. Mostafa, E. Neftci (2019) Surrogate Gradient Learning in Spiking Neural Networks
    https://doi.org/10.48550/arXiv.1901.09948*"""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, alpha: float, beta: float, slope: float) -> torch.Tensor:
        ctx.save_for_backward(input_)
        
        ctx.slope = slope
        ctx.alpha = alpha # synampic trace decay
        ctx.beta = beta # membrane potential decay (I think we don't need that)

        # Initialize dU_dW and dI_dW at the first step
        # if not hasattr(ctx, 'dU_dW'):
        #     ctx.dU_dW = torch.zeros_like(input_)
        #     ctx.dI_dW = torch.zeros_like(input_)


        out = (input_ > 0).float()
        return out
    

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_outputs: torch.Tensor):
        # input refers to the pre-trace
        input_, = ctx.saved_tensors
        grad_input = grad_outputs.clone()
    
        # compute the gradient (derivative of sigmoid)
        grad = (
            grad_input
            * ctx.slope
            * torch.exp(-ctx.slope * input_)
            / ((torch.exp(-ctx.slope * input_) + 1) ** 2)
        )

        # Update the gradient terms (without negative term)
        # dI_dW = ctx.alpha * ctx.dI_dW + input_  # ∂I/∂W # before: ctx.presynaptic_spike
        # dU_dW = ctx.beta * ctx.dU_dW + dI_dW  # ∂U/∂W

        # Compute final weight update
        # weight_update = grad * dU_dW  

        # Store updated gradients for the next step
        # ctx.dU_dW = dU_dW
        # ctx.dI_dW = dI_dW


        # needs to return as many arguments as the forward function accepts
        return grad, None, None, None