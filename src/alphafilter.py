from imports import DEVICE, math, torch


class AlphaFilter(torch.nn.Module):
    """
    A PyTorch module that applies an Alpha function filter to spike trains.

    The Alpha function models the Post-Synaptic Potential (PSP):
        epsilon(t) = (t / tau) * exp(1 - t / tau)

    This implementation is optimized for the tensor format (Time, Batch, Neurons).
    It uses 1D convolution along the time dimension.
    """
    def __init__(
        self, 
        tau: float = 5.0,
        ts: float = 1.0,
        learnable_tau: bool = False,
        filter_length: int | None = None
    ):
        """
        Args:
            tau (float): Time constant of the alpha function.
            ts (float): Time step duration (simulation resolution).
            learnable_tau (bool): If True, tau becomes a learnable parameter.
            filter_length (int, optional): Explicit length of the kernel. 
                                           If None, calculated as 5 * tau / ts.
        """
        super().__init__()

        self.ts = ts
        self.learnable_tau = learnable_tau

        if learnable_tau:
            # Initialize tau as a parameter. 
            # We use a softplus-like constraint or simply clamp during forward pass 
            # to ensure tau stays positive. Here we initialize with the provided value.
            self.tau_param = torch.nn.Parameter(torch.tensor(tau, dtype = torch.float32))
            self.current_tau = tau
        else:
            self.register_buffer('tau_param', torch.tensor(tau, dtype = torch.float32))
            self.current_tau = tau

        # Determine kernel length
        if filter_length is None:
            # Standard heuristic: 5 * tau covers most of the curve
            self.kernel_length = int(math.ceil(5 * self.current_tau / self.ts))
        else:
            self.kernel_length = filter_length

        # Initialize the kernel
        self.register_buffer('kernel', self._generate_kernel(self.current_tau))

        # Padding for causal convolution
        # We pad (kernel_length - 1) to the left to keep output size == input size
        self.padding = self.kernel_length - 1

    def _generate_kernel(self, tau: float | torch.Tensor) -> torch.Tensor:
        """Generates the alpha kernel tensor."""
        t = torch.arange(
            0,
            self.kernel_length * self.ts,
            self.ts,
            device = self.tau_param.device
        )

        # Alpha function: (t / tau) * exp(1 - t / tau)
        # Note: At t=0, this is 0.
        kernel = (t / tau) * torch.exp(1 - t / tau)

        # Normalize kernel (optional, but good for stability)
        kernel = kernel / kernel.sum()

        # Reshape for Conv1d: (OutChannels, InChannels, KernelLength)
        # Since we apply the same filter to every neuron independently,
        # we use groups=1 logic but effectively treat it as depthwise convolution
        # or just broadcast.
        # Here we return shape (1, 1, KernelLength) and will expand in forward.
        return kernel.view(1, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the alpha filter to the input spike tensor.

        Args:
            x (torch.Tensor): Input spikes of shape (Time, Batch, Neurons).

        Returns:
            torch.Tensor: Filtered output of shape (Time, Batch, Neurons).
        """
        # Update tau if learnable
        if self.learnable_tau:
            # Ensure tau is positive
            tau_val = torch.nn.functional.softplus(self.tau_param) + 1e-5 # Add small epsilon
            self.current_tau = tau_val.item()

            # Regenerate kernel if tau changed significantly (or just every step for simplicity)
            # For efficiency, one might check if tau changed, but regenerating is cheap for 1D.
            self.kernel = self._generate_kernel(tau_val).to(x.device)

        # Input shape: (T, B, N)
        T, B, N = x.shape

        # Reshape to (B, N, T) for Conv1d which expects (Batch, Channels, Length)
        x = x.permute(1, 2, 0)

        # Flatten Batch and Neurons into a single Channel dimension for Conv1d
        # New shape: (B * N, 1, T)
        x = x.contiguous().view(B * N, 1, T)

        # Expand kernel to apply to all "channels" (neurons)
        # We want to treat each neuron as an independent input channel.
        # Conv1d weight shape: (OutChannels, InChannels, K)
        # We want (1, B*N, K). 
        # However, standard Conv1d mixes channels. We want Depthwise convolution.
        # Trick: Use groups = B*N.

        kernel_expanded = self.kernel.expand(1, B * N, self.kernel_length)

        # Apply causal convolution
        # Pad manually to ensure causality (padding on the left)
        x = torch.nn.functional.pad(x, (self.padding, 0))

        # Perform convolution
        # groups = B*N ensures each neuron is filtered independently
        out = torch.nn.functional.conv1d(
            input = x,
            weight = kernel_expanded,
            bias = None,
            stride = 1,
            padding = 0,
            groups = B * N
        )

        # Reshape back to (B, N, T)
        out = out.view(B, N, T)

        # Reshape back to (T, B, N)
        out = out.permute(2, 0, 1)

        return out

    def get_tau(self):
        """Returns the current value of tau."""
        if self.learnable_tau:
            return (torch.nn.functional.softplus(self.tau_param) + 1e-5).item()
        return self.tau_param.item()


class RecurrentAlphaFilter(torch.nn.Module):
    """
    A stateful implementation of the Alpha Filter (Leaky Integrator).

    This is designed to be applied iteratively during the forward pass:
        for t in range(T):
            current_input = layer_output[t]
            filtered_output = alpha_filter(current_input)

    It maintains internal membrane states (u and v) that decay over time.
    """
    def __init__(
        self,
        neurons: int,
        tau: float = 5.0,
        ts: float = 1.0,
        learn_tau: bool = True
    ):
        """
        Args:
            neurons (int): Number of neurons/features to filter.
            tau (float): Time constant.
            ts (float): Time step duration.
            learnable_tau (bool): If True, tau is a learnable parameter.
        """
        super().__init__()
        self.neurons = neurons
        self.ts = ts
        self.learnable_tau = learn_tau

        if learn_tau:
            self.tau_param = torch.nn.Parameter(
                torch.tensor(
                    tau,
                    dtype = torch.float32,
                    device = DEVICE
                )
            )
        else:
            self.register_buffer('tau_param', torch.tensor(
                tau,
                dtype = torch.float32,
                device = DEVICE
            ))

        # State buffers (u and v)
        # We initialize them as None and reset them in the first forward pass
        # or explicitly via a reset method.
        self.register_buffer('state_u', torch.zeros(1, neurons))
        self.register_buffer('state_v', torch.zeros(1, neurons))

    def reset_states(self, batch_size: int = 1):
        """Resets the internal states to zero. Call this at the start of a sequence."""
        self.state_u = torch.zeros(
            (batch_size, self.neurons),
            device = self.tau_param.device
        )
        self.state_v = torch.zeros(
            (batch_size, self.neurons),
            device = self.tau_param.device
        )

    def get_decay(self):
        """Calculates the decay factor alpha."""
        if self.learnable_tau:
            # Ensure tau is positive
            tau_val = torch.nn.functional.softplus(self.tau_param) + 1e-5
        else:
            tau_val = self.tau_param

        alpha = torch.exp(-self.ts / tau_val)
        return alpha, tau_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs one time-step of filtering.

        Args:
            x (torch.Tensor): Input at current time step. Shape: (Batch, Neurons).

        Returns:
            torch.Tensor: Filtered output. Shape: (Batch, Neurons).
        """
        batch_size = x.shape[0]

        # Initialize states if batch size changed or first run
        if self.state_u.shape[0] != batch_size:
            self.reset_states(batch_size)

        alpha, tau_val = self.get_decay()

        # 1. Update u (Fast)
        # u[t] = alpha * u[t-1] + x[t]
        self.state_u = alpha * self.state_u + x

        # 2. Update v (Slow)
        # v[t] = alpha * v[t-1] + u[t]
        self.state_v = alpha * self.state_v + self.state_u

        # 3. Output
        # y[t] = (1 - alpha) / tau * (v[t] - u[t])
        # Note: The scaling factor (1-alpha)/tau approximates 1/ts for small ts, 
        # but strictly follows the alpha function derivation.
        scale = (1.0 - alpha) / tau_val
        output = scale * (self.state_v - self.state_u)

        return output

    def detach_states(self):
        """Detaches states from the computational graph. Useful for BPTT."""
        self.state_u = self.state_u.detach()
        self.state_v = self.state_v.detach()