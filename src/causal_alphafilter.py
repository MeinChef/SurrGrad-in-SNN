from imports import torch, math

class CausalConvAlphaFilter(torch.nn.Module):
    """
    A non-recurrent, causal 1D convolutional filter that applies an alpha function to spike trains.

    The alpha function is:
        epsilon(t) = (t / tau) * exp(1 - t / tau)

    This implementation:
    - Uses 1D convolution with causal padding.
    - Is non-recurrent (no state).
    - Is learnable (tau is a parameter).
    - Preserves spike timing.
    - Can be applied in the same forward pass as the recurrent version.
    """

    def __init__(
        self,
        neurons: int,
        tau: float = 5.0,
        ts: float = 1.0,
        learn_tau: bool = True,
        filter_length: int | None = None,
    ):
        """
        Args:
            neurons (int): Number of neurons (features).
            tau (float): Time constant of the alpha function.
            ts (float): Time step duration (simulation resolution).
            learn_tau (bool): If True, tau becomes a learnable parameter.
            filter_length (int, optional): Explicit length of the kernel. If None, calculated as 5 * tau / ts.
        """
        super().__init__()
        self.neurons = neurons
        self.ts = ts
        self.learn_tau = learn_tau

        # Learnable tau parameter
        if learn_tau:
            self.tau_param = torch.nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        else:
            self.register_buffer('tau_param', torch.tensor(tau, dtype=torch.float32))

        # Determine kernel length
        if filter_length is None:
            self.kernel_length = int(math.ceil(5 * tau / ts))
        else:
            self.kernel_length = filter_length

        # Generate the alpha kernel (shape: [1, 1, kernel_length])
        self.register_buffer('kernel', self._generate_kernel(tau))

        # Causal padding: pad (kernel_length - 1) on the left
        self.padding = self.kernel_length - 1

    def _generate_kernel(self, tau: float | torch.Tensor) -> torch.Tensor:
        """Generates the alpha kernel tensor."""
        t = torch.arange(
            0,
            self.kernel_length * self.ts,
            self.ts,
            device=self.tau_param.device
        )

        # Alpha function: (t / tau) * exp(1 - t / tau)
        kernel = (t / tau) * torch.exp(1 - t / tau)

        # Normalize kernel (optional, but good for stability)
        kernel = kernel / kernel.sum()

        # Reshape for Conv1d: (1, 1, kernel_length)
        return kernel.view(1, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the causal convolutional alpha filter to the input spike tensor.

        Args:
            x (torch.Tensor): Input spikes at current time step. Shape: (Batch, Neurons).

        Returns:
            torch.Tensor: Filtered output. Shape: (Batch, Neurons).
        """
        # x: (B, N)

        # Update tau if learnable
        if self.learn_tau:
            tau_val = torch.nn.functional.softplus(self.tau_param) + 1e-5
            # Regenerate kernel if tau changed
            self.kernel = self._generate_kernel(tau_val).to(x.device)

        # Reshape to (B, N, 1) for Conv1d
        x = x.unsqueeze(-1)  # (B, N, 1)

        # Now, we want weight to be: (N, 1, K) → one filter per neuron
        # So expand kernel from (1, 1, K) to (N, 1, K)
        kernel_expanded = self.kernel.expand(self.neurons, 1, -1)  # (N, 1, K)

        # Apply causal convolution
        # Pad on the left to keep output size = input size
        x_padded = torch.nn.functional.pad(x, (self.padding, 0))  # (B, N, T + padding)

        # Perform convolution: (B, N, 1) -> (B, N, 1)
        out = torch.nn.functional.conv1d(
            input=x_padded,
            weight=kernel_expanded,
            bias=None,
            stride=1,
            padding=0,
            groups=self.neurons  # Each neuron is filtered independently
        )

        # Remove the last dimension: (B, N, 1) -> (B, N)
        out = out.squeeze(-1)

        return out