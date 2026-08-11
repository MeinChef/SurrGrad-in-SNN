from imports import torch


class PSPFilter(torch.nn.Module):
    def __init__(
        self,
        neurons,
        tau_init=10.0,
        ts=1.0,
        max_tau=100.0,
    ):
        super().__init__()

        self.neurons = neurons
        self.ts = ts

        self.log_tau = torch.nn.Parameter(
            torch.full((neurons,), tau_init).log()
        )

        self.max_tau = max_tau
        self.K = int(torch.ceil(
            torch.tensor(5 * max_tau / ts)
        ).item())

        self.register_buffer(
            "history",
            torch.zeros(1, neurons, self.K)
        )

    @property
    def tau(self):
        return self.log_tau.exp()

    def reset(self, batch_size=None):
        if batch_size is None:
            batch_size = self.history.shape[0]

        self.history = torch.zeros(
            batch_size,
            self.neurons,
            self.K,
            device=self.history.device,
            dtype=self.history.dtype,
        )

    def forward(self, spikes):
        # spikes: [batch, neurons]

        B, N = spikes.shape

        if self.history.shape[0] != B:
            self.reset(B)

        # Shift history into the past.
        self.history = torch.roll(
            self.history, shifts=1, dims=-1
        )
        self.history[..., 0] = spikes

        # Build one alpha PSP kernel per neuron.
        t = torch.arange(
            self.K,
            device=spikes.device,
            dtype=spikes.dtype,
        ) * self.ts

        tau = self.tau.to(spikes.dtype).clamp(
            min=self.ts
        )

        kernel = (
            (t[None, :] / tau[:, None])
            * torch.exp(
                1.0 - t[None, :] / tau[:, None]
            )
        )

        # [B, N, K] * [N, K] -> [B, N]
        psp = (self.history * kernel[None, :, :]).sum(dim=-1)

        return psp * self.ts