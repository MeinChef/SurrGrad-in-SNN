from imports import torch


class PSPFilter(torch.nn.Module):
    """
    Stateful recursive realization of the SLAYER alpha PSP kernel.

    SLAYER kernel:

        k(t) = (t / tau) * exp(1 - t / tau)

    sampled at:

        t = n * Ts

    so:

        k[n] = (n * Ts / tau) * exp(1 - n * Ts / tau)

    The recurrent realization is:

        a[n] = exp(-Ts / tau)

        q[n] = x[n] + a[n] * q[n-1]

        r[n] = a[n] * (r[n-1] + q[n-1])

        psp[n] = exp(1) * (Ts / tau) * r[n]

    IMPORTANT:
        States are NOT detached inside forward().
        This allows gradients to propagate through time.

    Detach the states explicitly at BPTT boundaries if truncated
    BPTT is desired.
    """

    def __init__(
        self,
        neurons: int,
        tau_init: float = 10.0,
        ts: float = 1.0,
        max_tau: float = 100.0,
    ):
        super().__init__()

        self.neurons = neurons
        self.ts = ts
        self.max_tau = max_tau

        self.raw_tau = torch.nn.Parameter(
            torch.full((neurons,), float(tau_init))
        )

        self.register_buffer(
            "q",
            torch.zeros(1, neurons),
            persistent = False
        )

        self.register_buffer(
            "r",
            torch.zeros(1, neurons),
            persistent = False
        )

    def reset(self, batch_size = None):
        """
        Resets the hidden states.
        If batch_size is not given, will infer it from the last shape it got.
        """
        if batch_size is None:
            batch_size = self.q.shape[0]


        self.q = self.q.new_zeros(
            batch_size,
            self.neurons,
        )

        self.r = self.r.new_zeros(
            batch_size,
            self.neurons,
        )

    def detach_state(self):
        """
        Detaches the hidden states from the autograd graph. Call when desired.
        """
        self.q = self.q.detach()
        self.r = self.r.detach()

    def forward(self, spikes):
        # spikes: [batch, neurons]

        B, N = spikes.shape

        if N != self.neurons:
            raise ValueError(
                f"Expected {self.neurons} neurons, got {N}"
            )

        if self.q.shape[0] != B:
            self.reset(B)

        # to keep tau in the range [ts, tau_max]
        tau = self.ts + (
            self.max_tau - self.ts
        ) * torch.sigmoid(self.raw_tau)

        # Exact exponential decay over one timestep.
        a = torch.exp(-self.ts / tau)

        q_prev = self.q
        r_prev = self.r

        # Update q
        q = spikes + a[None, :] * q_prev

        r = a[None, :] * (
            r_prev + q_prev
        )

        psp = (
            torch.e
            * (self.ts / tau)[None, :]
            * r
        )

        # Store state for the next timestep.
        self.q = q
        self.r = r

        return psp