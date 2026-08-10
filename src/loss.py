from imports import Literal, torch


class FirstSpikeLoss(torch.nn.Module):
    def __init__(
        self,
        on_target = 0,
        off_target = -1,
        alpha=0.01,
        wrong_weight=5.0,
        no_spike_weight=1.0,
        eps=1e-8,
    ):
        super().__init__()

        self.alpha = alpha
        self.wrong_weight = wrong_weight
        self.no_spike_weight = no_spike_weight
        self.eps = eps

        self.on_target = on_target
        self.off_target = off_target

    # # TODO: Think long and hard about this function
    # # I think it should work a tiny bit differently.
    def forward(self, spikes, target):
        """
        spikes : [T, B, 2]
        target : [B]
        """

        T, B, N = spikes.shape
        assert N == 2

        # Probability that no spike has happened before time t
        no_prev = torch.cumprod(
            torch.cat([
                torch.ones(1, B, N, device=spikes.device),
                1 - spikes[:-1]
            ], dim=0),
            dim=0
        )

        # Earlier spikes are worth more
        t = torch.arange(T, device=spikes.device).float()

        # First-spike probability
        p_first = spikes * no_prev
        # Probability of no spikes
        p_none = (1 - spikes).prod(dim=0)

        expected = (p_first * t[:, None, None]).sum(dim=0)
        expected = expected + T * p_none

        ideal = torch.full_like(expected, T)
        ideal[torch.arange(B), target] = 0

        loss = (expected - ideal).square()
        return loss.mean()

class MeanCELoss(torch.nn.Module):
    def __init__(
        self,
        intermediate_reduction: Literal["mean", "sum"] = "mean",
        reduction: Literal["mean", "sum", "none"] = "mean"
    ):
        super().__init__()
        if intermediate_reduction == "sum":
            self._inter_reduct = torch.sum
        elif intermediate_reduction == "mean":
            self._inter_reduct = torch.mean
        else:
            raise ValueError("Did not get valid intermediate reduction. "
                             f"Expected 'mean' or 'sum', got {intermediate_reduction} instead")
        self._CE = torch.nn.CrossEntropyLoss(reduction = reduction)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # prediction has a shape of Time, Batch, Neurons
        prediction = self._inter_reduct(prediction, dim = 0)

        return self._CE(prediction, target)

class SpikemaxLoss(torch.nn.Module):
    def __init__(self, window_size=30, tau_s=1.0, tau_r=1.0):
        super().__init__()
        self.window_size = window_size
        self.tau_s = tau_s
        self.tau_r = tau_r

    def forward(self, outputs, targets):
        T, B, N = outputs.size()    # noqa: RUF059

        # Calculate spike count for each neuron in the sliding window
        spike_counts = []
        for t in range(T):
            if t < self.window_size:
                spike_count = torch.sum(outputs[:, :t+1], dim=1)
            else:
                spike_count = torch.sum(outputs[:, t-self.window_size:t+1], dim=1)
            spike_counts.append(spike_count.unsqueeze(1))
        spike_counts = torch.cat(spike_counts, dim=1)

        # Calculate global probability estimates
        global_probs = spike_counts / self.window_size

        # Calculate negative log-likelihood loss
        loss = -torch.mean(targets * torch.log(global_probs + 1e-8))

        return loss