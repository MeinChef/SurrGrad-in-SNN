from imports import torch

class FirstSpikeLoss(torch.nn.Module):
    def __init__(
        self,
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

    # TODO: Think long and hard about this function
    # I think it should work a tiny bit differently.
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
        # clamp to non-inf realm
        no_prev = no_prev.clamp(
            min = -1e25,
            max = 1e25
        )

        # First-spike probability
        first = spikes * no_prev

        # Earlier spikes are worth more
        t = torch.arange(T, device=spikes.device).float()
        weights = torch.exp(-self.alpha * t).view(T, 1, 1)

        score = (first * weights).sum(0)       # [B,2]

        target_score = score.gather(
            1,
            target[:, None]
        ).squeeze(1)

        wrong_score = score.gather(
            1,
            (1 - target)[:, None]
        ).squeeze(1)

        classification = -torch.log(target_score + self.eps)

        # wrong_penalty = wrong_score

        # Penalise neither neuron ever spiking
        any_spike = spikes.sum(dim=(0, 2))
        no_spike = (1 - any_spike).clamp(min=0)

        loss = (
            classification
            + self.wrong_weight * wrong_score
            + self.no_spike_weight * no_spike
        )
        breakpoint()
        if loss.isnan().any():
            breakpoint()
        return loss.mean()