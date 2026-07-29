import torch
from typing import Literal

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




    #     weights = torch.exp(-self.alpha * t).view(T, 1, 1)

    #     score = (first * weights).sum(0)       # [B,2]

    #     target_score = score.gather(
    #         1,
    #         target[:, None]
    #     ).squeeze(1)

    #     wrong_score = score.gather(
    #         1,
    #         (1 - target)[:, None]
    #     ).squeeze(1)

    #     classification = -torch.log(target_score + self.eps)

    #     # wrong_penalty = wrong_score

    #     # Penalise neither neuron ever spiking
    #     any_spike = spikes.sum(dim=(0, 2))
    #     no_spike = (1 - any_spike).clamp(min=0)

    #     loss = (
    #         classification
    #         + self.wrong_weight * wrong_score
    #         + self.no_spike_weight * no_spike
    #     )
    #     if loss.isnan().any():
    #         breakpoint()
    #     return loss.mean()

    # non differentiable because of nonzero, ayyyy
    # def forward(
    #     self,
    #     spikes: torch.Tensor,
    #     target: torch.Tensor,
    # ) -> torch.Tensor:

    #     T, B, N = spikes.shape
    #     assert N == 2

    #     # off_target is T if -1
    #     if self.off_target == -1:
    #         self.off_target = T

    #     times, batches, classes = spikes.nonzero(as_tuple=True)

    #     # First spike times, initialized to T ("no spike")
    #     first = torch.full(
    #         (B * N,),
    #         self.off_target,
    #         device = spikes.device,
    #         dtype = times.dtype,
    #     )

    #     # flatten the indices manually
    #     flat_idx = batches * N + classes

    #     first.scatter_reduce_(
    #         0,                  # reduce the first dimension, B
    #         flat_idx,           # tell where the values go
    #         times,              # the values to put in there
    #         reduce = "amin",    # oh, and apply the argmin operation
    #         include_self = True,
    #     )

    #     # reshape to be unflattened
    #     first = first.view(B, N)

    #     # make the ideal values, and for every batch the target should be 0
    #     ideal = torch.full_like(first, self.off_target)
    #     ideal[torch.arange(B, device = spikes.device), target] = self.on_target

    #     # (M)SE calculation
    #     loss = (ideal - first).float().square()

    #     # and reduction (M)
    #     return loss.mean()

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
