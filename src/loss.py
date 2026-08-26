from imports import Literal, torch


class MeanCELoss(torch.nn.Module):
    def __init__(
        self,
        intermediate_reduction: Literal["mean", "sum"] = "mean",    # noqa: F821 (Undefined Literal)
        reduction: Literal["mean", "sum", "none"] = "mean"          # noqa: F821 (Undefined Literal)
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