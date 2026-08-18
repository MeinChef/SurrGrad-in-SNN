from imports import DEVICE, TORCH_RNG, torch, tqdm


class InformationDecoder(torch.nn.Module):
    """
    Neural decoder estimating P(S | R).

    This follows the basic architecture used by Heller et al.:
        R -> tanh hidden layer -> softmax output

    The default hidden dimension of 6 mirrors their original analysis.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 6,
    ) -> None:
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, num_classes),
        )
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)



class InformationEstimator:
    """
    Decoder-based estimator of I(S;R), following Heller et al. (1995).

    Parameters
    ----------
    hidden_dim:
        Number of hidden units in the decoder.
        Heller et al. used 6.

    learning_rate:
        Learning rate for the decoder.

    weight_decay:
        L2 regularisation.

    max_epochs:
        Maximum number of decoder training epochs.

    patience:
        Early-stopping patience measured in validation epochs.

    prior:
        "decoder" reproduces Heller's estimator:
            P(s) = mean_r O_s(r)

        "labels" instead uses the empirical class distribution.
    """

    def __init__(
        self,
        hidden_dim: int = 6,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        max_epochs: int = 500,
        patience: int = 25,
        prior: str = "decoder",
        device: torch.device | None = None,
    ) -> None:

        if prior not in {"decoder", "labels"}:
            raise ValueError(
                "prior must be either 'decoder' or 'labels'."
            )

        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.prior = prior

        self.device = device if device is not None else DEVICE

        self.decoder = None
        self.classes = None
        self.train_loss = []
        self.validation_loss = []

    def fit(
        self,
        train: torch.utils.data.TensorDataset | torch.utils.data.Subset,
        val: torch.utils.data.TensorDataset | torch.utils.data.Subset,
        time_steps: int = 1000,
        num_classes: int = 2,
        batch_size: int = 128,
    ) -> "InformationEstimator":

        """
        Fit P(S|R).

        Parameters
        ----------
        responses:
            Tensor [samples, features].

        labels:
            Tensor [samples].

        Returns
        -------
        self
        """

        self.decoder = InformationDecoder(
            input_dim = time_steps,
            num_classes = num_classes,
            hidden_dim = self.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weight_decay,
        )

        lossfn = torch.nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(
            train,
            batch_size = batch_size,
            shuffle = True,
        )
        val_loader = torch.utils.data.DataLoader(
            val, 
            batch_size = batch_size,
            shuffle = True
        )

        best_validation_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        self.train_loss = []
        self.validation_loss = []
        for _ in tqdm.tqdm(
            range(self.max_epochs),
            total = self.max_epochs,
            desc = "Training InformationDecoder",
        ):
            # running_loss = 0.0
            # n_samples = 0
            val_loss = []

            self.decoder.train()
            for x, label in train_loader:
                pred = self.decoder(x)
                loss = lossfn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # don't care about the train loss, thus no saving that

            #     running_loss += (
            #         loss.item() * x.shape[0]
            #     )
            #     n_samples += x.shape[0]   # tf is this

            # train_loss = running_loss / n_samples

            self.decoder.eval()
            with torch.no_grad():
                for x, label in val_loader:
                    pred = self.decoder(x)
                    loss = lossfn(pred, label).item()
                    val_loss.append(loss)

            epoch_val_loss = torch.mean(torch.tensor(val_loss))

            if epoch_val_loss < best_validation_loss:
                best_validation_loss = epoch_val_loss

                # save state of decoder
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in self.decoder.state_dict().items()
                }
                # and reset patience counter
                epochs_without_improvement = 0

            else:
                # increase patience counter
                epochs_without_improvement += 1

            # womp womp training done
            if epochs_without_improvement >= self.patience:
                break

        if best_state is not None:
            self.decoder.load_state_dict(best_state)

        self.decoder.eval()
        return self

    @torch.no_grad()
    def posterior(
        self,
        responses: torch.Tensor,
        batch_size: int = 128,
    ) -> torch.Tensor:

        """
        Calculate O_s(r) = estimated P(S=s | R=r).

        Returns
        -------
        Tensor [samples, classes].
        """

        if self.decoder is None:
            raise RuntimeError(
                "The decoder has not been fitted yet."
            )

        responses = responses.float().to(self.device)

        probabilities = []

        for batch in responses.split(batch_size):

            logits = self.decoder(batch)

            probabilities.append(
                torch.softmax(logits, dim=-1)
            )

        return torch.cat(probabilities, dim=0)

    @torch.no_grad()
    def estimate(
        self,
        test: torch.utils.data.TensorDataset | torch.utils.data.Subset,
        batch_size: int = 4096,
    ) -> float:

        """
        Calculate the Heller information estimate in bits.

        If responses/labels are omitted, the held-out test set from
        fit() is used.
        """

        if self.decoder is None:
            raise RuntimeError(
                "The decoder has not been fitted yet."
            )
        # get x, labels from tensordataset
        if isinstance(test, torch.utils.data.TensorDataset):
            responses, labels = test.tensors
        elif isinstance(test, torch.utils.data.Subset):
            raw_responses = test.dataset.tensors[0]         # pyright: ignore[reportAttributeAccessIssue] (stupid linter doesn't know)
            raw_labels    = test.dataset.tensors[1]         # pyright: ignore[reportAttributeAccessIssue] (that these are tensordatasets)
            responses = raw_responses[test.indices]
            labels    =    raw_labels[test.indices]
        else:
            raise TypeError(f"test must be a TensorDataset or Subset. Got {type(test)} instead.")

        probabilities = self.posterior(
            responses,
            batch_size = batch_size,
        )

        # ---------------------------------------------------------
        # Estimate P(S)
        # ---------------------------------------------------------

        if self.prior == "decoder":
            # This is Heller et al.'s Eq. (3):
            #
            # P(s) = 1/N sum_n O_s(r_n)
            prior = probabilities.mean(dim=0)

        else:
            prior = torch.bincount(
                labels,
                minlength=probabilities.shape[1],
            ).float()

            prior /= prior.sum()

        # ---------------------------------------------------------
        # P(s_n | r_n)
        # ---------------------------------------------------------

        p_correct = probabilities[
            torch.arange(
                labels.shape[0],
                device=self.device,
            ),
            labels,
        ]

        p_prior = prior[labels]

        # Numerical protection.
        eps = torch.finfo(probabilities.dtype).tiny

        information = torch.mean(
            torch.log2(
                p_correct.clamp_min(eps)
                /
                p_prior.clamp_min(eps)
            )
        )

        return information.item()

    @torch.no_grad()
    def estimate_with_details(
        self,
        test: torch.utils.data.TensorDataset | torch.utils.data.Subset,
        batch_size: int = 128
    ) -> dict:

        """
        Same as estimate(), but also returns useful diagnostics.
        """
        if self.decoder is None:
            raise RuntimeError(
                "The decoder has not been fitted yet."
            )
        # get x, labels from tensordataset
        if isinstance(test, torch.utils.data.TensorDataset):
            responses, labels = test.tensors
        elif isinstance(test, torch.utils.data.Subset):
            raw_responses = test.dataset.tensors[0]         # pyright: ignore[reportAttributeAccessIssue] (stupid linter doesn't know)
            raw_labels    = test.dataset.tensors[1]         # pyright: ignore[reportAttributeAccessIssue] (that these are tensordatasets)
            responses = raw_responses[test.indices]
            labels    =    raw_labels[test.indices]
        else:
            raise TypeError(f"test must be a TensorDataset or Subset. Got {type(test)} instead.")

        probabilities = self.posterior(
            responses,
            batch_size = batch_size,
        )
        probabilities = self.posterior(responses)
        labels = labels.flatten().long().to(self.device)

        if self.prior == "decoder":
            prior = probabilities.mean(dim=0)
        else:
            prior = torch.bincount(
                labels,
                minlength=probabilities.shape[1],
            ).float()
            prior /= prior.sum()

        p_correct = probabilities[
            torch.arange(
                labels.shape[0],
                device=self.device,
            ),
            labels,
        ]

        eps = torch.finfo(probabilities.dtype).tiny

        information_per_sample = torch.log2(
            p_correct.clamp_min(eps)
            /
            prior[labels].clamp_min(eps)
        )

        predictions = probabilities.argmax(dim=1)

        accuracy = (
            predictions == labels
        ).float().mean().item()

        return {
            "information": information_per_sample.mean().item(),
            "information_per_sample": (
                information_per_sample.cpu()
            ),
            "decoder_accuracy": accuracy,
            "prior": prior.cpu(),
            "posterior": probabilities.cpu(),
            "classes": self.classes,
        }