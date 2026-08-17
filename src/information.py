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

    @staticmethod
    def _prepare_labels(
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        classes = torch.unique(labels, sorted=True)

        encoded = torch.empty_like(labels, dtype=torch.long)

        for i, cls in enumerate(classes):
            encoded[labels == cls] = i

        return encoded, classes

    @staticmethod
    def _split(
        n: int,
        train_fraction: float,
        validation_fraction: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if train_fraction <= 0 or validation_fraction < 0:
            raise ValueError("Invalid train/validation fractions.")

        if train_fraction + validation_fraction >= 1:
            raise ValueError(
                "train_fraction + validation_fraction must be < 1."
            )

        indices = torch.randperm(n, generator=TORCH_RNG)

        n_train = int(n * train_fraction)
        n_validation = int(n * validation_fraction)

        train_idx = indices[:n_train]
        validation_idx = indices[
            n_train:n_train + n_validation
        ]
        test_idx = indices[
            n_train + n_validation:
        ]

        return train_idx, validation_idx, test_idx

    def fit(
        self,
        responses: torch.Tensor,
        labels: torch.Tensor,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.15,
        batch_size: int = 128,
        verbose: bool = True,
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

        if responses.ndim != 2:
            raise ValueError(
                "responses must have shape [samples, features]. "
                f"Got {tuple(responses.shape)}."
            )

        if labels.ndim != 1:
            labels = labels.flatten()

        if responses.shape[0] != labels.shape[0]:
            raise ValueError(
                "Number of responses and labels differs."
            )

        # Make labels contiguous: e.g. {2, 4, 7} -> {0, 1, 2}
        encoded_labels, classes = self._prepare_labels(
            labels.cpu()
        )

        self.classes = classes

        train_idx, validation_idx, test_idx = self._split(
            n=responses.shape[0],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )

        x_train = responses[train_idx].float()
        y_train = encoded_labels[train_idx]

        x_validation = responses[validation_idx].float()
        y_validation = encoded_labels[validation_idx]

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        x_validation = x_validation.to(self.device)
        y_validation = y_validation.to(self.device)

        self.decoder = InformationDecoder(
            input_dim=responses.shape[1],
            num_classes=len(classes),
            hidden_dim=self.hidden_dim,
        )# .to(self.device)

        optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        criterion = torch.nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(
            x_train,
            y_train,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        best_validation_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        self.train_loss = []
        self.validation_loss = []

        iterator = range(self.max_epochs)

        if verbose:
            iterator = tqdm.tqdm(
                iterator,
                desc="Training Heller decoder",
            )

        for _ in iterator:

            self.decoder.train()

            running_loss = 0.0
            n_samples = 0

            for x_batch, y_batch in loader:

                optimizer.zero_grad()

                logits = self.decoder(x_batch)
                loss = criterion(logits, y_batch)

                loss.backward()
                optimizer.step()

                running_loss += (
                    loss.item() * x_batch.shape[0]
                )
                n_samples += x_batch.shape[0]

            train_loss = running_loss / n_samples

            self.decoder.eval()

            with torch.no_grad():
                validation_logits = self.decoder(
                    x_validation
                )

                validation_loss = criterion(
                    validation_logits,
                    y_validation,
                ).item()

            self.train_loss.append(train_loss)
            self.validation_loss.append(validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in self.decoder.state_dict().items()
                }

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        if best_state is not None:
            self.decoder.load_state_dict(best_state)

        self.decoder.eval()

        # Store the test split for estimate()
        self._test_responses = (
            responses[test_idx].float().to(self.device)
        )
        self._test_labels = encoded_labels[test_idx].to(
            self.device
        )

        return self

    @torch.no_grad()
    def posterior(
        self,
        responses: torch.Tensor,
        batch_size: int = 4096,
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
        responses: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
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

        if responses is None:
            responses = self._test_responses

        if labels is None:
            labels = self._test_labels

        labels = labels.flatten().long().to(self.device)

        probabilities = self.posterior(
            responses,
            batch_size=batch_size,
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
        responses: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:

        """
        Same as estimate(), but also returns useful diagnostics.
        """

        if responses is None:
            responses = self._test_responses

        if labels is None:
            labels = self._test_labels

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