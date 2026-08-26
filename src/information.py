from imports import DEVICE, TORCH_RNG, os, torch


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
        num_classes: int = 2,
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

    :param hidden_dim: Number of hidden units in the decoder. Heller et al. used 6.
    :type hidden_dim: int
    :param learning_rate: Learning rate for the decoder.
    :type learning_rate: float
    :param weight_decay: L2 regularisation.
    :type weight_decay: float
    :param max_epochs: Maximum number of decoder training epochs.
    :type max_epochs: int
    :param patience: Early-stopping patience measured in validation epochs.
    :type patience: int
    :param prior: Prior distribution for the stimulus. 
        - "decoder": Uses P(s) = mean_r O_s(r), as in Heller et al. 
        - "labels": Uses the empirical class distribution.
    :type prior: str
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
        self.validation_loss = []

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

    def _prep_data(
        self,
        responses, labels,
        train_frac, val_frac,
        batch_size: int = 128
    ):
        train_idx, val_idx, test_idx = self._split(
            responses.shape[0],
            train_frac, val_frac
        )
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        x_train = responses[train_idx].float()
        x_val   = responses[val_idx  ].float()
        x_test  = responses[test_idx ].float()

        y_train = labels[train_idx]
        y_val   = labels[val_idx  ]
        y_test  = labels[test_idx ]

        ds_train = torch.utils.data.TensorDataset(x_train, y_train)
        ds_val   = torch.utils.data.TensorDataset(x_val, y_val)
        ds_test  = torch.utils.data.TensorDataset(x_test, y_test)

        ld_train = torch.utils.data.DataLoader(ds_train, batch_size, True)
        ld_val   = torch.utils.data.DataLoader(ds_val,   batch_size, True)
        ld_test  = torch.utils.data.DataLoader(ds_test,  batch_size, True)

        return ld_train, ld_val, ld_test


    def fit(
        self,
        responses: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int = 2,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.15,
        batch_size: int = 128,
    ) -> "InformationEstimator":

        """
        Fit P(S|R).

        :param responses: Tensor of shape [samples, features].
        :type responses: torch.Tensor
        :param labels: Tensor of shape [samples], representing class labels.
        :type labels: torch.Tensor
        :returns: Returns self to allow method chaining.
        :rtype: self
        """

        self.time_steps = responses.shape[1]

        self.decoder = InformationDecoder(
            input_dim = self.time_steps,
            num_classes = num_classes,
            hidden_dim = self.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weight_decay,
        )

        lossfn = torch.nn.CrossEntropyLoss()

        train, val, test = self._prep_data(
            responses, labels,
            train_fraction, validation_fraction,
            batch_size
        )
        self.test = test

        best_validation_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        self.validation_loss = []
        for _ in range(self.max_epochs):
            val_loss = []

            self.decoder.train()
            for x, label in train:
                x = x.to(DEVICE)
                label = label.to(DEVICE)
                pred = self.decoder(x)
                loss = lossfn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # don't care about the train loss, thus no saving that

            self.decoder.eval()
            with torch.no_grad():
                for x, label in val:
                    x = x.to(DEVICE)
                    label = label.to(DEVICE)
                    pred = self.decoder(x)
                    loss = lossfn(pred, label).item()
                    val_loss.append(loss)

            epoch_val_loss = torch.mean(torch.tensor(val_loss)).cpu()
            self.validation_loss.append(epoch_val_loss)

            # early stopping: if validation loss doesn't improve
            # for patience epochs, stop training to prevent overfitting.
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
        data: torch.utils.data.DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        """
        Calculate O_s(r) = estimated P(S=s | R=r).

        :param data: Data that the posterior distribution will be applied on
        :type data: torch.utils.data.DataLoader
        :returns: Probabilities and Labels
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        if self.decoder is None:
            raise RuntimeError(
                "The decoder has not been fitted yet."
            )

        probabilities = []
        labels = []

        for x, label in data:

            logits = self.decoder(x)

            # softmax to convert logits to probabilities
            probabilities.append(
                torch.softmax(logits, dim = -1)
            )
            labels.append(label)

        return torch.cat(probabilities, dim = 0), torch.cat(labels, dim = 0)

    @torch.no_grad()
    def estimate(
        self,
        test: torch.utils.data.DataLoader | None =  None,
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
        if test is None:
            test = self.test

        probabilities, labels = self.posterior(test)

        # ---------------------------------------------------------
        # Estimate P(S)
        # ---------------------------------------------------------

        if self.prior == "decoder":
            # This is Heller et al.'s Eq. (3):
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

        # numerical protection.
        eps = torch.finfo(probabilities.dtype).tiny

        # information estimate: I(S;R) = E[log2( P(S|X) / P(S) )]
        # equivalent to the expected log-likelihood ratio 
        # under the estimated posterior
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
        test: torch.utils.data.DataLoader | None = None,
    ) -> dict:

        """
        Same as estimate(), but also returns useful diagnostics.
        """
        if self.decoder is None:
            raise RuntimeError(
                "The decoder has not been fitted yet."
            )

        if test is None:
            test = self.test

        probabilities, labels = self.posterior(test)
        probabilities = probabilities.to(self.device)
        labels = labels.to(self.device)

        if self.prior == "decoder":
            prior = probabilities.mean(dim = 0)
        else:
            prior = torch.bincount(
                labels,
                minlength = probabilities.shape[1],
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

        # p_correct: P(s_n | r_n) for the true class s_n
        # prior[labels]: P(s_n) for each sample (broadcasted)
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
            # "classes": self.classes,
        }

    def save(
        self,
        path: str,
        ident: str
    ) -> None:

        path = os.path.join(path, "estim")
        os.makedirs(path, exist_ok = True)

        if self.decoder is None:
            raise RuntimeError(
                "Cannot save an unfitted estimator."
            )

        torch.save(
            {
                "decoder_state_dict": self.decoder.state_dict(),

                # save indices to allow reproducible splits later
                "train_idx": self.train_idx,
                "val_idx": self.val_idx,
                "test_idx": self.test_idx,

                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "prior": self.prior,

                "validation_loss": self.validation_loss,
            },
            os.path.join(path, ident),
        )

    @classmethod
    def load(
        cls,
        path,
        ident,
        input_dim: int,
    ):
        # input_dim is required because the decoder architecture depends on it
        # (we can't reconstruct the model without knowing the input size)
        path = os.path.join(path, "estim")

        checkpoint = torch.load(
            os.path.join(path, ident),
            map_location="cpu",
        )

        estimator = cls(
            hidden_dim=checkpoint["hidden_dim"],
            learning_rate=checkpoint["learning_rate"],
            weight_decay=checkpoint["weight_decay"],
            max_epochs=checkpoint["max_epochs"],
            patience=checkpoint["patience"],
            prior=checkpoint["prior"],
            device=DEVICE,
        )

        estimator.train_idx = checkpoint["train_idx"]
        estimator.val_idx = checkpoint["val_idx"]
        estimator.test_idx = checkpoint["test_idx"]

        estimator.decoder = InformationDecoder(
            input_dim=input_dim,
            hidden_dim=estimator.hidden_dim,
        ).to(estimator.device)

        estimator.decoder.load_state_dict(
            checkpoint["decoder_state_dict"]
        )

        estimator.decoder.eval()

        estimator.validation_loss = checkpoint[
            "validation_loss"
        ]

        return estimator