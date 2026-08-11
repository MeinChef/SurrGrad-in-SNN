from imports import NOW, Figure, Path, os, pickle, plt, warnings
from imports import numpy as np


class MetricsTracker:
    def __init__(
        self,
        path: str | None = None
    ) -> None:
        """
        :param path: in which directory to save stuff
        :type path: str
        """
        if path is None:
            self.path = os.path.join(
                Path(__file__).parent.parent,
                "img",
                NOW
            )
        else:
            self.path = path

        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def update_train(
        self,
        loss: list,
        acc: list
    ) -> None:
        self.train_loss.append(loss)
        self.train_acc.append(acc)

    def update_val(
        self,
        loss: list,
        acc: list
    ) -> None:
        self.val_loss.append(loss)
        self.val_acc.append(acc)

    def save(
        self,
        force: bool = False
    ) -> None:

        trainfile = os.path.join(self.path, "train-metrics.pkl")
        if not os.path.exists(trainfile):
            with open(trainfile, "wb+") as file:
                pickle.dump(
                    (self.train_loss, self.train_acc),
                    file
                )
        else:
            if force:
                # save anyway
                with open(trainfile, "wb+") as file:
                    pickle.dump(
                        (self.train_loss, self.train_acc),
                        file
                    )
            else:
                warnings.warn(f"File {trainfile} already exists, not saving metrics again.")

        testfile = os.path.join(self.path, "test-metrics.pkl")
        if not os.path.exists(testfile):
            with open(testfile, "wb+") as file:
                pickle.dump(
                    (self.val_loss, self.val_acc),
                    file
                )
        else:
            if force:
                with open(testfile, "wb+") as file:
                    pickle.dump(
                        (self.val_loss, self.val_acc),
                        file
                    )
            else:
                warnings.warn(f"File {testfile} already exists, not saving metrics again.")

    def load(
        self
    ) -> None:

        trainfile = os.path.join(self.path, "train-metrics.pkl")
        if  os.path.exists(trainfile):
            with open(trainfile, "rb") as file:
                self.train_loss, self.train_acc = pickle.load(
                    file
                )
        else:
            warnings.warn(f"Train metrics file {trainfile} not found.")

        testfile = os.path.join(self.path, "test-metrics.pkl")
        if not os.path.exists(testfile):
            with open(testfile, "rb") as file:
                self.val_loss, self.val_acc = pickle.load(
                    file
                )
        else:
            warnings.warn(f"Train metrics file {trainfile} not found.")

    def announce(
        self,
        train = False
    ) -> None:
        mode = "Training" if train else "Testing"

        if train:
            loss = self.train_loss
            acc  = self.train_acc
        else:
            loss = self.val_loss
            acc  = self.val_acc

        print(
            f"Model achieved during {mode}:\n"
            f"Loss of {np.asarray(loss).mean()} "
            f"and Accuracy of {np.asarray(acc).mean() * 100}%"
        )


    def plot(
        self,
        train: bool = False,
        save: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> Figure:
        """
        Plot training loss over steps for multiple epochs.


        :param epoch_losses: A list where each sublist contains loss values for one epoch.
        :type epoch_losses: list[list[float]]
        :param epoch_acc: A list where each sublist contains accuracy values for one epoch.
        :type epoch_acc: list[list[float]]
        :param figsize: Figure size for matplotlib.
        :type figsize: tuple[float, float]
        """

        # make stuff more accessible
        label = "train" if train else "test"
        if train:
            losses = self.train_loss
            accuracies = self.train_acc
        else:
            losses = self.val_loss
            accuracies = self.val_acc

        epochs = len(losses)
        steps_per_epoch = len(losses[0])


        # create figure
        fig = plt.figure(
            num = "losses",
            figsize = figsize,
            dpi = 300,
            clear = True
        )
        axes = fig.subplots(1,1)

        # convert lists to numpy arrays
        losses = np.concatenate(
            [np.asarray(epoch) for epoch in losses] # type: ignore
        )
        accuracies = np.concatenate(
            [np.asarray(epoch) for epoch in accuracies] # type: ignore
        )

        # and make the xaxis
        steps = np.arange(losses.size) / steps_per_epoch

        # and plot
        axes.plot(
            steps,
            losses,
            label = "Losses",
            color = "blue"
        )
        secax = axes.twinx()
        secax.plot(
            steps,
            accuracies,
            label = "Accuracies",
            color = "orange"
        )

        axes.set_xlabel(f"{label.title()}ing Epoch")
        axes.set_ylabel("Loss")
        secax.set_ylabel("Accuracies")
        axes.set_title(f"{label.title()}ing Loss over Epochs")

        # legend
        lines1, labels1 = axes.get_legend_handles_labels()
        lines2, labels2 = secax.get_legend_handles_labels()
        axes.legend(lines1 + lines2, labels1 + labels2)

        axes.grid(True)
        fig.tight_layout()

        # save figure
        if save:
            # create folder if it didn't exist
            folderpth = os.path.join(
                Path(__file__).parent.parent,
                "img",
                NOW,
            )
            os.makedirs(
                folderpth,
                exist_ok = True
            )

            fig.savefig(
                os.path.join(
                    folderpth,
                    f"{label}-metrics.pdf"
                ),
                format = "pdf"
            )
            fig.clear()

        return fig