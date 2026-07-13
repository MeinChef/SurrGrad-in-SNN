from imports import torch
from imports import tqdm
from imports import DEVICE
from torcheval.metrics import functional

class MLP(torch.nn.Module):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__()

        self.con1 = torch.nn.Linear(
            in_features = 10000,
            out_features = config["neurons_hidden_1"],
            device = DEVICE
        )
        self.neuron1 = torch.nn.Sigmoid()
        self.con2 = torch.nn.Linear(
            in_features = config["neurons_hidden_1"],
            out_features = config["neurons_hidden_2"],
            device = DEVICE
        )
        self.neuron2 = torch.nn.Sigmoid()
        self.con3 = torch.nn.Linear(
            in_features = config["neurons_hidden_2"],
            out_features = config["neurons_out"],
            device = DEVICE
        )
        self.neuron3 = torch.nn.Softmax(dim = 0)

        self.lossfn = torch.nn.BCELoss()
        self.acc = functional.multiclass_accuracy
        self.optim = torch.optim.Adam(self.parameters())
        self._best_loss = torch.inf

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.con1(x)
        x = self.neuron1(x)
        x = self.con2(x)
        x = self.neuron2(x)
        x = self.con3(x)
        x = self.neuron3(x)
        return x

    def fit(
        self,
        data: torch.utils.data.DataLoader
    ) -> tuple[list, list]:
        self.train()

        loss_hist = []
        acc_hist = []

        for i, (x, target) in tqdm.tqdm(enumerate(data)):
            x = x.permute(1,0,-1)
            x = x.flatten(start_dim = 1)
            acctarget = target.clone()
            target = torch.nn.functional.one_hot(target, num_classes = 2)
            target = target.to(torch.float)
            acctarget = acctarget.to(DEVICE)


            if x.device != DEVICE:
                x = x.to(DEVICE)
            if target.device != DEVICE:
                target = target.to(DEVICE)
            pred = self(x)

            loss = self.lossfn(pred, target)
            acc = self.acc(pred, acctarget, num_classes = 2)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            loss_hist.append(loss.item())
            acc_hist.append(acc)

        return loss_hist, acc_hist

    def evaluate(
        self,
        data: torch.utils.data.DataLoader
    ) -> tuple[list, list]:
        # pre-define variables
        loss_hist = []
        acc_hist  = []
        # set model in evaulating mode
        self.eval()


        # test loop
        with torch.no_grad():
            for i, (x, target) in tqdm.tqdm(enumerate(data)):

                x = x.permute(1,0,-1)
                x = x.flatten(start_dim = 1)
                acctarget = target.clone()
                target = torch.nn.functional.one_hot(target, num_classes = 2)
                target = target.to(torch.float)

                # move tensors to device
                x = x.to(DEVICE)
                target = target.to(DEVICE)
                acctarget = acctarget.to(DEVICE)

                pred = self.forward(x)

                loss = self.lossfn(pred, target)
                acc = self.acc(pred, acctarget, num_classes = 2)

                loss_hist.append(loss.item())
                acc_hist.append(acc)

        # update best loss
        self._best_loss = min(
            self._best_loss,
            torch.tensor(loss_hist).mean()
        )

        return loss_hist, acc_hist