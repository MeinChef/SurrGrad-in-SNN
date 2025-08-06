import misc
import data
from model import Model
from imports import functional
from imports import torch


def main() -> None:
    config_data, config_model = data.load_config()
    train, test, num_classes = data.data_prep(config_data)
    handler = data.DataHandler(
        config_data,
        )

    # if config_data["DEBUG"]:
    #     exit(0)

    # print("Train", misc.get_sample_distribution(train))
    # print("Test", misc.get_sample_distribution(test))
    # exit(0)

    config_model["num_classes"] = num_classes
    model = Model(config = config_model)


    loss_hist_train = []
    loss_hist_test = []
    # training
    for epoch in range(config_model["epochs"]):
        
        loss, acc, rec = model.fit(data = train)
        handler.flush_to_file(
            loss,
            acc,
        )
        loss_hist_train.append(loss)

        loss, acc, rec = model.evaluate(data = test)
        # handler.flush_to_file(
        #     loss,
        #     acc,
        #     # rec,
        #     # identifier = f"test-ep{epoch}"
        # )
        loss_hist_test.append(loss)

        # reset counter needed for recording hidden layers
        model.reset()

    # handler.plot_spikes(
    #     identifier = "test",
    #     infer_epochs = True
    # )
    misc.plot_loss_acc(config_data)
    # misc.cleanup(config = config_data)
    

def main_loss_calc():
    config_data, config_model = data.load_config()
    train, test, num_classes = data.data_prep(config_data)

    config_model["num_classes"] = num_classes
    model = Model(config = config_model)
    model.expand_config(train)
    model.__init_tensors__()
    model.config["expanded"] = True

    # lossfn = functional.loss.ce_rate_loss()
    lossfn = functional.loss.ce_temporal_loss()

    x, y = next(iter(train))
    x = x.to(model.device)
    y = y.to(model.device)
    model.cur_steps = x.shape[0]
    model.train()

    # what happens on a single forward and backward pass?
    params_before = [p.clone().detach() for p in model.parameters()]
    pred = model(x)
    loss = lossfn(pred, y)
    model.optim.zero_grad()
    loss.backward()
    model.optim.step()

    params_after = [p.clone().detach() for p in model.parameters()]
    equality = [
        torch.equal(p0, p1) for p0, p1 in zip(params_before, params_after)
    ]
    print(f"No of spikes in prediction: {pred.sum().item()}")
    print(f"Are parameters different after a single forward and backward pass?\n{equality}")
    print(f"Loss value for real prediction: {loss.item()}")


    # what happens when we are using a great prediction?
    params_before = params_after

    one_hot = torch.nn.functional.one_hot(y, num_classes = 10)  # shape: [batch, 10]
    one_hot = one_hot.unsqueeze(0).expand(pred.shape[0], -1, -1)  # shape: [steps, batch, 10]
    pred.copy_(one_hot)
    loss = lossfn(pred, y)
    print(f"Loss value for perfect prediction: {loss.item()}")


    # what happens when we are using a random prediction?
    params_before = params_after

    pred = torch.randint_like(pred, low = 0, high = 2)
    loss = lossfn(pred, y)
    print(f"Loss value for random prediction: {loss.item()}")




if __name__ == "__main__":
    # comment next line out, if you have given your directory a custom name 
    misc.check_working_directory()
    
    main()
    # main_loss_calc()
    
