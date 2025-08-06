import misc
import data
from model import Model


def main() -> None:
    config_data, config_model = data.load_config()
    train, test, num_classes = data.data_prep(config_data)
    handler = data.DataHandler(
        config_data,
        )

    if config_data["DEBUG"]:
        exit(0)

    print("Train", misc.get_sample_distribution(train))
    print("Test", misc.get_sample_distribution(test))
    exit(0)

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
        handler.flush_to_file(
            loss,
            acc,
            rec,
            identifier = f"test-ep{epoch}"
        )
        loss_hist_test.append(loss)

        # reset counter needed for recording hidden layers
        model.reset()

    handler.plot_spikes(
        identifier = "test",
        infer_epochs = True
    )
    # misc.plot_loss_acc(config_data)
    # misc.cleanup(config = config_data)
    

if __name__ == "__main__":
    # comment next line out, if you have given your directory a custom name 
    misc.check_working_directory()
    
    main()

    
