import misc
import data
from model import Model


def main() -> None:
    config_data, config_model = data.load_config()
    train, test, num_classes = data.data_prep(config_data)
    
    if config_data["DEBUG"]:
        exit(0)

    config_model["num_classes"] = num_classes
    model = Model(config = config_model)

    # training
    for epoch in range(config_model["epochs"]):
        
        loss, acc, rec = model.train_loop(data = train)
        misc.stats_to_file(
            config_data,
            loss,
            acc,
        )

        loss, acc, rec = model.test_loop(data = test)
        misc.stats_to_file(
            config_data,
            loss,
            acc,
            rec,
            identifier = f"test-ep{epoch}"
        )

        # reset counter needed for recording hidden layers
        model.reset()

    # misc.plot_loss_acc(config_data)
    # misc.cleanup(config = config_data)
    

if __name__ == "__main__":
    # comment next line out, if you have given your directory a custom name 
    misc.check_working_directory()
    
    main()

    
