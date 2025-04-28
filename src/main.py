import misc
import data
from imports import torch
from imports import time
from imports import functional
from model import Model


def main() -> None:
    config_data, config_model = data.load_config()
    train, test = data.data_prep(config_data)

    model = Model(config = config_model)

    # setter for rate encoding
    # setter functions
    # model.set_loss(functional.loss.mse_count_loss(correct_rate = 0.8, incorrect_rate = 0.2))
    # model.set_acc(functional.acc.accuracy_rate)
    # model.set_optim(
    #     torch.optim.Adam(
    #         model.parameters(), 
    #         lr = config_model["learning_rate"], 
    #         betas = (0.9, 0.99)
    #     )
    # )

    # setter for latency encoding
    model.set_loss(functional.loss.ce_temporal_loss())
    model.set_acc(functional.acc.accuracy_temporal)
    model.set_optim(
        torch.optim.Adam(
            model.parameters(), 
            lr = config_model["learning_rate"], 
            betas = (0.9, 0.99)
        )
    )


    # training
    for epoch in range(config_model["epochs"]):
        loss, acc, rec = model.train_loop(data = train)
        
        misc.stats_to_file(
            config_data,
            loss,
            acc
        )
        # model spikes in Memory usage after Training (~486 Batches a 128)
        model.test_loop(data = test)


    

if __name__ == "__main__":
    # comment next line out, if you have given your directory a custom name 
    misc.check_working_directory()
    
    main()

    
