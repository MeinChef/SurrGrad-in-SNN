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

    
