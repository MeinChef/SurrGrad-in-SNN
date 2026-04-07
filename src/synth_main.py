from synth_data import DataGenerator
from synth_model import SynthModel
from data import load_config, DataHandler
from misc import check_working_directory
from imports import argparse
from imports import functional
from imports import snntorch as snn

def main(
    args: argparse.Namespace
):
    cfg_data, cfg_model = load_config()

    # initialise datagenerator
    print("Initialising Classes...")
    datagen = DataGenerator(
        time_steps = cfg_data["time_steps"]["val"],
        jitter = cfg_data["jitter"],
        neurons = cfg_data["neurons"]["val"],
        min_isi = cfg_data["min_isi"],
        max_isi = cfg_data["max_isi"],
        min_rate = cfg_data["min_rate"],
        max_rate = cfg_data["max_rate"],
        # precision = np.float32
    )

    # initialise model
    model = SynthModel(
        config = cfg_model,
    )

    # hot new shit
    recorder = functional.probe.OutputMonitor(
        net = model,
        instance = snn.Leaky
    )

    # 
    handler = DataHandler(
        time_steps = cfg_data["time_steps"]["val"],
        datapath = "data/"
    )
    print("Done!")

    # generate dataset
    print("Generating Data...")
    train, test = datagen.generate_dataset(
        no_samples = cfg_data["no_samples"],
        batch_size = cfg_data["batch_size"],
        train_split = cfg_data["train_split"],
        shuffle = cfg_data["shuffle"],
        prefetch = cfg_data["prefetch"],
    )

    curated = datagen.generate_dataset(
        no_samples = cfg_model["samples"],
        batch_size = 1,
        train_split = 0,
        shuffle = False,
        prefetch = cfg_data["prefetch"],
    )
    print("Done!")

    print("Training...")
    for e in range(cfg_model["epochs"]):
        print(f"Epoch: {e}")
        recorder.disable()
        loss, acc = model.fit(train)

        # handler.plot_loss_accuracy(
        #     loss = loss,
        #     accuracy = acc,
        #     training = True,
        #     epoch = e,
        #     filename = f"train-epoch{e}",
        #     show = False
        # )
        loss, acc = model.evaluate(
            data = test,
        )

        # do the recording of the hidden states
        recorder.enable()
        _, _ = model.evaluate(
            data = curated
        )

        handler.visualise(
            recorder = recorder
        )

        # loss, acc = model.evaluate(
        #     data = curated_test
        # )
        # handler.visualise(recorder)

        # save the spike recordings cleanly to a file
        # handler.flush_to_file(
        #     loss = loss,
        #     loss_ident = f"test-{e}",
        #     acc = acc,
        #     acc_ident = f"test-{e}",
        #     spk_rec = rec,
        #     spk_ident = None
        # )

        # plot the spikes and loss/accuracy
        # handler.plot_loss_accuracy(
        #     loss = loss,
        #     accuracy = acc,
        #     training = False,
        #     epoch = e,
        #     filename = f"test-epoch{e}",
        #     show = False
        # )
        recorder.clear_recorded_data()

        
    print("Success!")
    return True
    

def resolve_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        "-d",
        type = str,
        required = False,
        default = "data/",
        help = "Path to the data directory. Defaults to ./data/"
    )
    parser.add_argument(
        "--record-hidden",
        "-r",
        action = argparse.BooleanOptionalAction,
        required = False,
        help = "Flag whether to record the hidden layers and save them"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    check_working_directory()
    args = resolve_arguments()
    main(args)