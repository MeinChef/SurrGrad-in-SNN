from synth_data import DataGenerator
from synth_model import SynthModel
from data import load_config, DataHandler
from misc import check_working_directory
from imports import argparse

def main(
    args: argparse.Namespace
):
    cfg_data, cfg_model = load_config()

    # initialise datagenerator
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
        record = args.record_hidden
    )

    # initialise Datahandler
    if args.record_hidden:
        handler = DataHandler(
            path = args.data_path
        )

    # generate dataset
    dataset = datagen.generate_dataset(
        no_samples = cfg_data["no_samples"],
        batch_size = cfg_data["batch_size"],
        shuffle = cfg_data["shuffle"],
        prefetch = cfg_data["prefetch"]
    )

    model.forward(next(iter(dataset))[0])
    print("Success!")

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