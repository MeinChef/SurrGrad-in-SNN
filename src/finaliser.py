from data import DataHandler, load_config, load_model
from imports import argparse, functional, os
from imports import snntorch as snn
from synth_data import DataGenerator


def main(args: argparse.Namespace):
    print("Initialising Classes...")
    # assume there is a config saved alongside the model
    try:
        cfg_data, cfg_model = load_config(os.path.join(args.data_path, "model", "synthmodel-" + args.identifier + ".yml"))
    except FileNotFoundError:
        cfg_data, cfg_model = load_config()

    # load model
    model = load_model(
        data_path = os.path.join(args.data_path),
        identifier = args.identifier
    )

    # the data generator
    datagen = DataGenerator(
        time_steps = cfg_data.get("time_steps", {"val": 1000})["val"],
        shuffle    = cfg_data.get("shuffle_spikes", 0.0),
        neurons    = cfg_data.get("neurons", {"val": 10})["val"],
        min_isi    = cfg_data.get("min_isi", 1),
        max_isi    = cfg_data.get("max_isi", 50),
        min_rate   = cfg_data.get("min_rate", 2),
        max_rate   = cfg_data.get("max_rate", 10),
    )

    # initialise the recorder
    recorder = functional.probe.OutputMonitor(
        net = model,
        instance = snn.Leaky
    )
    recorder.disable()
    handler = DataHandler(
        recorder = recorder,
        time_steps = cfg_data.get("time_steps", {"val": 1000})["val"],
        data_path = args.data_path
    )
    print("Done!")

    # generate tiny dataset
    total_samples = cfg_model.get("info_samples", 1000) - \
                cfg_model.get("info_samples", 1000) % cfg_model.get("samples", 10)

    print(f"Generating Data ({total_samples} total)...")
    curated = datagen.generate_dataset(
        no_samples  = total_samples,
        batch_size  = cfg_model.get("samples", 10),
        train_split = 0,
        shuffle     = cfg_data.get("shuffle", True),
        prefetch    = cfg_data.get("prefetch", 1),
    )[0]
    print("Done!")

    # record one forward pass
    handler.enable()
    if args.augment:
        rec_loss, rec_acc = model.augmented_eval(
            data = curated,
            augment = args.augment,
            jitter = cfg_model.get("jitter", 30),
            only_nth_layer = cfg_model.get("augmented_layer", 1)
        )
    else:
        rec_loss, rec_acc = model.evaluate(
            data = curated
        )

    # TODO: fit information criterion here

    # and visualise
    handler.measure_tendencies(
        curated
    )
    handler.visualise_tendencies(
        name_ext = args.identifier
    )
    # TODO: visualise information criteria
    print("Success!")
    return True


def resolve_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "identifier",
        type = str,
        help = "A unique identifier for searching a folder in the data-path. Will be also used to find models/visualisations/etc"
    )
    parser.add_argument(
        "--data-path",
        "-p",
        type = str,
        required = False,
        default = None,
        help = "Path to the data directory. Defaults to ./data/<identifier>"
    )
    parser.add_argument(
        "--augment",
        "-a",
        default = None,
        choices = ["jitter", "shuffle"],
        required = False,
        help = "Defines how to augment the forward pass of the model when recording. Options: 'jitter', 'shuffle'. Default: None"
    )
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = os.path.join("data", args.identifier)
    return args

if __name__ == "__main__":
    args = resolve_arguments()
    main(args)
