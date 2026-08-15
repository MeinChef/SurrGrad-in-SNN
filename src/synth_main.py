from data import DataHandler, load_config, request_model_save, save_model
from imports import NOW, argparse, functional, sys, torch
from imports import snntorch as snn
from misc import check_working_directory, create_datapath
from synth_data import DataGenerator
from synth_model import SynthModel
from tracker import MetricsTracker


def main(
    args: argparse.Namespace
):
    cfg_data, cfg_model = load_config(args.cfg_path)
    data_path = create_datapath(
        data_path = args.save_path,
        identifier = args.identifier
    )

    # initialise datagenerator
    print("Initialising Classes...")
    datagen = DataGenerator(
        time_steps = cfg_data.get("time_steps", {"val": 1000})["val"],
        shuffle    = cfg_data.get("shuffle_spikes", 0.0),
        neurons    = cfg_data.get("neurons", {"val": 10})["val"],
        min_isi    = cfg_data.get("min_isi", 1),
        max_isi    = cfg_data.get("max_isi", 50),
        min_rate   = cfg_data.get("min_rate", 2),
        max_rate   = cfg_data.get("max_rate", 10),
        # only_even = cfg_data.get("only_even", True)
        # precision = np.float32
    )

    # initialise model
    model = SynthModel(
        config = cfg_model,
    )

    # hot new shit
    if args.record_hidden:
        # initialise the recorder
        recorder = functional.probe.OutputMonitor(
            net = model,
            instance = snn.Leaky
        )
        recorder.disable()

        # and wrap it for more functions
        handler = DataHandler(
            recorder = recorder,
            time_steps = cfg_data.get("time_steps", {"val": 1000})["val"],
            data_path = data_path
        )

    # and some simple loss/acc tracker
    tracker = MetricsTracker(data_path)

    print("Done!")

    # generate dataset
    print(f"Generating Data. ({cfg_data.get('no_samples', 1e5)} total)...")
    train, test = datagen.generate_dataset(
        no_samples  = cfg_data.get("no_samples", 1e5),
        batch_size  = cfg_data.get("batch_size", 512),
        train_split = cfg_data.get("train_split", 0.7),
        shuffle     = cfg_data.get("shuffle", True),
        prefetch    = cfg_data.get("prefetch", 16),
    )
    datagen.visualise_classes()

    curated = datagen.generate_dataset(
        no_samples  = cfg_model.get("samples", 20),
        batch_size  = cfg_model.get("samples", 20),
        train_split = 0,
        shuffle     = False,
        prefetch    = cfg_data.get("prefetch", 1),
    )[0]
    print("Done!")

    print("Training...")

    best_loss = torch.inf
    cur_loss = torch.inf
    cur_patience = 0

    for e in range(cfg_model.get("epochs", 100)):
        print(f"Epoch: {e}")

        # training
        loss, acc = model.fit(train)
        tracker.update_train(loss, acc)

        # validation
        loss, acc = model.evaluate(
            data = test
        )
        tracker.update_val(loss, acc)
        tracker.announce()

        cur_loss = torch.tensor(loss).mean()
        print(
            f"DEBUG: cur_loss={cur_loss!r}, "
            f"best_loss={best_loss!r}, "
            f"cur_loss < best_loss = {cur_loss < best_loss}"
        )
        # save model if it got better
        if cur_loss < best_loss:
            # update saved model
            print("Model performance improved!")

            if args.save_model:
                save_model(
                    model = model,
                    data_path = data_path,
                    config_path = args.cfg_path,
                    identifier = args.identifier
                )
            best_loss = cur_loss
            cur_patience = 0
        else:
            cur_patience += 1
            print(
                f"No improvement. Patience: "
                f"{cur_patience}/{cfg_model.get("patience", 5)}"
            )

        # and update the current loss

        if args.record_hidden:
            # record data
            handler.enable()                    # pyright: ignore[reportPossiblyUnboundVariable]

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

            # and visualise
            handler.measure_tendencies(         # pyright: ignore[reportPossiblyUnboundVariable]
                curated
            )
            handler.visualise_tendencies(       # pyright: ignore[reportPossiblyUnboundVariable]
                name_ext = f"{args.identifier}-ep{e}"
            )
            handler.save_metrics(               # pyright: ignore[reportPossiblyUnboundVariable]
                rec_loss, rec_acc,
                name_ext = f"{args.identifier}-ep{e}"
            )

            handler.disable()                   # pyright: ignore[reportPossiblyUnboundVariable]
            handler.clear_recorded_data()       # pyright: ignore[reportPossiblyUnboundVariable]

        if args.save_model:
            tracker.save(force = True)

        if cur_patience >= cfg_model.get("patience", 5):
            break

    if model._best_loss != cur_loss and sys.stdin.isatty():
        request_model_save(
            model = model,
            data_path = data_path,
            config_path = args.cfg_path,
            identifier = f"{args.identifier}-ep{cfg_model.get("epochs", 100)}.pt"
        )

    tracker.plot(train = True)
    tracker.plot(train = False)
    tracker.save()

    print("Success!")
    return True


def resolve_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        "-c",
        type = str,
        required = False,
        default = None,
        help = "Path to the config directory. Defaults to ./config.yml"
    )
    parser.add_argument(
        "--record-hidden",
        "-r",
        action = "store_true",
        required = False,
        help = "Flag whether to record the hidden layers and save them"
    )
    parser.add_argument(
        "--save-model",
        "-s",
        action = "store_true",
        required = False,
        help = "Flag whether to save the model checkpoints"
    )
    parser.add_argument(
        "--augment",
        "-a",
        default = None,
        choices = ["jitter", "shuffle"],
        required = False,
        help = "Defines how to augment the forward pass of the model when recording. Options: 'jitter', 'shuffle'. Default: None"
    )
    parser.add_argument(
        "--save-path",
        "-p",
        type = str,
        required = False,
        default = None,
        help = "Path for the visualisations / models / outputs to be saved."
    )
    parser.add_argument(
        "--identifier",
        "-i",
        type = str,
        required = False,
        default = NOW,
        help = "Default Filename for visualisations / models / outputs."
    )
    args = parser.parse_args()
    if args.augment and not args.record_hidden:
        parser.error("-a/--augment can only be used together with -r/--record-hidden")
    return args

if __name__ == "__main__":
    check_working_directory()
    args = resolve_arguments()
    main(args)
