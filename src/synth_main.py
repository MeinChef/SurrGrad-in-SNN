from data import DataHandler, load_config, request_model_save, save_model
from imports import NOW, argparse, functional, torch
from imports import snntorch as snn
from misc import check_working_directory
from synth_data import DataGenerator
from synth_model import SynthModel
from tracker import MetricsTracker


def main(
    args: argparse.Namespace
):
    cfg_data, cfg_model = load_config(args.cfg_path)

    # initialise datagenerator
    print("Initialising Classes...")
    datagen = DataGenerator(
        time_steps = cfg_data["time_steps"]["val"],
        shuffle = cfg_data["shuffle_spikes"],
        neurons = cfg_data["neurons"]["val"],
        min_isi = cfg_data["min_isi"],
        max_isi = cfg_data["max_isi"],
        min_rate = cfg_data["min_rate"],
        max_rate = cfg_data["max_rate"],
        # only_even = cfg_data["only_even"]
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
            time_steps = cfg_data["time_steps"]["val"],
            datapath = "img/"
        )

    # and some simple loss/acc tracker
    tracker = MetricsTracker()

    print("Done!")

    # generate dataset
    print(f"Generating Data. ({cfg_data['no_samples']} total)...")
    train, test = datagen.generate_dataset(
        no_samples = cfg_data["no_samples"],
        batch_size = cfg_data["batch_size"],
        train_split = cfg_data["train_split"],
        shuffle = cfg_data["shuffle"],
        prefetch = cfg_data["prefetch"],
    )
    datagen.visualise_classes()

    curated = datagen.generate_dataset(
        no_samples = cfg_model["samples"],
        batch_size = cfg_model["samples"],
        train_split = 0,
        shuffle = False,
        prefetch = cfg_data["prefetch"],
    )[0]
    print("Done!")

    print("Training...")

    best_loss = torch.inf
    cur_loss = torch.inf

    for e in range(cfg_model["epochs"]):
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

        if model._best_loss < best_loss:
            # update saved model
            print("Model performance improved!")

            if args.save_model:
                save_model(
                    model,
                    f"{NOW}.pt"
                )
            best_loss = model._best_loss
        cur_loss = torch.tensor(loss).mean()

        if args.record_hidden:
            # record data
            handler.enable()                    # pyright: ignore[reportPossiblyUnboundVariable]

            if args.augment:
                rec_loss, rec_acc = model.augmented_eval(
                    data = curated,
                    augment = args.augment,
                    jitter = cfg_model["jitter"],
                    only_nth_layer = cfg_model["augmented_layer"]
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
                name_ext = f"{NOW}-ep{e}"
            )
            handler.save_metrics(               # pyright: ignore[reportPossiblyUnboundVariable]
                rec_loss, rec_acc,
                name_ext = f"{NOW}-ep{e}"
            )

            handler.disable()                   # pyright: ignore[reportPossiblyUnboundVariable]
            handler.clear_recorded_data()       # pyright: ignore[reportPossiblyUnboundVariable]


    if model._best_loss != cur_loss:
        request_model_save(
            model = model,
            identifier = f"{NOW}-ep{cfg_model["epochs"]}.pt"
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
        default = "config.yml",
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
    args = parser.parse_args()
    if args.augment and not args.record_hidden:
        parser.error("-a/--augment can only be used together with -r/--record-hidden")
    return args

if __name__ == "__main__":
    check_working_directory()
    args = resolve_arguments()
    main(args)
