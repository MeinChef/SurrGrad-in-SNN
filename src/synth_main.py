from synth_data import DataGenerator
from synth_model import SynthModel
from data import load_config, save_model, load_model, DataHandler
from misc import check_working_directory
from visualisation import plot_epoch_losses
from imports import NOW
from imports import argparse
from imports import functional
from imports import torch
from imports import snntorch as snn

def main(
    args: argparse.Namespace
):
    cfg_data, cfg_model = load_config(args.cfg_path)

    # initialise datagenerator
    print("Initialising Classes...")
    datagen = DataGenerator(
        time_steps = cfg_data["time_steps"]["val"],
        shuffle = cfg_data["shuffle"],
        neurons = cfg_data["neurons"]["val"],
        min_isi = cfg_data["min_isi"],
        max_isi = cfg_data["max_isi"],
        min_rate = cfg_data["min_rate"],
        max_rate = cfg_data["max_rate"],
        only_even = cfg_data["only_even"]
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
            datapath = "data/"
        )

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

    curated = datagen.generate_dataset(
        no_samples = cfg_model["samples"],
        batch_size = cfg_model["samples"],
        train_split = 0,
        shuffle = False,
        prefetch = cfg_data["prefetch"],
    )[0]
    print("Done!")

    print("Training...")
    trainlist = []
    evallist = []
    cur_loss = torch.inf

    for e in range(cfg_model["epochs"]):
        print(f"Epoch: {e}")

        loss, acc = model.fit(train)
        trainlist.append(loss)

        loss, acc = model.evaluate(
            data = test
        )
        evallist.append(loss)

        # loss, acc = model.augmented_eval(
        #     data = test,
        #     augment = "jitter",
        #     jitter = 20,
        #     only_nth_layer = 1
        # )
        print(
            f"Loss of {torch.tensor(loss).mean()} "
            f"and Accuracy of {torch.tensor(acc).mean()}%"
        )

        if model._best_loss < cur_loss:
            # update saved model
            print("Model performance improved!")

            if args.save_model:
                save_model(
                    model,
                    f"{NOW}.pt"
                )
            cur_loss = model._best_loss

        if args.record_hidden:
            # record data
            handler.enable()                    # pyright: ignore[reportPossiblyUnboundVariable]
            _, _ = model.evaluate(
                data = curated
            )

            # and visualise
            handler.measure_tendencies(         # pyright: ignore[reportPossiblyUnboundVariable]
                curated
            )
            handler.visualise_tendencies(       # pyright: ignore[reportPossiblyUnboundVariable]
                name_ext = f"{NOW}-ep{e}"
            )

            handler.disable()                   # pyright: ignore[reportPossiblyUnboundVariable]
            handler.clear_recorded_data()       # pyright: ignore[reportPossiblyUnboundVariable]

    plot_epoch_losses(
        trainlist,
        train = True,
        save = True
    )
    plot_epoch_losses(
        evallist,
        train = False,
        save = True
    )

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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    check_working_directory()
    args = resolve_arguments()
    main(args)
