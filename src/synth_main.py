from synth_data import DataGenerator
from synth_model import SynthModel
from data import load_config, DataHandler
from misc import check_working_directory

def main():
    cfg_data, cfg_model = load_config()

    # initialise datagenerator
    datagen = DataGenerator(
        time_steps = cfg_data["time_steps"],
        jitter = cfg_data["jitter"],
        neurons = cfg_data["neurons"],
        min_isi = cfg_data["min_isi"],
        max_isi = cfg_data["max_isi"],
        min_rate = cfg_data["min_rate"],
        max_rate = cfg_data["max_rate"],
        # precision = np.float32
    )

    # initialise model
    model = SynthModel()

    # generate dataset
    dataset = datagen.generate_dataset(
        no_samples = cfg_data["no_samples"],
        batch_size = cfg_data["batch_size"],
        shuffle = cfg_data["shuffle"],
        prefetch = cfg_data["prefetch"]
    )

def resolve_arguments():
    raise NotImplementedError()

if __name__ == "__main__":
    check_working_directory()
    # resolve_arguments()
    main()