import misc
import data
from model import Model

def main() -> None:
    config_data, config_model = data.load_config()
    train, test = data.data_prep(config_data)
    model = Model(config = config_model)
    model.train_loop(data = train)
    # model.test_loop(data = test)

if __name__ == "__main__":
    misc.check_working_directory()
    main()