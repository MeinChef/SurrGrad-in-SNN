import misc
import data

def main() -> None:
    config_data, config_model = data.load_config()
    data = data.data_prep(config_data)


if __name__ == "__main__":
    misc.check_working_directory()
    main()