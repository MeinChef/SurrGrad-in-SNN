from imports import yaml
from imports import torch
from imports import tonic
from imports import torchvision


# load the config.yml
def load_config(path: str = "config.yml") -> tuple[dict, dict]:
    with open(path, "r") as file:
        configs = yaml.safe_load(file)

    return configs["config_data"], configs["config_model"]

def data_prep(config: dict) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config["DEBUG"]:
        from imports import pickle


    sensor = tonic.datasets.NMNIST.sensor_size
    
    # accumulate events to discrete "frames"
    transform_frame = tonic.transforms.Compose(
        [
            tonic.transforms.Denoise(
                filter_time = config["filter_time"]
            ),

            tonic.transforms.ToFrame(
                sensor_size = sensor,
                time_window = config["time_window"]
            )
        ]
    )

    # apply the transform to the datasets
    trainset = tonic.datasets.NMNIST(
        save_to = config["data_path"],
        transform = transform_frame,
        train = True    
    )
    testset = tonic.datasets.NMNIST(
        save_to = config["data_path"],
        transform = transform_frame,
        train = False
    )

    # augment the trainset with rotations of the "frames"
    transform = tonic.transforms.Compose(
        [
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10,10])
        ]
    )

    # cache the datasets into Memory (requires 2.3GB of RAM)
    cached_trainset = tonic.MemoryCachedDataset(trainset, transform = transform)
    cached_testset  = tonic.MemoryCachedDataset(testset)


    if config["DEBUG"]:
        print("Sensor:", sensor)
        print("Rough Size of Dataset in Memory:", len(pickle.dumps(cached_trainset)) + len(pickle.dumps(cached_testset)))
        
    # prepare them for training, 
    trainloader = torch.utils.data.DataLoader(
        cached_trainset, 
        batch_size = config["batch_size"], 
        collate_fn = tonic.collation.PadTensors(batch_first = False), 
        shuffle = True,
        num_workers = config["worker"],
        prefetch_factor = config["prefetch"]
    )
    testloader = torch.utils.data.DataLoader(
        cached_testset, 
        batch_size = config["batch_size"], 
        collate_fn = tonic.collation.PadTensors(batch_first = False),
        num_workers = config["worker"],
        prefetch_factor = config["prefetch"]
    )

    return trainloader, testloader