from imports import yaml
from imports import torch
from imports import tonic
from imports import torchvision
from misc import make_path

###############################
### OR SOME DATALOADER LEAK ###
###############################

# load the config.yml
def load_config(path: str = "config.yml") -> tuple[dict, dict]:
    with open(path, "r") as file:
        configs = yaml.safe_load(file)

    return configs["config_data"], configs["config_model"]

def data_prep(config: dict) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config["DEBUG"]:
        from imports import pickle


    sensor = tonic.datasets.NMNIST.sensor_size
    num_classes = len(tonic.datasets.NMNIST.classes)

    # accumulate events to discrete "frames"
    transform_train = tonic.transforms.Compose(
        [
            tonic.transforms.Denoise(
                filter_time = config["filter_time"]
            ),

            tonic.transforms.ToFrame(
                sensor_size = sensor,
                time_window = config["time_window"]
            ),
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10,10])
        ]
    )
    
    transform_test = tonic.transforms.Compose(
        [
            tonic.transforms.Denoise(
                filter_time = config["filter_time"]
            ),

            tonic.transforms.ToFrame(
                sensor_size = sensor,
                time_window = config["time_window"]
            )
            # without random rotation for testset
        ]
    )
    

    # apply the transform to the datasets
    trainset = tonic.datasets.NMNIST(
        save_to = make_path(config["data_path"]),
        transform = transform_train,
        train = True    
    )
    testset = tonic.datasets.NMNIST(
        save_to = make_path(config["data_path"]),
        transform = transform_test,
        train = False
    )


    # this thing eats RAM as snack
    # cached_trainset = tonic.MemoryCachedDataset(trainset)
    # cached_testset  = tonic.MemoryCachedDataset(testset)
    
    cached_trainset = tonic.DiskCachedDataset(
        dataset = trainset, 
        cache_path = make_path(config["cache_path"]),
    )
    
    cached_testset = tonic.DiskCachedDataset(
        dataset = testset,
        cache_path = make_path(config["cache_path"])
    )


    if config["DEBUG"]:
        print("Sensor:", sensor)
        print("Rough Size of Dataset in Memory:", len(pickle.dumps(cached_trainset)) + len(pickle.dumps(cached_testset)))
        
    torch.manual_seed(config["seed"])
    # prepare them for training, 
    trainloader = torch.utils.data.DataLoader(
        dataset = cached_trainset, 
        batch_size = config["batch_size"], 
        collate_fn = tonic.collation.PadTensors(batch_first = False), 
        shuffle = True,
        num_workers = config["worker"],
        prefetch_factor = config["prefetch"],
        pin_memory = True
    )
    
    testloader = torch.utils.data.DataLoader(
        dataset = cached_testset, 
        batch_size = config["batch_size"], 
        collate_fn = tonic.collation.PadTensors(batch_first = False),
        num_workers = config["worker"],
        prefetch_factor = config["prefetch"],
        pin_memory = True
    )

    return trainloader, testloader, num_classes