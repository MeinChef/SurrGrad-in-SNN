from imports import os
from imports import shutil
from imports import warnings
from imports import surrogate
from imports import Callable
from imports import datetime
from imports import numpy as np
from imports import torch
from imports import tqdm
from imports import re
from imports import functional

# check if the cwd is correct, try to change if Git-Repo exists in cwd.
def check_working_directory() -> bool:
    '''
    Function for checking if the working direcory is in fact the top-level directory of the Git-Repo.
    Tries to descend one folder
    '''

    if "SurrGrad-in-SNN" in os.getcwd():
        return True
    else:
        if "SurrGrad-in-SNN" in os.listdir():
            try:
                os.chdir(os.path.join(os.getcwd(), "SurrGrad-in-SNN"))
            except Exception as e:
                print(e)
                raise LookupError("Could not find the folder SurrGrad-in-SNN in your current working directory. \
                                  Consider changing the working directory")
            warnings.warn("Changed Working directory. Descended into \"SurrGrad-in-SNN\".")
            return True 
        else:
            warnings.warn("Could not find the folder SurrGrad-in-SNN in your current working directory. \
                          No guarantees for working code from this point on.\n\
                          Proceeding...")   
            return False

def resolve_gradient(config: dict) -> Callable:
    '''
    Function for resolving the gradient, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.
    '''

    name = config["type"].lower()

    if name == "atan":
        return surrogate.atan(config["alpha"])
    elif name == "fast_sigmoid":
        return surrogate.fast_sigmoid(config["slope"])
    elif name == "heavside":
        return surrogate.heaviside()
    elif name == "sigmoid":
        return surrogate.sigmoid(config["slope"])
    elif name == "spike_rate_escape":
        return surrogate.spike_rate_escape(config["beta"], config["slope"])
    elif name == "straight_through":
        return surrogate.straight_through_estimator()
    elif name == "triangular":
        return surrogate.triangular(config["threshold"])
    elif name == "super_spike_21":
        raise NotImplementedError()
        # return super_spike_21(config["surrogate_arg"][0],config["surrogate_arg"][1],config["surrogate_arg"][2])
    else:
        raise NameError("The surrogate function specified in config is unresolveable. Check source code and typos")

def resolve_loss(config: dict) -> Callable:
    '''
    Function for resolving the loss function, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.
    '''

    name = config["type"].lower()

    if name == "ce_temporal":
        return functional.loss.ce_temporal_loss(
            inverse = config["inverse"],
        )
    elif name == "ce_rate":
        return functional.loss.ce_rate_loss()
    elif name == "mse_temporal":
        return functional.loss.mse_temporal_loss(
            tolerance = config["tolerance"]
            )
    elif name == "mse_count":
        return functional.loss.mse_count_loss(
            correct_rate = config["correct_rate"],
            incorrect_rate = config["incorrect_rate"]
        )
    else:
        raise NameError("The loss function specified in config is unresolveable. Check source code and typos")

def resolve_acc(config: dict) -> Callable:
    '''
    Function for resolving the accuracy function, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.
    '''

    name = config["type"].lower()

    if name == "rate":
        return functional.acc.accuracy_rate
    elif name == "temporal":
        return functional.acc.accuracy_temporal
    else:
        raise NameError("The accuracy function specified in config is unresolveable. Check source code and typos")

def resolve_optim(config: dict, params) -> Callable:
    '''
    Function for resolving the optimizer, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.

    :param config: config dictionary
    :type config: dict
    
    :param params: parameters of the model
    :type params: ParamT

    :return: optimizer
    '''

    name = config["type"].lower()

    if name == "adam":
        return torch.optim.Adam(
            params = params,
            lr = config["learning_rate"],
            betas = (config["betas"][0], config["betas"][1])
        )
    else:
        raise NameError("The optimizer specified in config is unresolveable. Check source code and typos")

def resolve_encoding_map(config: dict) -> Callable:
    
    name = config["target"].lower()

    if name == "rate":
        return lambda x: x
    elif name == "latency":
        return lambda x: x
    elif name == "latency_timing":
        def transf(x):
            '''
            Function for generating on_target spikes, given a class label.
            '''
            torch.manual_seed(x)
            return torch.randint(
                # quickest timing can be at 0
                low = 0,
                # slowest at 300 (see get_shortest_observation function)
                high = 300, 
                # arbitrarily set size (thought 4 spikes seem nice)
                size = (4,),
                dtype = torch.float32
            )
        return transf
    else:
        raise NameError("The target encoding specified in config is unresolveable. Check source code and typos")


def spk_rec_to_file(
    data: list = None,
    identifier: str = None,
    path: str = "data/rec/"
) -> None:
    '''
    Function for saving the spike recording of the hidden layers into a file on disk.
    
    :param data: list of the structure [[recording_of_layer1], [recording_of_layer2], [recording_of_layer3]]
    :type data: list, required
    
    :param identifier: Useful for saving the data with a custom filename
    :type identifier: str or list[str], optional

    :param path: Path where to save the data. Default data/rec/
    :type path: str, optional
    '''

    # make path os-independent
    path = make_path(path)

    # TODO: change path resolving with str.split() and os.path.join()
    if isinstance(identifier, list):
        assert len(identifier) == 3
    elif isinstance(identifier, str):
        identifier = [identifier + '-layer1.npz', identifier + '-layer2.npz', identifier + '-layer3.npz']
    elif identifier is None:
        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        identifier = [now + '-layer1.npz', now + '-layer2.npz', now + '-layer3.npz']


    for i, layer in enumerate(data):
        for j, rec in enumerate(layer):

            # if recording is in GPU memory
            if rec.get_device() >= 0:
                rec = rec.cpu().numpy()
                layer[j] = rec.astype(np.int8) # we shouldn't loose any expressiveness, since spikes are usually 0s or 1s
            # or it's on cpu
            elif rec.get_device() == -1:
                layer[j] = rec.numpy().astype(np.int8)
            
        np.savez_compressed(os.path.join(path, identifier[i]), *layer)

def load_spk_rec(
        config: dict,
        identifier: str = "test-ep0"
) -> list[np.ndarray]:
    '''
    Function for loading the spike recordings of the hidden layers from a file on disk.
    
    :param config: config dictionary
    :type config: dict
    
    :param identifier: Useful for loading the data with a custom filename
    :type identifier: str, optional

    :return: list of the structure [[recording_of_layer1], [recording_of_layer2], [recording_of_layer3]]
    :rtype: list[np.ndarray]
    '''

    paths = [make_path(config["data_path"] + "/rec/" + identifier + f"-layer{layer}.npz") for layer in range(1, 4)]

    recordings = []
    for path in paths:
        try:
            data = np.load(path, allow_pickle = True)
            recordings.append(data)
        except FileNotFoundError:
            print(f"File {path} not found. Skipping...")
            continue
    return recordings


def stats_to_file(
        config: dict, 
        loss: list, 
        acc: list = None, 
        spk_rec: list[list,list,list] = None,
        identifier: str = None
) -> None:
    '''
    Saves the output from the model to a file, human-readable.

    :param config: config dictionary
    :type config: dict
    
    :param loss: list of loss values
    :type loss: list
    
    :param acc: list of accuracy values
    :type acc: list

    :param spk_rec: spk_rec of all layers 
    :type spk_rec: list, optional
    '''

    if len(loss) != 0:
        try:
            np.savetxt(
                make_path(config["data_path"] + "/loss.txt"),
                loss,
                fmt="%.8f"
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            breakpoint()
    if acc:
        try:
            if len(acc) != 0:
                np.savetxt(
                    make_path(config["data_path"] + "/acc.txt"),
                    acc,
                    fmt="%.8f"
                )
        except Exception as e:
            print(f"An error occurred: {e}")
            breakpoint()

    if spk_rec:
        try:
            spk_rec_to_file(
                data = spk_rec,
                identifier = identifier,
                path = config["data_path"] + "/rec/"
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            breakpoint()





def get_shortest_observation(
        data: torch.utils.data.DataLoader
) -> int:
    '''
    Function for getting the shortest observation in the dataset.
    '''
    shortest = 2**32
    for x, y in tqdm.tqdm(data):
        if x.shape[0] < shortest:
            shortest = x.shape[0]
    
    return shortest # 307 in the whole dataset

def get_longest_observation(
        data: torch.utils.data.DataLoader
) -> int:
    '''
    Function for getting the longest observation in the dataset.
    '''
    longest = 0
    for x, y in tqdm.tqdm(data):
        if x.shape[0] > longest:
            longest = x.shape[0]
    
    return longest # 314 in the whole dataset

def get_sample_distribution(
        data: torch.utils.data.DataLoader,
        num_classes: int = 10
) -> torch.Tensor:
    '''
    Function for getting the sample distribution in the dataset.
    '''
    amount = torch.zeros(num_classes, dtype = torch.int32)
    for x, y in tqdm.tqdm(data):
        amount += torch.bincount(y, minlength = 10)
    return amount

def get_sample_distribution_from_tonic(
        data,
        num_classes: int = 10
) -> torch.Tensor:
    '''
    Function for getting the sample distribution in the dataset.
    '''
    amount = torch.zeros(num_classes, dtype = torch.int32)
    for x, y in tqdm.tqdm(data):
        amount[y] += 1

    return amount


def make_path(path: str) -> os.PathLike:
    '''
    Function for creating cross-os-compatible paths from strings.
    
    :param path: String to be converted to os.PathLike
    :type path: str or list of str, required
    :return: Path in the correct format for the current OS
    :rtype: os.PathLike object
    '''

    if isinstance(path, str):
        path = [path]

    path_lst = []
    for part in path:
        part = re.split(r'[/\\]', part)
        path_lst.extend(part)
    path = os.path.join(*path_lst)
    return path

def cleanup(config: dict) -> None:
    '''
    Function for cleaning up DiskCachedDataset files. 
    Since they caused weird happenings to the targets, it is better to delete them.

    :param config: config dictionary
    :type config: dict
    '''

    to_clean = make_path(config["cache_path"])
 
    print(f"Cleaning up {to_clean}...")

    if os.path.exists(to_clean):
        try:
            shutil.rmtree(to_clean)
        except OSError as e:
            print(f"Error: {e.strerror}. Could not delete {to_clean}.")
    else:
        print(f"Path {to_clean} does not exist. Nothing to clean up.")

    print("Done.")
    
    return