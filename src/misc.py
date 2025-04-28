from imports import os
from imports import warnings
from imports import typing
from imports import surrogate
from imports import Callable
from imports import surrogate
from imports import numpy as np
from imports import datetime
from imports import plt
# from grad import super_spike_21
from imports import torch
from imports import tqdm

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
            except:
                raise LookupError("Could not find the folder SurrGrad-in-SNN in your current working directory. \
                                  Consider changing the working directory")
            warnings.warn("Changed Working directory. Descended into \"SurrGrad-in-SNN\".")
            return True 
        else:
            warnings.warn("Could not find the folder SurrGrad-in-SNN in your current working directory. \
                          No guarantees for working code from this point on.\
                          Proceeding...")   
            return False

def resolve_gradient(config: dict) -> Callable:
    '''
    Function for resolving the gradient, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.
    '''

    name = config["surrogate"].lower()

    if name == "atan":
        return surrogate.atan(config["surrogate_arg"][0])
    elif name == "fast_sigmoid":
        return surrogate.fast_sigmoid(config["surrogate_arg"][0])
    elif name == "heavside":
        return surrogate.heaviside()
    elif name == "sigmoid":
        return surrogate.sigmoid(config["surrogate_arg"][0])
    elif name == "spike_rate_escape":
        return surrogate.spike_rate_escape(config["surrogate_arg"][0], config["surrogate_arg"][1])
    elif name == "straight_through":
        return surrogate.straight_through_estimator()
    elif name == "triangular":
        return surrogate.triangular(config["surrogate_arg"][0])
    elif name == "super_spike_21":
        raise NotImplementedError()
        # return super_spike_21(config["surrogate_arg"][0],config["surrogate_arg"][1],config["surrogate_arg"][2])
    else:
        raise NameError("The surrogate function specified in config is unresolveable. Check source code and typos")

def resolve_encoding(config: dict) -> Callable:
    
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
    
    ### Args:
    data: list - list of the structure [[recording_of_layer1], [recording_of_layer2], [recording_of_layer3]]
    identifier: str or list[str] - Optional. Useful for saving the data with a custom filename
    path: str - Optional. Path where to save the data. Default data/rec/
    '''


    # TODO: change path resolving with str.split() and os.path.join()
    now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if isinstance(identifier, list):
        assert len(identifier) == 3
    elif isinstance(identifier, str):
        identifier = [identifier + '-layer1.npz', identifier + '-layer2.npz', identifier + '-layer3.npz']
    elif identifier == None:
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
            
        np.savez_compressed(path + identifier[i], *layer)   
            

# def get_test_array_sffn():
#     layer1 = [torch.full((20,), 0, dtype = torch.float32), torch.full((15,), 1, dtype = torch.float32), torch.full((10,), 2, dtype = torch.float32)]
#     layer2 = [torch.full((20,), 1, dtype = torch.float32), torch.full((15,), 2, dtype = torch.float32), torch.full((10,), 0, dtype = torch.float32)]
#     layer3 = [torch.full((20,), 2, dtype = torch.float32), torch.full((15,), 0, dtype = torch.float32), torch.full((10,), 1, dtype = torch.float32)]
#     return [layer1, layer2, layer3]

# liste = get_test_array_sffn()
# spk_rec_to_file(liste)

def stats_to_file(config: dict, loss: list, acc: list = None, spk_rec: list[list,list,list] = None) -> None:
    """
    Saves the output from the model to a file, human-readable.

    ### Arguments
    config: dict - config dictionary
    loss: list - list of loss values
    acc: list - list of accuracy values
    spk_rec: list - optional. spk_rec of all layers 
    """

    if len(loss) != 0:
        try:
            np.savetxt(
                config["data_path"] + "/loss.txt",
                loss,
                fmt="%.8f"
            )
        except:
            breakpoint()
    if acc:
        try:
            if len(acc) != 0:
                np.savetxt(
                    config["data_path"] + "/acc.txt",
                    acc,
                    fmt="%.8f"
                )
        except:
            breakpoint()

    

def plot_loss_acc(config:dict) -> None:
    breakpoint()
    # load values from files
    loss = np.loadtxt(
        config["data_path"] + "/loss.txt"
    )

    acc = np.loadtxt(
        config["data_path"] + "/acc.txt"
    )

    assert len(loss) == len(acc), print(f"Loss ain't acc, off by {len(loss)-len(acc)}")
    
    epochs = np.arange(1, len(loss) + 1)

    fig, ax1 = plt.subplots()

    # Plot loss on the left y-axis
    ax1.set_xlabel('Batches')
    ax1.set_ylabel('Loss', color='orange')
    ax1.plot(epochs, loss, color='orange', label='Loss')
    ax1.tick_params(axis='y', labelcolor='orange')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='blue')
    ax2.plot(epochs, acc, color='blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title('Loss and Accuracy during Training')
    fig.tight_layout()
    plt.show()

def get_shortest_observation(
        data: torch.utils.data.DataLoader
) -> int:
    """
    Function for getting the shortest observation in the dataset.
    """
    shortest = 2**32
    for x, y in tqdm.tqdm(data):
        if x.shape[0] < shortest:
            shortest = x.shape[0]
    
    return shortest # 307 in the whole dataset