from imports import os
from imports import warnings
from imports import typing
from imports import surrogate
from imports import Callable
from imports import surrogate
from imports import numpy as np
from imports import datetime
from grad import super_spike_19

from imports import torch

# check if the cwd is correct, try to change if Git-Repo exists in cwd.
def check_working_directory() -> bool:
    '''
    Function for checking if the working direcory is in fact the top-level directory of the 
    '''

    if "SurrGrad-in-SNN" in os.getcwd():
        return True
    else:
        if "SurrGrad-in-SNN" in os.listdir():
            try:
                os.chdir(os.getcwd()+"/SurrGrad-in-SNN")
            except:
                raise LookupError("Could not find the folder SurrGrad-in-SNN in your current working directory. \
                                  Consider changing the working directory")
        warnings.warn("Changed Working directory. Descended into \"SurrGrad-in-SNN\".")
        return True    

def resolve_gradient(config: dict) -> Callable:
    '''
    Function for resolving the gradient, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.
    '''

    name = config["surrogate"]
    name = name.lower()

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
    elif name == "super_spike_19":
        return surrogate.custom_surrogate(super_spike_19)
    else:
        raise NameError("The surrogate function specified in config is unresolveable. Check source code and typos")

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