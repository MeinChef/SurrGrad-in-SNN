from imports import os
from imports import warnings
from imports import surrogate
from imports import Callable
from imports import torch
from imports import re
from imports import functional


# check if the cwd is correct, try to change if Git-Repo exists in cwd.
def check_working_directory() -> bool:
    '''
    Function for checking if the working direcory is in fact the top-level directory of the Git-Repo.
    Tries to descend one folder
    '''

    if "SurrGrad-in-SNN" in os.getcwd()[-16:]:
        return True
    else:
        if "SurrGrad-in-SNN" in os.listdir():
            try:
                os.chdir(os.path.join(os.getcwd(), "SurrGrad-in-SNN"))
            except Exception as e:
                print(e)
                raise LookupError("Could not find the folder SurrGrad-in-SNN in your current working directory. "
                                  "Consider changing the working directory")
            warnings.warn("Changed Working directory. Descended into \"SurrGrad-in-SNN\".")
            return True 
        else:
            warnings.warn("Could not find the folder SurrGrad-in-SNN in your current working directory. "
                          "No guarantees for working code from this point on.\n"
                          "Proceeding...")   
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
    elif name == "mse_membrane":
        return functional.loss.mse_membrane_loss(
            time_var_targets = False,
            on_target = config["on_target"],
            off_target = config["off_target"]
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