from imports import os
from imports import warnings
from imports import typing
from imports import surrogate
from imports import Callable
from imports import surrogate
from grad import super_spike_21

# check if the cwd is correct, try to change if Git-Repo exists in cwd.
def check_working_directory() -> bool:
    if "SurrGrad-in-SNN" in os.getcwd():
        return True
    else:
        if "SurrGrad-in-SNN" in os.listdir():
            try:
                os.chdir(os.getcwd()+"/SurrGrad-in-SNN")
            except:
                raise LookupError("Could not find the folder SurrGrad-in-SNN in your current working directory. Consider changing the working directory")
        warnings.warn("Changed Working directory. Descended into \"SurrGrad-in-SNN\".")
        return True    

def resolve_gradient(config: dict) -> Callable:
    name = config["surrogate"]

    if name == "atan":
        return surrogate.atan(config["surrogate_arg"])
    elif name == "fast_sigmoid":
        return surrogate.fast_sigmoid(config["surrogate_arg"])
    elif name == "heavside":
        return surrogate.heaviside()
    elif name == "sigmoid":
        return surrogate.sigmoid(config["surrogate_arg"])
    elif name == "spike_rate_escape":
        return surrogate.spike_rate_escape(config["surrogate_arg"][0], config["surrogate_arg"][1])
    elif name == "straight_through":
        return surrogate.straight_through_estimator()
    elif name == "triangular":
        return surrogate.triangular(config["surrogate_arg"])
    elif name == "super_spike_21":
        return surrogate.custom_surrogate(super_spike_21)
