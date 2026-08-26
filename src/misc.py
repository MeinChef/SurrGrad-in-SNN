from imports import NOW, ROOT, Callable, Path, functional, os, surrogate, torch, warnings
from loss import FirstSpikeLoss, MeanCELoss, SpikemaxLoss
from surrogate import stable_sigmoid


# check if the cwd is correct, try to change if Git-Repo exists in cwd.
def check_working_directory() -> bool:
    """
    Function for checking if the working direcory is in fact the top-level directory of the Git-Repo.
    Tries to descend one folder
    """

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
    """
    Function for resolving the gradient, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.
    """

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
    elif name == "stable_sigmoid":
        return stable_sigmoid(config["slope"])
    else:
        raise NameError("The surrogate function specified in config is unresolveable. Check source code and typos")

def resolve_loss(config: dict) -> Callable:
    """
    Function for resolving the loss function, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.
    """

    name = config["type"].lower()

    if name == "ce_temporal":
        return functional.loss.ce_temporal_loss(
            inverse = config["inverse"],
            reduction = config["reduction"] 
        )
    elif name == "ce_max_membrane":
        return functional.loss.ce_max_membrane_loss(
            reduction = config["reduction"] 
        )
    elif name == "ce_rate":
        return functional.loss.ce_rate_loss(
            population_code = config["population"]["is_pop"],
            num_classes = config["population"]["num_classes"],
            reduction = config["reduction"]
        )
    elif name == "ce_count":
        return functional.loss.ce_count_loss(
            population_code = config["population"]["is_pop"],
            num_classes = config["population"]["num_classes"],
            reduction = config["reduction"]
        )
    elif name == "mse_temporal":
        return functional.loss.mse_temporal_loss(
            on_target = config["on_target"],
            off_target = config["off_target"],
            tolerance = config["tolerance"],
            reduction = config["reduction"]
        )
    elif name == "mse_rate":
        return functional.loss.mse_count_loss(
            population_code = config["population"]["is_pop"],
            num_classes = config["population"]["num_classes"],
            correct_rate = config["correct_rate"],
            incorrect_rate = config["incorrect_rate"],
            reduction = config["reduction"]
        )
    elif name == "mse_membrane":
        return functional.loss.mse_membrane_loss(
            on_target = config["on_target"],
            off_target = config["off_target"],
            reduction = config["reduction"]
        )
    elif name == "mse":
        return torch.nn.MSELoss(
            reduction = config["reduction"]
        )
    elif name == "ttfs":
        return FirstSpikeLoss(
            alpha = config["alpha"],
        )
    elif name == "mean_ce":
        return MeanCELoss(
            intermediate_reduction = config["intermediate_reduction"],
            reduction = config["reduction"]
        )
    elif name == "spikemax":
        return SpikemaxLoss()
    else:
        raise NameError("The loss function specified in config is unresolveable. Check source code and typos")

def resolve_acc(config: dict) -> Callable:
    """
    Function for resolving the accuracy function, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.
    """

    name = config["type"].lower()

    if name == "rate":
        return functional.acc.accuracy_rate
    elif name == "temporal":
        return functional.acc.accuracy_temporal
    else:
        raise NameError("The accuracy function specified in config is unresolveable. Check source code and typos")

def resolve_optim(config: dict, params) -> torch.optim.Optimizer:
    """
    Function for resolving the optimizer, given as a string in config.yml, and returning a function, with proper fromatting
    for further use.

    :param config: config dictionary
    :type config: dict

    :param params: parameters of the model
    :type params: ParamT

    :return: optimizer
    """

    name = config["type"].lower()

    if name == "adam":
        return torch.optim.Adam(
            params = params,
            lr = config["learning_rate"],
            betas = (config["betas"][0], config["betas"][1]),
            weight_decay = config["weight_decay"]
        )
    else:
        raise NameError("The optimizer specified in config is unresolveable. Check source code and typos")

def create_datapath(
    data_path: str | Path | None,
    identifier: str | None
) -> str:
    if data_path is None:
        path = os.path.join(
            ROOT,
            "data"
        )
    elif not os.path.isabs(data_path):
        path = os.path.join(
            ROOT,
            data_path
        )
    else:
        path = data_path

    if identifier is None:
        identifier = NOW

    new_path = os.path.join(
        path,
        identifier
    )

    # create the top-data directory
    os.makedirs(new_path)
    # and directories for model, img, and bin
    for dir in ["model", "img", "bin"]:
        os.makedirs(os.path.join(
            new_path, dir
        ))

    return new_path