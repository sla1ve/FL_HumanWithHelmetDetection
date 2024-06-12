from collections import OrderedDict
from omegaconf import DictConfig
import torch
import numpy as np
from model import Net, test


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):

    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Ensure the parameters are in the correct format
        if isinstance(parameters[0], np.ndarray):
            parameters = [torch.tensor(param) for param in parameters]

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: v for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn
