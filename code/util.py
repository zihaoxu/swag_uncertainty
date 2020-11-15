import torch
import torch.nn as nn
import numpy as np


def model_param_to_1D(net: nn.Module):
    """ Takes in a NN and returns its params in 1D,
        as well as the weights and lengths lookup
    """
    net_params = nn.ParameterList(net.parameters())
    params_1d = np.array([])
    shape_lookup = []
    len_lookup = []

    for param in net_params:
        flat_params = param.detach().numpy().flatten()
        params_1d = np.concatenate((params_1d, flat_params))

        # Store weights len and shape
        shape_lookup.append(param.shape)
        len_lookup.append(len(flat_params))

    return params_1d, shape_lookup, len_lookup


def params_1d_to_weights(params_1d: np.ndarray,
                         shape_lookup: list,
                         len_lookup: list) -> torch.nn.modules.container.ParameterList:
    """ Takes in 1d params and the shape_lookup, len_lookup
        to reconstruct the model weights
    """
    model_params = nn.ParameterList([])
    pointer = 0
    for shape, length in zip(shape_lookup, len_lookup):
        curr_params = torch.tensor(params_1d[pointer:pointer+length]).reshape(shape)
        model_params.append(torch.nn.Parameter(curr_params))
    return model_params


def create_NN_with_weights(NN_class, model_params):
    net = NN_class()
    state_dict = net.state_dict()
    for i, k in enumerate(state_dict.keys()):
        state_dict[k] = model_params[i]
    net.load_state_dict(state_dict)
    return net
