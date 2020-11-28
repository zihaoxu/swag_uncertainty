import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


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
        pointer = pointer + length
    return model_params


def create_NN_with_weights(NN_class, model_params):
    net = NN_class()
    state_dict = net.state_dict()
    for i, k in enumerate(state_dict.keys()):
        state_dict[k] = model_params[i]
    net.load_state_dict(state_dict)
    return net


def plot_decision_boundary(swag, x, y, ax, poly_degree=1, test_points=None, shaded=True):
    '''
    plot_decision_boundary plots the training data and the decision boundary of the classifier.
    input:
       swag - the trained swag model
       x - a numpy array of size N x 2, each row is a patient, each column is a biomarker
       y - a numpy array of length N, each entry is either 0 (no cancer) or 1 (cancerous)
       models - an array of classification models
       ax - axis to plot on
       poly_degree - the degree of polynomial features used to fit the model
       test_points - test data
       shaded - whether or not the two sides of the decision boundary are shaded
    returns:
       ax - the axis with the scatter plot
    '''
    # Plot data
    ax.scatter(x[y == 1, 0], x[y == 1, 1], alpha=0.2, c='red', label='class 1')
    ax.scatter(x[y == 0, 0], x[y == 0, 1], alpha=0.2, c='blue', label='class 0')

    # Create mesh
    interval = np.arange(-6, 6, 0.1)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    xx = np.concatenate((x1, x2), axis=1)

    # Predict on mesh points
    if(poly_degree > 1):
        polynomial_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        xx = polynomial_features.fit_transform(xx)

    alpha_line = 0.1
    linewidths = 0.1
    i = 0

    for _ in range(100):
        yy = swag.predict(torch.Tensor(xx), [0, 1], S=1, expanded=False)
        yy = yy.reshape((n, n))

        # Plot decision surface
        x1 = x1.reshape(n, n)
        x2 = x2.reshape(n, n)
        if shaded:
            ax.contourf(x1, x2, yy, alpha=0.1 * 1. / (i + 1)**2, cmap='bwr')
        ax.contour(x1, x2, yy, colors='black', linewidths=linewidths, alpha=alpha_line)

        i += 1

    if test_points is not None:
        for i in range(len(test_points)):
            pt = test_points[i]
            if i == 0:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black', label='test data')
            else:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black')

    ax.set_xlim((-5.5, 5.5))
    ax.set_ylim((-5.5, 5.5))
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.legend(loc='best')
    return ax
