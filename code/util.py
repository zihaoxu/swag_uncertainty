import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


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


def plot_decision_boundary(swag, x, y, ax, xlim, n_models,
                           poly_degree=1, test_points=None, shaded=True):
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
    interval = np.arange(-xlim, xlim, 0.1)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    xx = np.concatenate((x1, x2), axis=1)

    # Predict on mesh points
    if(poly_degree > 1):
        polynomial_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        xx = polynomial_features.fit_transform(xx)

    alpha_line = 0.2
    linewidths = 0.1
    i = 0

    for _ in range(n_models):
        yy = swag.predict(torch.Tensor(xx), [0, 1], S=1, expanded=False)
        yy = yy.reshape((n, n))

        # Plot decision surface
        x1 = x1.reshape(n, n)
        x2 = x2.reshape(n, n)
        if shaded:
            ax.contourf(x1, x2, yy, alpha=0.1 / (i + 1)**2, cmap='bwr')
        ax.contour(x1, x2, yy, colors='black', linewidths=linewidths, alpha=alpha_line)

        i += 1

    if test_points is not None:
        for i in range(len(test_points)):
            pt = test_points[i]
            if i == 0:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black', label='test data')
            else:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black')

    ax.set_xlim((-xlim+0.5, xlim-0.5))
    ax.set_ylim((-xlim+0.5, xlim-0.5))
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.legend(loc='best')
    return ax


def uncertainty_measurement(X_test, expand_pred):
    ''' Measures model uncertainty in as the variance in SWAG model predictions
        Params:
            X_test: test points
            expand_pred: swag prediction in expanded form
    '''
    var_list = []
    for i in range(len(X_test)):
        preds_i = np.argmax(expand_pred[:, i], axis=1)
        var_list.append(np.var(preds_i))

    print("\nBegin uncertainty assessment...")
    for x, var in zip(X_test, var_list):
        print(f"Test point: {x} \tVariance in prediction: {var:.4f}")
        
def uncertainty_estimation(swag, X_test, pred,  X_valid, y_valid, ensemble = False, verbose = True):
    ''' Measures model uncertainty in as the variance in SWAG model predictions (for regression tasks) 
        Params:
            swag: SWAG model to be evaluated
            X_test: test points
            pred: swag predictions on the test points; dx1 array where d is the number of SWAG models
            X_valid: valid points
            y_valid: true valid y values
            ensemble: whether or not it is an ensemble of SWAG models
    '''
    var_list = []
    mean_list = []
    up_list = []
    low_list = []
    for i in range(len(X_test)):
        preds_i = pred[:, i]
        var_list.append(np.var(preds_i))
        mean_list.append(np.mean(preds_i,axis=0))
        up_list.append(np.percentile(preds_i, 2.5, axis=0))
        low_list.append(np.percentile(preds_i, 97.5, axis=0))

    if verbose:
        print("\nBegin uncertainty assessment...")
        if ensemble:
            ensemble_mse = []
            for s in swag:
                agg_mse = []
                X_valid_tensor = torch.as_tensor([X_valid]).reshape((-1,1))
                train_pred = s.predict(X_valid_tensor.float(),None, S=200, expanded=True)
                for i in range(train_pred.shape[0]):
                    agg_mse.append(mean_squared_error(y_valid,train_pred[i,:]))
                mse = np.mean(agg_mse,axis=0)
                ensemble_mse.append(mse)
            print('Valid MSE: ', np.mean(ensemble_mse))
            return mean_list, up_list, low_list, np.mean(ensemble_mse)
        else:
            agg_mse = []
            X_valid_tensor = torch.as_tensor([X_valid]).reshape((-1,1))
            train_pred = swag.predict(X_valid_tensor.float(),None, S=200, expanded=True)
            for i in range(train_pred.shape[0]):
                agg_mse.append(mean_squared_error(y_valid,train_pred[i,:]))
            mse = np.mean(agg_mse,axis=0)
            print('Valid MSE: ', mse)
            return mean_list, up_list, low_list, mse
        
    return mean_list, up_list, low_list, False

class SGD:
    def __init__(self, NN_class, **kwargs):
        # Neural Network related params
        self.NN_class = NN_class
        self.net = NN_class(**kwargs)
        self.params_1d, self.shape_lookup, self.len_lookup = model_param_to_1D(self.net)
        self.weigt_D = len(self.params_1d)
        self.optimizer = optim.SGD(self.net.parameters(), 1e-3, 0.9)
        self.loss_fn = nn.MSELoss()
        self.loss_list = []
    
    def fit(self, train_loader, train_epoch: int,log_freq: int = 2000,verbose: bool = True):
        self.train_loader = train_loader
        for i in range(train_epoch):
            self.weights_param,loss = self.net_step(i, log_freq, verbose, return_weights=True)
            self.loss_list.append(loss)
        return self.weights_param
    
    def net_step(self,
                 epoch: int,
                 log_freq: int,
                 verbose: bool,
                 return_weights: bool = False):
        
        if not self.optimizer:
            raise RuntimeError("Please compile the model before training.")

        # Store and print running_loss
        running_loss = 0.0
        losses = []
        for i, data in enumerate(self.train_loader, 0):
            
            X_train, y_train = data
            
            if len(y_train.shape) > 1:
                y_train = y_train.view(-1)
            
            self.optimizer.zero_grad()
            
            loss = self.loss_fn(self.net(X_train), y_train)
            
            loss.backward()
            
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if verbose and i % log_freq == log_freq-1:
                print('[Epoch: %d, \tIteration: %5d] \tTraining Loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / log_freq))
                losses.append(running_loss/log_freq)
                running_loss = 0.0
                
            if not verbose:
                losses.append(running_loss/log_freq)
                running_loss = 0.0
            

        # If return_weights
        if return_weights:
            new_weights, _, _ = model_param_to_1D(self.net)
            return new_weights,np.mean(losses)
        
    def predict(self, X_test):
        model_params = params_1d_to_weights(self.weights_param, self.shape_lookup, self.len_lookup)
        new_net = create_NN_with_weights(self.NN_class, model_params)
        output = new_net.forward(X_test)
        return output