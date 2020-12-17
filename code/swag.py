import torch
import numpy as np
from util import model_param_to_1D, params_1d_to_weights, create_NN_with_weights


class SWAG:
    """ Implements the SWAG paper: https://arxiv.org/pdf/1902.02476.pdf
    """
    def __init__(self, NN_class, K, pretrained=False, NNModel=None, **kwargs):
        ''' Params:
                nn (nn.Module): the NN on which Swag is performed
                K (int): maximum number of columns in deviation matrix
        '''

        # Neural Network related params
        self.NN_class = NN_class
        if not pretrained:
            self.net = NN_class(**kwargs)
        else:
            self.net = NNModel
        self.params_1d, self.shape_lookup, self.len_lookup = model_param_to_1D(self.net)
        self.weigt_D = len(self.params_1d)

        # SWAG params
        self.K = K

    def init_storage(self):
        first_mom = np.zeros(self.weigt_D)
        second_mom = np.zeros(self.weigt_D)
        D = np.zeros((self.weigt_D, self.K))
        return first_mom, second_mom, D

    def net_step(self,
                 epoch: int,
                 log_freq: int,
                 verbose: bool,
                 train_mode: bool = False,
                 return_weights: bool = False):
        if not self.optimizer:
            raise RuntimeError("Please compile the model before training.")

        # Store and print running_loss
        running_loss = 0.0
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
                running_loss = 0.0

        # update swa_scheduler in Swag mode
        if not train_mode:
            self.swa_scheduler.step()

        # If return_weights
        if return_weights:
            new_weights, _, _ = model_param_to_1D(self.net)
            return new_weights

    def update_moments(self,
                       n: int,
                       first_mom: np.ndarray,
                       second_mom: np.ndarray,
                       new_weights: np.ndarray) -> (np.ndarray, np.ndarray):
        ''' Updates the momements storage vectors
            Params:
                n(int): number of models so far
                first_mom(np.ndarray): storage for first moment
                second_mom(np.ndarray): storage for second moment
                new_weights(np.ndarray): updated weights
            Output:
                first_mom_new(np.ndarray): updated first moment
                second_mom_new(np.ndarray): updated second moment
        '''
        second_mom_step = np.power(new_weights, 2)
        first_mom_new = (n*first_mom+new_weights) / (n+1)
        second_mom_new = (n*second_mom+second_mom_step) / (n+1)
        return first_mom_new, second_mom_new

    def update_D(self,
                 swag_step: int,
                 D: np.ndarray,
                 first_mom: np.ndarray,
                 new_weights: np.ndarray):
        ''' Update the Deviation matrix
            Params:
                swag_step(int): step number of swag epochs
                D(np.ndarray): deviation matrix
                first_mom(np.ndarray): storage for first moment
                new_weights(np.ndarray): updated weights
            Output:
                D_new: updated deviation matrix
        '''
        D_new = D.copy()
        update_col = swag_step % self.K
        diff_vec = new_weights - first_mom
        D_new[:, update_col] = diff_vec
        return D_new

    def compile(self,
                objective: str,
                lr: float,
                swa_const_lr: float,
                momentum: float,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.modules.loss._Loss,
                swa_scheduler):
        ''' Compiles the model
        '''
        if objective not in ['regression', 'classification']:
            raise ValueError("objective must be one of 'regression' or 'classification'.")
        self.objective = objective
        self.optimizer = optimizer(self.net.parameters(), lr, momentum)
        self.loss_fn = loss_fn
        self.swa_const_lr = lambda x: swa_const_lr
        self.swa_scheduler_cls = swa_scheduler

    def compile_customize_optimizer(self,
                objective: str,
                swa_const_lr: float,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.modules.loss._Loss,
                swa_scheduler):
        ''' Compiles the model
        '''
        if objective not in ['regression', 'classification']:
            raise ValueError("objective must be one of 'regression' or 'classification'.")
        self.objective = objective
        self.optimizer = optimizer(self.net.parameters())
        self.loss_fn = loss_fn
        self.swa_const_lr = lambda x: swa_const_lr
        self.swa_scheduler_cls = swa_scheduler
        
    def fit(self,
            train_loader,
            train_epoch: int,
            swag_epoch: int,
            c: int = 1,
            log_freq: int = 2000,
            verbose: bool = True,
            pretrained: bool = False
           ) -> (np.array, np.array, np.ndarray):
        ''' Main func that fits the swag model
            Params:
                train_loader()
                train_epoch(int): the number of steps to train NN
                swag_epoch(int): number of steps to perform swag
                c(int): moment update frequency, thinning factor
                pretrained: If initialize with pretrained-weights
            Output:
                first_mom(np.ndarray): the trained first mom
                second_mom(np.ndarray): the trained second mom
                D(np.ndarray): the trained deviation matrix

        '''
        # Save train_loader
        self.train_loader = train_loader

        if (swag_epoch // c) < self.K:
            raise ValueError(f"swag_epoch//c={swag_epoch//c} needs to be at least K={self.K}")

        # Init storage
        first_mom, second_mom, D = self.init_storage()
        
        if pretrained == True:
            first_mom = self.params_1d
            second_mom = self.params_1d**2

        # Train nn for train_epoch
        if verbose:
            print("Begin NN model training...")
        for i in range(train_epoch):
            self.net_step(i, log_freq, verbose, train_mode=True)

        # Perform SWAG inference
        if verbose:
            print("\nBegin SWAG training...")
        for i in range(swag_epoch):
            # Activate scheduler
            self.swa_scheduler = self.swa_scheduler_cls(self.optimizer,
                                                        lr_lambda=self.swa_const_lr)

            # Perform SGD for 1 step
            new_weights = self.net_step(i, log_freq, verbose,
                                        return_weights=True,
                                        train_mode=False)

            # Update the first and second moms
            n_model = i // c
            first_mom, second_mom = self.update_moments(n_model,
                                                        first_mom,
                                                        second_mom,
                                                        new_weights)
            # Update D matrix
            if i % c == 0:
                D = self.update_D(i, D, first_mom, new_weights)

        # Save the learn moments and D
        self.first_mom = first_mom
        self.second_mom = second_mom
        self.D = D
        return first_mom, second_mom, D

    def weight_sampler(self):
        """ Outputs:
                weights(theta): the weights sampled from the multinomial distribution
        """
        # Store theta_SWA
        mean = torch.tensor(self.first_mom, requires_grad=False)
        # Compute the sigma diagonal matrix
        sigma_diag = torch.tensor(self.second_mom - self.first_mom**2)
        # Draw a sample from the N(0,sigma_diag)
        var_sample = ((1/2)*sigma_diag).sqrt() * torch.randn_like(sigma_diag, requires_grad=False)
        # Prepare the covariance matrix D
        D_tensor = torch.tensor(self.D, requires_grad=False)
        # Draw a sample from the N(0,D)
        D_sample = np.sqrt((1/2*self.K-1)) * D_tensor @ torch.randn_like(D_tensor[0, :], requires_grad=False)
        D_reshaped = D_sample.view_as(mean)
        # Add mean and two variance samples together
        weights = mean + var_sample + D_reshaped
        return weights

    def predict(self, X_test, classes, S, expanded=False):
        """ Params:
                X_test(np.ndarray): test data
                classes(np.ndarray): list of all labels
                first_mom(np.ndarray): the trained first mom
                second_mom(np.ndarray): the trained second mom
                D(np.ndarray): the trained deviation matrix
                S(int): number of posterior inference steps
            Outputs:
                predictions: model predictions
        """
        if self.objective == 'classification':
            return self._predict_classification(X_test, classes, S, expanded)
        elif self.objective == 'regression':
            return self._predict_regression(X_test, classes, S, expanded)

    def _predict_classification(self, X_test, classes, S, expanded):
        # Initialize storage for probabilities
        prob_matrix = np.zeros((S, len(X_test), len(classes)))

        # Generate weight samples
        weight_samples = []
        for i in range(S):
            samples = self.weight_sampler()
            weight_samples.append(samples)

        # Recreate new net
        for s, weight_param in enumerate(weight_samples):
            model_params = params_1d_to_weights(weight_param, self.shape_lookup, self.len_lookup)
            new_net = create_NN_with_weights(self.NN_class, model_params)
            output = new_net.forward(X_test)
            prob_matrix[s] = output.detach().numpy()

        # Whether return the expanded prob_matrix
        if expanded:
            return prob_matrix
        else:
            mean_pred = np.mean(prob_matrix, axis=0)
            return np.argmax(mean_pred, axis=1)

    def _predict_regression(self, X_test, classes, S, expanded):
        # Initialize storage for results
        out_matrix = np.zeros((S, len(X_test)))

        # Generate weight samples
        weight_samples = []
        for i in range(S):
            samples = self.weight_sampler()
            weight_samples.append(samples)

        # Recreate new net
        for s, weight_param in enumerate(weight_samples):
            model_params = params_1d_to_weights(weight_param, self.shape_lookup, self.len_lookup)
            new_net = create_NN_with_weights(self.NN_class, model_params)
            output = new_net.forward(X_test)
            out_matrix[s] = output.detach().numpy().flatten()

        # Return outputs
        return out_matrix
