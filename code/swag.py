import torch
import numpy as np
from util import model_param_to_1D, params_1d_to_weights, create_NN_with_weights


class SWAG:
    """ Implements the SWAG paper: https://arxiv.org/pdf/1902.02476.pdf
    """
    def __init__(self, NN_class, K):
        ''' Params:
                nn (): the NN on which Swag is performed
                K (int): maximum number of columns in deviation matrix
        '''

        # Neural Network related params
        self.NN_class = NN_class
        self.net = NN_class()
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
                 train_mode: bool = False,
                 return_weights: bool = False):
        if not self.optimizer:
            raise RuntimeError("Please compile the model before training.")

        # Store and print running_loss
        running_loss = 0.0
        for i, data in enumerate(self.train_loader, 0):
            X_train, y_train = data
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.net(X_train), y_train)
            loss.backward()
            self.optimizer.step()

            # Training mode
            if train_mode:
                self.train_scheduler.step()
            # Swag mode
            else:
                self.swa_scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

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
                lr: float,
                momentum: float,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.modules.loss._Loss,
                train_scheduler,
                swa_scheduler):
        ''' Compiles the model
        '''
        self.optimizer = optimizer(self.net.parameters(), lr, momentum)
        self.loss_fn = loss_fn
        self.train_scheduler = train_scheduler(self.optimizer, T_max=100)
        const_lr = lambda x: 1
        self.swa_scheduler = swa_scheduler(self.optimizer, lr_lambda=const_lr)

    def fit(self,
            train_loader,
            train_epoch: int,
            swag_epoch: int,
            c: int = 1) -> (np.array, np.array, np.ndarray):
        ''' Main func that fits the swag model
            Params:
                train_loader()
                train_epoch(int): the number of steps to train NN
                swag_epoch(int): number of steps to perform swag
                c(int): moment update frequency, thinning factor
            Output:
                first_mom(np.ndarray): the trained first mom
                second_mom(np.ndarray): the trained second mom
                D(np.ndarray): the trained deviation matrix
        '''
        # Save train_loader
        self.train_loader = train_loader

        if (swag_epoch // c) < self.K:
            raise ValueError(f"swag_epoch//c={swag_epoch//c} needs to be at least K={K}")

        # Init storage
        first_mom, second_mom, D = self.init_storage()

        # Train nn for train_epoch
        print("Beging NN model training:")
        for i in range(train_epoch):
            self.net_step(i)

        # Perform SWAG inference
        print("\nBeging SWAG training:")
        for i in range(swag_epoch):
            # Perform SGD for 1 step
            new_weights = self.net_step(i, return_weights=True)

            # Update the first and second moms
            n_model = i // c
            first_mom, second_mom = self.update_moments(n_model,
                                                        first_mom,
                                                        second_mom,
                                                        new_weights)

            # Update D matrix
            if i % c == 0:
                D = self.update_D(i, D, first_mom, new_weights)
        return first_mom, second_mom, D

    def predict(self, X_test, first_mom, second_mom, D, S):
        """ Params:
                X_test(np.ndarray): test data
                first_mom(np.ndarray): the trained first mom
                second_mom(np.ndarray): the trained second mom
                D(np.ndarray): the trained deviation matrix
                S(int): number of posterior inference steps
            Outputs:
                predictions: model predictions
        """
        # Initialize storage for predictions
        predictions = np.empty((X_test.shape[0], S))

        # Generate weight samples
        weight_samples = self.weight_sampler(first_mom, second_mom, D)
        # Recreate new net
        for s, model_params in enumerate(weight_samples):
            new_net = create_NN_with_weights(self.NN_class, model_params)
            predictions[s] = new_net.predict(X_test)
        return predictions

    def weight_sampler(first_mom, second_mom, D, S):
        pass
