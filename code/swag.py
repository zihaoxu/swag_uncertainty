import torch
import numpy as np


class SWAG:
    """ Implements the SWAG paper: https://arxiv.org/pdf/1902.02476.pdf
    """
    def __init__(self, nn, K):
        ''' Params:
                nn (): the NN on which Swag is performed
                K (int): maximum number of columns in deviation matrix
        '''

        # Neural Network related params
        self.nn = nn
        self.weigt_D = nn.get_weights().flatten().shape[0]

        # SWAG params
        self.K = K

    def init_storage(self):
        first_mom = np.zeros(self.weigt_D)
        second_mom = np.zeros(self.weigt_D)
        D = np.zeros((self.weigt_D, self.K))
        return first_mom, second_mom, D

    def nn_step(self,
                train_mode: bool = False):
        if not self.optimizer:
            raise RuntimeError("Please compile the model before training.")

        for X_train, y_train in self.train_loader():
            self.optimizer.zero_grad()
            self.loss_fn(self.nn(X_train), y_train).backward()
            self.optimizer.step()
        # Training mode
        if train_mode:
            self.train_scheduler.step()
        # Swag mode
        else:
            self.swa_scheduler.step()

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
        second_mom_step = np.pow(new_weights, 2)
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
                optimizer: torch.optim.Optimizer,
                lr: float,
                loss_fn: torch.nn.modules.loss._Loss,
                train_scheduler: torch.optim.lr_scheduler._LRScheduler,
                swa_scheduler: torch.optim.lr_scheduler._LRScheduler):
        ''' Compiles the model
        '''
        self.optimizer = optimizer(self.nn, lr)
        self.loss_fn = loss_fn
        self.train_scheduler = train_scheduler
        self.swa_scheduler = swa_scheduler

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
        if (swag_epoch // c) < self.K:
            raise ValueError(f"swag_epoch//c={swag_epoch//c} needs to be at least K={K}")

        # Init storage
        first_mom, second_mom, D = self.init_storage()

        # Train nn for train_epoch
        for i in range(train_epoch):
            self.nn_step()

        # Perform SWAG inference
        for i in range(swag_epoch):
            # Perform SGD for 1 step
            new_weights = self.nn_step(return_weights=True)

            # Update the first and second moms
            n_model = i // c
            first_mom, second_mom = self.update_moments(n_model,
                                                        first_mom, second_mom)

            # Update D matrix
            if i % c == 0:
                D = self.update_D(D, i, first_mom, new_weights)
        return first_mom, second_mom, D

    def predict(self, test_loader):
        pass
