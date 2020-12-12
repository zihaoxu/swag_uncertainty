from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.datasets import make_moons


class ClassificationDataSet(Dataset):
    def __init__(self, mode, n_samples=500):
        self.mode = mode
        self.n_samples = n_samples
        self.test_points = [(-1, -1.5), (1, 1), (-5, -5), (5, 5), (-5, 3.5),
                            (5, -3.5), (-1, 1), (1, -1)]
        if mode == 'train':
            self.X, self.Y = self.get_samples()

    def get_samples(self):
        half_n_sample = int(np.floor(self.n_samples/2))
        class_0 = np.random.multivariate_normal([-1, -1], 0.5 * np.eye(2), half_n_sample)
        class_1 = np.random.multivariate_normal([1, 1], 0.5 * np.eye(2), half_n_sample)
        x = np.vstack((class_0, class_1))
        y = np.array([0] * half_n_sample + [1] * half_n_sample).reshape(-1, 1)
        return x, y

    def __len__(self):
        if self.mode == 'train':
            return self.n_samples
        else:
            return len(self.test_points)

    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt = torch.Tensor(self.X[idx])
            oupt = torch.Tensor(self.Y[idx]).type(torch.LongTensor)
            return inpt, oupt
        else:
            inpt = torch.Tensor(self.test_points[idx])
            return inpt


class TwoMoons(Dataset):
    def __init__(self, mode, n_samples=500, noise=0.1):
        self.mode = mode
        self.n_samples = n_samples
        self.noise = noise
        self.test_points = [(1, -.5), (0, 1), (-0.5, 0.25), (0.5, 0.25), (1.5, 0.25),
                            (-1, -1.5), (-1, 1.5), (2, -1.5), (2, 1.5)]
        if mode == 'train':
            self.X, self.Y = self.get_samples()

    def get_samples(self):
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise)
        return X, y.reshape(-1, 1)

    def __len__(self):
        if self.mode == 'train':
            return self.n_samples
        else:
            return len(self.test_points)

    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt = torch.Tensor(self.X[idx])
            oupt = torch.Tensor(self.Y[idx]).type(torch.LongTensor)
            return inpt, oupt
        else:
            inpt = torch.Tensor(self.test_points[idx])
            return inpt
        
class RegressionDataSet(Dataset):
    def __init__(self, mode, gap, n_samples=20, noise_variance=0.3):
        self.mode = mode
        self.n_samples = n_samples
        self.noise_variance = noise_variance
        self.X, self.Y, self.X_test = self.get_samples(gap)

    def get_samples(self, gap = True):
        if gap:
            x_train = np.hstack((np.linspace(-1, -0.5, self.n_samples), np.linspace(0.5, 1, self.n_samples)))
            f = lambda x: 3 * x**3
            y_train = f(x_train) + np.random.normal(0, self.noise_variance**0.5, 2 * self.n_samples)
        else:
            x_train = np.linspace(-1, 1, 2*100)
            f = lambda x: -10*x**2 + 3 if (-0.5<=x) & (x<=0.5) else 3 * x**3 
            y_train = [f(i) for i in x_train] + np.random.normal(0, 0.3**0.5, 2 * 100)
        x_test = np.array(list(set(list(np.hstack((np.linspace(-1, 1, 200), x_train))))))
        x_test = np.sort(x_test)
        return x_train, y_train, x_test

    def __len__(self):
        if self.mode == 'train':
            return len(self.X)
        else:
            return len(self.X_test)

    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt = torch.as_tensor([self.X[idx]])
            oupt = torch.as_tensor([self.Y[idx]])
            return inpt, oupt
        else:
            inpt = torch.as_tensor([self.X_test[idx]])
            return inpt

