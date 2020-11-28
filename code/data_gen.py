from torch.utils.data import Dataset
import numpy as np
import torch


class ClassificationDataSet(Dataset):
    def __init__(self, mode, n_samples=1000):
        self.mode = mode
        self.n_samples = n_samples
        self.test_points = [(-1, -1.5), (2, 1.5), (-5, -5), (5, 5), (-5, 3.5),
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
