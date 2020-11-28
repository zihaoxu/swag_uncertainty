from torch.utils.data import Dataset
import numpy as np
import torch


def get_classification_data(samples=100):
    # Generate a toy dataset for classification
    class_0 = np.random.multivariate_normal([-1, -1], 0.5 * np.eye(2), samples)
    class_1 = np.random.multivariate_normal([1, 1], 0.5 * np.eye(2), samples)
    x = np.vstack((class_0, class_1))
    y = np.array([0] * samples + [1] * samples)
    # Define test data points
    test_points = [(-1, -1.5), (1, 1.5), (-5, -5), (5, 5), (-5, 3.5),
                   (5, -3.5), (-1, 1), (1, -0.2)]
    return (x, y), test_points


class ClassificationDataSet(Dataset):
    def __init__(self, mode, n_samples=1000):
        self.mode = mode
        self.n_samples = n_samples
        self.test_points = [(-1, -1.5), (1, 1.5), (-5, -5), (5, 5), (-5, 3.5),
                            (5, -3.5), (-1, 1), (1, -0.2)]
        if mode == 'train':
            self.X, self.Y = self.get_samples()
            print(self.X.shape, self.Y.shape)

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
