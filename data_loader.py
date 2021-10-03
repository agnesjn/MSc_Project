import torch
import numpy as np


class CNNDataset(torch.utils.data.Dataset):
    'SoLid 1D waveform dataset for BiPonator CNN'
    def __init__(self, samples, labels, channels=1):
        'Initialization'
        self.labels = labels
        self.samples = samples
        self.size = 30720
        self.channels = channels
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample string
        X = torch.from_numpy(self.samples[index])
        X = X.type(torch.float)
        if self.channels == 1:
            X = X.unsqueeze(0)
        # need to treat the value as an array x -> [x]
        y = torch.from_numpy(np.asarray(self.labels[index]))
        y = y.type(torch.float)
        return X, y
