import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import NASBiPoCNN
from data_loader import CNNDataset
from train import train, test

model_path = 'model_CNN_60_epochs.pt'
xtest = np.load('xtest.npy')
ytest = np.load('ytest.npy')

#checking the model is giving out the right output
device = torch.device('cuda')
model = NASBiPoCNN(in_channels=1)
model.load_state_dict(torch.load(model_path))
model.to(device)
print(model)

params_t = {'batch_size': 512,
          'shuffle': True,
          'num_workers': 4}
validation_set = CNNDataset(xtest, ytest, channels=1)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_t)

val_acc_i = test(model, validation_generator)

        