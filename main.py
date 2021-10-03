import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import NASBiPoCNN
from data_loader import CNNDataset
from train import train, test

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

channel = 1
xtrain = np.load('xtrain.npy')
ytrain = np.load('ytrain.npy')
xtest = np.load('xtest.npy')
ytest = np.load('ytest.npy')

#checking the model is giving out the right output
device = torch.device('cuda')
model = NASBiPoCNN(in_channels=channel)
model.to(device)
print(model)

numel_list = [p.numel() for p in model.parameters()]

#Parameters
params = {'batch_size': 512,
          'shuffle': True,
          'num_workers': 4}
training_set = CNNDataset(xtrain, ytrain, channels=channel)
training_generator = torch.utils.data.DataLoader(training_set, **params,)
trainiter = iter(training_generator)

params_t = {'batch_size': 512,
          'shuffle': True,
          'num_workers': 4}
validation_set = CNNDataset(xtest, ytest, channels=channel)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_t)

n_epochs = 60
valid_loss_min = np.Inf
val_acc = []
train_loss = []
train_acc = []
total_step = len(training_generator)

loss_fn = nn.BCELoss()
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.00005)
optimizer =torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer_arch = torch.optim.SGD([model.weight1, model.weight2], lr=0.00001)

valid_acc_max = 0.0
for epoch in range(1, n_epochs + 1):
    train_loss_i, train_acc_i = train(model, optimizer, optimizer_arch, loss_fn, training_generator, epoch, writer)
    if epoch % 1 == 0:
        val_acc_i = test(model, validation_generator)
        network_learned = val_acc_i > valid_acc_max
        train_loss.append(train_loss_i)
        train_acc.append(train_acc_i)
        val_acc.append(val_acc_i)
        writer.add_scalar('Accuracy/test', val_acc_i, epoch)
        if network_learned:
            valid_acc_max = val_acc_i
            print("Dectected improved accuracy - SAVING current model")
            torch.save(model.state_dict(), './model_CNN_60_epochs_c4.pt')
            print(
                "---------------------------------------------------------------------------------------------------------")
