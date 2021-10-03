import torch
import numpy as np
from model import BiPoCNN
from data_loader import CNNDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

channel = 4
xtrain = np.load('/home/featurize/biponator_dev/xtrain4.npy')
ytrain = np.load('/home/featurize/biponator_dev/ytrain4.npy')
xtest = np.load('/home/featurize/biponator_dev/xtest4.npy')
ytest = np.load('/home/featurize/biponator_dev/ytest4.npy')

device = torch.device('cuda')
model = BiPoCNN(in_channels=channel)
model.load_state_dict(torch.load('model_CNN_60_epochs_c4.pt'))
model.to(device)
model.eval()

params_t = {'batch_size': 12,
          'shuffle': True,
          'num_workers': 0}
validation_set = CNNDataset(xtest, ytest, channels=channel)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_t)

for id, input in enumerate(validation_generator):
    # LOAD THE DATA IN A BATCH
    data_cpu, target = input
    data = data_cpu.requires_grad_()
    data = data.to(device)

    output = model.forward(data)
    mean_out = torch.mean(output)
    mean_out.backward()

    pred = np.round(output.detach().cpu().numpy())
    target = target.float().tolist()
    val_acc = accuracy_score(pred, target)
    print(val_acc)

    saliency = abs(data_cpu.grad.data).detach().cpu().numpy()
    saliency = np.mean(saliency, axis=0)
    plt.figure()
    plt.plot(saliency[0].flatten())
    # plt.figure()
    plt.plot(saliency[1].flatten())
    # plt.figure()
    plt.plot(saliency[2].flatten())
    # plt.figure()
    plt.plot(saliency[3].flatten())
    plt.show()

    # mean_data_0 = torch.std(data[target == 0], axis=0).detach().cpu().numpy()
    # mean_data_1 = torch.std(data[target == 1], axis=0).detach().cpu().numpy()
    # plt.figure()
    # plt.plot(mean_data_0)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(mean_data_1)
    # plt.show()

    break



