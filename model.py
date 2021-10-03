import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np


device = torch.device('cuda')

torch.set_printoptions(edgeitems=2)

torch.manual_seed(123)

class BiPoCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # need to check of difference of padding option
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=7, kernel_size=2, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(7, 14, kernel_size=2, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(14 * 768, 128) # 2x max pooling gives 250 depth with 14 filters
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 14 * 768)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        probs = F.sigmoid(out)
        return probs


class NASBiPoCNN(nn.Module):
    def __init__(self, in_channels=1, var_map_path='var_map.npy'):
        super().__init__()
        
        # Set up candidate paths for layer 1
        self.conv1 = nn.ModuleList([])
        for i in range(5):
            self.conv1.append(nn.Conv1d(in_channels=in_channels, out_channels=7, kernel_size=2 ** (i + 1), stride=1, padding=2 ** i))
        self.weight1 = Parameter(torch.ones(len(self.conv1), requires_grad=True))
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        # Set up candidate paths for layer 2
        self.conv2 = nn.ModuleList([])
        for i in range(5):
            self.conv2.append(nn.Conv1d(7, 14, kernel_size=2 ** (i + 1), stride=1, padding=2 ** i))
        self.weight2 = Parameter(torch.ones(len(self.conv2), requires_grad=True))
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(14 * 768, 128) # 2x max pooling gives 250 depth with 14 filters
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

        self.var_map = torch.from_numpy(np.load(var_map_path)).to(device)

    def forward(self, x):
        out = torch.tensor(0., requires_grad=True)
        # w1 = F.gumbel_softmax(self.weight1, tau=10.)
        for i in range(len(self.conv1)):
            out = out + self.weight1[i] * self.conv1[i](x)
        out = self.pool1(self.act1(out))
        out2 = torch.tensor(0., requires_grad=True)
        # w2 = F.gumbel_softmax(self.weight2, tau=10.)
        for i in range(len(self.conv2)):
            out2 = out2 + self.weight2[i] * self.conv2[i](out)
        out = self.pool2(self.act2(out2))
        out = out.view(-1, 14 * 768)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        probs = F.sigmoid(out)
        return probs

    def sal_foward(self, x):
        # Initialize output of layer 1
        out = torch.tensor(0., requires_grad=True)

        # Calculate path probability of layer 1 from architecture weights
        w1 = F.gumbel_softmax(self.weight1)

        # Initialize correlation coefficient of layer 1
        corr1 = torch.zeros(len(self.conv1), requires_grad=False).to(device)

        # Initialize path reward of layer 1
        reward = torch.tensor(0., requires_grad=True).to(device)

        # Calculate output and rewards in a FOR loop
        for i in range(len(self.conv1)):

            # Get output for the path
            o1 = self.weight1[i] * self.conv1[i](x)
            out = out + o1

            # Set the gradient of input data (image) to none, preparing for the saliency map.
            x.grad = None
            self.zero_grad()

            # Doing back-propagation on the input data rather than the network weights.
            o1.sum().backward(retain_graph=True)

            # Get the saliency map
            sal = torch.abs(x.grad.data.sum(dim=0))

            # Calculate the correlation coefficient between saliency map and variance map
            corr1[i] = torch.cosine_similarity(sal[0], self.var_map, dim=0)

        # Get the overall reward for layer 1 (using policy gradient algorithm with baseline)
        corr1 = corr1 - corr1.mean()
        reward = reward - corr1.dot(torch.log(w1))
        
        # METHOD BELOW IS THE SAME WITH CODE ABOVE
        out = self.pool1(self.act1(out))

        w2 = F.gumbel_softmax(self.weight2)
        corr2 = torch.zeros(len(self.conv2), requires_grad=False).to(device)
        for i in range(len(self.conv2)):
            o2 = self.weight2[i] * self.conv2[i](out)
            x.grad = None
            self.zero_grad()
            o2.sum().backward(retain_graph=True)
            sal = torch.abs(x.grad.data.sum(dim=0))
            corr2[i] = torch.cosine_similarity(sal[0], self.var_map, dim=0)
        corr2 = corr2 - corr2.mean()
        reward = reward - corr2.dot(torch.log(w2))
        
        return reward


