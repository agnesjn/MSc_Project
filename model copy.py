import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
# from saliency_map import get_corr


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
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv1 = nn.ModuleList([])
        for i in range(5):
            self.conv1.append(nn.Conv1d(in_channels=in_channels, out_channels=7, kernel_size=2 ** (i + 1), stride=1, padding=2 ** i))
        self.weight1 = Parameter(torch.ones(len(self.conv1), requires_grad=True))
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.ModuleList([])
        for i in range(5):
            self.conv2.append(nn.Conv1d(7, 14, kernel_size=2 ** (i + 1), stride=1, padding=2 ** i))
        # self.conv2 = nn.Conv1d(7, 14, kernel_size=2, stride=1, padding=1)
        self.weight2 = Parameter(torch.ones(len(self.conv2), requires_grad=True))
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(14 * 768, 128) # 2x max pooling gives 250 depth with 14 filters
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = torch.tensor(0., requires_grad=True)
        w1 = F.gumbel_softmax(self.weight1)
        for i in range(len(self.conv1)):
            out = out + self.weight1[i] * self.conv1[i](x)
        out = self.pool1(self.act1(out))

        out2 = torch.tensor(0., requires_grad=True)
        w2 = F.gumbel_softmax(self.weight2)
        for i in range(len(self.conv2)):
            out2 = out2 + self.weight2[i] * self.conv2[i](out)
        out = self.pool2(self.act2(out2))
        out = out.view(-1, 14 * 768)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        probs = F.sigmoid(out)
        return probs


class BiPoCNN4(nn.Module):
    def __init__(self):
        super().__init__()
        # need to check of difference of padding option
        self.alpha1 = nn.Parameter(torch.randn(5))
        self.conv1 = torch.nn.ModuleList()
        for i in range(1, 6):
            self.conv1.append(nn.Conv1d(in_channels=4,out_channels=7,kernel_size=2 * i + 1, stride=1, padding=i))
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv1 = torch.nn.ModuleList()
        self.alpha2 = nn.Parameter(torch.randn(5))
        for i in range(1, 6):
            self.conv2.append(nn.Conv1d(in_channels=7, out_channels=14, kernel_size=2 * i + 1, stride=1, padding=i))
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(14 * 768, 64) # 2x max pooling gives 250 depth with 14 filters
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = []
        weight1 = torch.nn.functional.gumbel_softmax(self.alpha1)
        for i in range(5):
            out.append(weight1[i] * self.pool1(self.act1(self.conv1[i](x))))
        out = torch.stack(out, dim=0)
        out1 = torch.sum(out, dim=0)
        self.alpha1 = self.alpha1 + torch.from_numpy(get_corr(out1))
        out = []
        weight2 = torch.nn.functional.gumbel_softmax(self.alpha2)
        for i in range(5):
            out.append(weight2[i] * self.pool2(self.act2(self.conv2[i](out1))))
        out = torch.stack(out, dim=0)
        out2 = torch.sum(out, dim=0)
        self.alpha2 = self.alpha2 + torch.from_numpy(get_corr(out2))
        out = out2.view(-1, 14 * 768)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        probs = F.sigmoid(out)
        return probs