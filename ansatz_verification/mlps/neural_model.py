import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
from torchvision import models
from torch.nn.functional import upsample
from copy import deepcopy
import torch.nn.functional as F


class Nonlinearity(torch.nn.Module):
    def __init__(self, name='relu'):
        super(Nonlinearity, self).__init__()
        self.name = name

    def forward(self, x):
        if self.name == 'relu':
            return F.relu(x)
        if self.name == 'sigmoid':
            return torch.sigmoid(x)
        if self.name == 'leaky_relu':
            return F.leaky_relu(x)
        if self.name == 'sine':
            return torch.sin(x)
        if self.name == 'tanh':
            return torch.tanh(x)


class Net(nn.Module):

    def __init__(self, dim, depth=1, width=1024, num_classes=2, act_name='relu'):
        super(Net, self).__init__()
        bias = False
        self.dim = dim
        self.width = width
        self.depth = depth
        self.name = act_name

        if depth == 1:
            self.first = nn.Linear(dim, width, bias=bias)
            self.fc = nn.Sequential(Nonlinearity(name=self.name),
                                    nn.Linear(width, num_classes, bias=bias))
        else:
            module = nn.Sequential(Nonlinearity(name=self.name),
                                   nn.Linear(width, width, bias=bias))
            num_layers = depth - 1
            self.first = nn.Sequential(nn.Linear(dim, width,
                                                 bias=bias))
            self.middle = nn.ModuleList([deepcopy(module) \
                                         for idx in range(num_layers)])

            self.last = nn.Sequential(Nonlinearity(name=self.name),
                                      nn.Linear(width, num_classes,
                                                bias=bias))


    def forward(self, x):
        if self.depth == 1:
            return self.fc(self.first(x))
        else:
            o = self.first(x)
            for idx, m in enumerate(self.middle):
                o = m(o)
            o = self.last(o)
            return o
