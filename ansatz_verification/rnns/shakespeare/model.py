# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(2*hidden_size, hidden_size)
        self.i2o = nn.Linear(2*hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)

        #print("i2h", self.i2h.weight.shape)
        #print("i2o", self.i2o.weight.shape)

        torch.nn.init.xavier_uniform_(self.i2h.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.i2o.weight, gain=0.01)

        self.softmax = nn.LogSoftmax(dim=1)

        self.embedding = nn.Embedding(input_size, hidden_size)

    def forward(self, x, hidden, i2h_perturb=None, i2o_perturb=None):
        #print("hidden", hidden.shape, "xin", x.shape)
        x = self.embedding(x)
        #print("hidden", hidden.shape, "xemb", x.shape)
        input_combined = torch.cat((x, hidden), 1)
        if i2h_perturb is not None:
            input_combined_h = input_combined + i2h_perturb
        else:
            input_combined_h = input_combined
        if i2o_perturb is not None:
            input_combined_o = input_combined + i2o_perturb
        else:
            input_combined_o = input_combined

        hidden = self.i2h(input_combined_h)
        output = self.i2o(input_combined_o)

        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, n):
        return torch.zeros(n, self.hidden_size)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        self.Wz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wy = nn.Linear(hidden_size, output_size, bias=False)

        torch.nn.init.xavier_uniform_(self.Wz.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Wr.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Wh.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Uz.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Ur.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Uh.weight, gain=0.01)


    def forward(self, x, h, p1=None, p2=None, p3=None, p4=None, p5=None, p6=None):
        
        x = self.embedding(x)

        if p1 is not None:
            x_Wz = x + p1
            x_Wr = x + p2
            x_Wh = x + p3

            h_Uz = h + p4
            h_Ur = h + p5
            h_Uh = h + p6
        else:
            x_Wz = x
            x_Wr = x
            x_Wh = x

            h_Uz = h
            h_Ur = h
            h_Uh = h

        z = nn.Sigmoid()(self.Wz(x_Wz) + self.Uz(h_Uz))
        r = nn.Sigmoid()(self.Wr(x_Wr) + self.Ur(h_Ur))
        ht = nn.Tanh()(self.Wh(x_Wh) + r * self.Uh(h_Uh))

        h = z*h + (1-z)*ht
        out = nn.LogSoftmax(dim=1)(self.Wy(h))
        return out, h

    def init_hidden(self, n):
        return torch.zeros(n, self.hidden_size)

