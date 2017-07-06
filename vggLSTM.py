import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


class VGGLSTM(nn.Module):

    def __init__(self, input_size, nhidden, nlayers=1, dropout=0.5):
        super(VGGLSTM, self).__init__()



        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=nhidden,
                           num_layers=nlayers,
                           batch_first=True,
                           dropout=dropout)

        self.input_size = input_size
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.dropout = dropout

        #self.init_weights()


    def init_weights(self):
        initrange = 0.1
        #self.vgg.bias.data.fill_(0)
        #self.vgg.weight.data.uniform_(-initrange, initrange)
        self.rnn.bias.data.fill_(0)
        self.rnn.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)

        #TODO: define output transform

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers, batch_size, self.nhidden).zero_()))
