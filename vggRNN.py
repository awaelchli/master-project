import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


class VGGRNN(nn.Module):

    def __init__(self, nhidden, nlayers=1, dropout=0.5):
        super(VGGRNN, self).__init__()

        # VGG without classifier
        self.vgg = models.vgg19(pretrained=True).features

        self.rnn = nn.LSTM(input_size=214016,
                           hidden_size=nhidden,
                           num_layers=nlayers,
                           batch_first=True,
                           dropout=dropout)

        self.nhidden = nhidden
        self.nlayers = nlayers
        self.dropout = dropout

        #self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.vgg.bias.data.fill_(0)
        self.vgg.weight.data.uniform_(-initrange, initrange)
        self.rnn.bias.data.fill_(0)
        self.rnn.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.vgg(input)
        output, hidden = self.rnn(emb, hidden)

        #TODO: define output transform

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers, batch_size, self.nhidden).zero_()))
