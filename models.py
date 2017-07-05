import torch
import torch.nn as nn
from torch.autograd import Variable


class CONVLSTM(nn.LSTM):

    def __init__(self, input_size, hidden_size, output_size):
        super(CONVLSTM, self).__init__()

        self.hidden_size = hidden_size


        self.h2h = nn.Conv2d(hidden_size, hidden_size, 5, padding = 2)

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden
                              ), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return hidden, output

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

vgg = models.vgg19(pretrained = True)
print(vgg)