import torch
from torch import nn as nn
from torch.autograd import Variable

from flownet.models import flownets
FLOWNET_MODEL_PATH = '../data/Pretrained Models/flownets_pytorch.pth'


class FullPose7DModel(nn.Module):

    def __init__(self, input_size, hidden=500, nlayers=3, dropout=0):
        super(FullPose7DModel, self).__init__()

        self.hidden = hidden
        self.nlayers = nlayers

        flownet = flownets(FLOWNET_MODEL_PATH)
        flownet.train(False)

        self.layers = flownet
        self.layers = torch.nn.Sequential(
            flownet.conv1,
            flownet.conv2,
            flownet.conv3,
            flownet.conv3_1,
            flownet.conv4,
            flownet.conv4_1,
            flownet.conv5,
            flownet.conv5_1,
            flownet.conv6,
            flownet.conv6_1,
        )

        fout = self.flownet_output_size(input_size)

        self.lstm = nn.LSTM(
            input_size=fout[1] * fout[2] * fout[3],
            hidden_size=self.hidden,
            num_layers=self.nlayers,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden, 6)

        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()

    def flownet_output_size(self, input_size):
        var = Variable(torch.zeros(1, 6, input_size[0], input_size[1]), volatile=True)
        if next(self.layers.parameters()).is_cuda:
            var = var.cuda()
        out = self.layers(var)
        return out.size(0), out.size(1), out.size(2), out.size(3)

    def forward(self, input, state=None):
        # Input shape: [1, sequence, channels, h, w]
        input.squeeze_(0)

        n = input.size(0)
        first = input[:n-1]
        second = input[1:]

        # New shape: [sequence - 1, 2 * channels, h, w]
        pairs = torch.cat((first, second), 1)

        assert pairs.size(0) == n - 1

        # Using batch mode to forward sequence
        pairs = self.layers(pairs)

        init = state
        if not state:
            h0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
            c0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
            if input.is_cuda:
                h0 = h0.cuda()
                c0 = c0.cuda()
            init = (h0, c0)

        outputs, state = self.lstm(pairs.view(1, n - 1, -1), init)

        outputs = self.drop(outputs)
        predictions = self.fc(outputs.squeeze(0))

        return predictions, state

    def train(self, mode=True):
        super(FullPose7DModel, self).train(mode)
        self.lstm.train(mode)

    def eval(self):
        super(FullPose7DModel, self).eval()
        self.lstm.eval()

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters()) + list(self.layers.parameters())
        return params