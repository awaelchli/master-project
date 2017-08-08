import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(self.input_channels + hidden_channels, 4 * hidden_channels, kernel_size, 1,
                              self.padding, bias=bias)

    def forward(self, input, h, c):
        combined = torch.cat((input, h), 1)
        a = self.conv(combined)
        split_size = int(a.size()[1] / self.num_features)
        (ai, af, ao, ag) = torch.split(a, split_size, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])))


class ConvLSTM(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.bias = bias

        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, internal_state=None):
        internal_state = []
        if not internal_state:
            # First forward pass in the time sequence
            # Initialize internal states of all layers
            for i in range(self.num_layers):
                bsize, _, height, width = input.size()
                (h, c) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[i], (height, width))
                if input.is_cuda:
                    h = h.cuda()
                    c = c.cuda()
                internal_state.append((h, c))

        # Forward pass through all layers in current time step
        x = input
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            (h, c) = internal_state[i]
            x, new_c = getattr(self, name)(x, h, c)
            internal_state[i] = (x, new_c)

        return x, internal_state
