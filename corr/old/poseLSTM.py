import torch
import torch.nn as nn
from torch.autograd import Variable


class PoseLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(PoseLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_size, 6)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()

    def forward(self, input):
        # Set initial states
        states = self.init_states(input.size(0))

        output, hidden = self.lstm(input, states)

        # Only works with batch size 1
        # Using batchmode of fc layer to transform entire sequence
        pose_sequence = self.fc(output.squeeze()).unsqueeze(0)

        return pose_sequence, hidden

    def init_states(self, batch_size):
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        return h0, c0

    def get_parameters(self):
        return list(self.lstm.parameters()) + list(self.fc.parameters())