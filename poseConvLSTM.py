import torch
import torch.nn as nn
from convolution_lstm import ConvLSTM


class PoseConvLSTM(nn.Module):

    def __init__(self, input_size, input_channels, hidden_channels, kernel_size):
        super(PoseConvLSTM, self).__init__()

        self.input_size = input_size
        self.clstm = ConvLSTM(input_channels, hidden_channels, kernel_size, bias=True)

        # The output size of the last cell defines the input size of the linear layer
        last_hidden_size = input_size * hidden_channels[-1]
        self.fc = nn.Linear(last_hidden_size, 4)

        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()

    def forward(self, input):
        # Input format: [sequence_length, channels, h, w]
        hidden = None
        outputs = []
        for i in range(input.size(0)):
            output, hidden = self.clstm.forward(input[i, :, :, :], hidden)
            outputs.append(output.view(1, -1))

        # Using batchmode of fc layer to transform entire sequence
        outputs = torch.cat(outputs, 0)
        pose_sequence = self.fc(outputs)

        return pose_sequence, hidden

    def get_parameters(self):
        return list(self.clstm.parameters()) + list(self.fc.parameters())