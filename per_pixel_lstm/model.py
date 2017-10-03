import torch
import torch.nn as nn
from torch.autograd import Variable


class FullPose7DModel(nn.Module):

    def __init__(self, input_size, hidden=500, nlayers=3):
        super(FullPose7DModel, self).__init__()

        h, w = input_size[0], input_size[1]

        # Per-pixel feature extraction (padding)
        self.feature_extraction = torch.nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

        # The 2D with normalized coordinates (2 channels)
        self.grid = self.generate_grid(h, w)

        # LSTM

        # TODO: reduce with pooling!
        lstm_input_size = h * w * (32 + 2 + 1)

        self.hidden = hidden
        self.nlayers = nlayers

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden,
            num_layers=self.nlayers,
            batch_first=True,
            #dropout=0.3
        )

        # Output transform
        self.fc = nn.Linear(self.hidden, 7)

    def generate_grid(self, h, w):
        x = torch.arange(0, w).view(1, -1).repeat(h, 1).unsqueeze(0)
        y = torch.arange(0, h).view(-1, 1).repeat(1, w).unsqueeze(0)
        assert x.size() == y.size()

        # Normalize
        x /= w - 1
        y /= h - 1

        # Output shape: [2, h, w]
        grid = torch.cat((x, y), 0)
        return grid

    def flownet_output_size(self, input_size):
        # 6 for pairwise forward, 3 for single image
        var = Variable(torch.zeros(1, 3, input_size[0], input_size[1]), volatile=True)
        if next(self.feature_extraction.parameters()).is_cuda:
            var = var.cuda()
        out = self.feature_extraction(var)
        return out.size(0), out.size(1), out.size(2), out.size(3)

    def forward(self, input):
        # Input shape: [sequence, channels, h, w]
        n = input.size(0)

        # Using batch mode to forward sequence
        features = self.feature_extraction(input)

        # Add 2D coordinates
        grid = self.grid.copy().unsqueeze(0).repeat(n, 1, 1, 1)

        # TODO: continue
        nn.MaxPool2d(kernel_size=9, stride=9, return_indices=True)

        features = torch.cat((features, grid), 1) # concatenate along channels



        h0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        c0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        if input.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        init = (h0, c0)

        if pairwise:
            outputs, _ = self.lstm(pairs.view(1, n - 1, -1), init)
        else:
            outputs, _ = self.lstm(pairs.view(1, n, -1), init)

        predictions = self.fc(outputs.squeeze(0))

        return predictions

    def train(self, mode=True):
        self.lstm.train(mode)

    def eval(self):
        self.lstm.eval()

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters())
        if not self.fix_flownet:
            params = list(self.feature_extraction.parameters()) + params
        return params