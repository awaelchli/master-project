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

        self.pool1 = nn.MaxPool2d(kernel_size=20, stride=20, return_indices=True)

        # LSTM
        lstm_input_size = 32 + 2 * 32 + 1

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

        # Output shape: [h, w]
        return x, y

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
        h, w = input.size(2), input.size(3)

        # Using batch mode to forward sequence
        # Feature shape: [sequence, feat_channels, h, w]
        features = self.feature_extraction(input)
        feat_channels = features.size(1)

        # TODO: is copy of data needed here (or expand suffices?)
        # The 2D with normalized coordinates (2 channels)
        x, y = self.generate_grid(h, w)
        if input.is_cuda:
            x = x.cuda()
            y = y.cuda()

        xgrid = x.unsqueeze(0).repeat(n, feat_channels, 1, 1)
        ygrid = y.unsqueeze(0).repeat(n, feat_channels, 1, 1)

        pool1, ind1 = self.pool1(features)

        x1 = xgrid.view(n, feat_channels, -1)
        y1 = ygrid.view(n, feat_channels, -1)
        i1 = ind1.data.view(n, feat_channels, -1)

        gx1 = torch.gather(x1, 2, i1).view(n, feat_channels, pool1.size(2), pool1.size(3))
        gy1 = torch.gather(y1, 2, i1).view(n, feat_channels, pool1.size(2), pool1.size(3))

        # Gathered x- and y coordinates tensor shape: [sequence, feat_channels, pool1_h, pool1_w]
        assert gx1.size() == gy1.size() == pool1.size()
        num_feat_per_frame = pool1.size(2) * pool1.size(3)


        tgrid = torch.arange(0, n).view(n, 1, 1, 1).repeat(1, 1, pool1.size(2), pool1.size(3))
        if input.is_cuda:
            tgrid = tgrid.cuda()

        # concatenate along channels
        lstm_input_tensor = torch.cat((pool1, gx1, gy1, tgrid), 1)

        # Re-arrange dimensions to: [sequence, ph, pw, channels]
        lstm_input_tensor = lstm_input_tensor.permute(0, 2, 3, 1).contiguous()
        lstm_input_tensor = lstm_input_tensor.view(n * num_feat_per_frame, -1)
        lstm_input_tensor = lstm_input_tensor.unsqueeze(0)
        # LSTM input shape [1, all_features, channels]

        print('Total sequence length: {:d}'.format(lstm_input_tensor.size(1)))
        print('Num features per frame: {:d}'.format(num_feat_per_frame))

        h0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        c0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        if input.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        init = (h0, c0)
        outputs, _ = self.lstm(lstm_input_tensor, init)

        # Not all outputs are needed. Only the last output per frame.
        assert outputs.size(1) == n * num_feat_per_frame

        output_inds = torch.LongTensor(n)
        if input.is_cuda:
            output_inds = output_inds.cuda()
        torch.arange(num_feat_per_frame - 1, n * num_feat_per_frame, num_feat_per_frame, out=output_inds)
        print('outputs: ', outputs.size())
        outputs = outputs.squeeze(0)
        print('outputs: ', outputs.size())
        print('arange:', output_inds)
        outputs = outputs.index_select(0, Variable(output_inds))

        print(outputs)

        assert outputs.size(0) == n
        predictions = self.fc(outputs)

        return predictions

    def train(self, mode=True):
        self.lstm.train(mode)

    def eval(self):
        self.lstm.eval()

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters()) + list(self.feature_extraction.parameters())
        return params