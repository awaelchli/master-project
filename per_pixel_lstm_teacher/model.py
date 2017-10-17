import torch
import torch.nn as nn
from torch.autograd import Variable
import plots


class FullPose7DModel(nn.Module):

    def __init__(self, input_size, hidden=500, nlayers=3, pool_size=50):
        super(FullPose7DModel, self).__init__()

        h, w = input_size[0], input_size[1]

        # Per-pixel feature extraction (padding)
        out_channels = 64
        self.feature_extraction = torch.nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, out_channels, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
        )
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, return_indices=True)

        # LSTM
        lstm_input_size = out_channels + 2

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
        x = torch.linspace(-1, 1, w).view(1, -1).repeat(h, 1).cuda()
        y = torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w).cuda()
        assert x.size() == y.size()

        # Output shape: [h, w]
        return Variable(x), Variable(y)

    def flownet_output_size(self, input_size):
        # 6 for pairwise forward, 3 for single image
        var = Variable(torch.zeros(1, 3, input_size[0], input_size[1]), volatile=True)
        if next(self.feature_extraction.parameters()).is_cuda:
            var = var.cuda()
        out = self.feature_extraction(var)
        return out.size(0), out.size(1), out.size(2), out.size(3)

    def forward(self, frame, prev_pose, state):
        # Frame shape: [channels, h, w]
        # pose shape: [1, 7]
        h, w = frame.size(1), frame.size(2)

        # Using batch mode to forward sequence
        # Feature shape: [1, feat_channels, h, w]
        features = self.feature_extraction(frame.unsqueeze(0))
        feat_channels = features.size(1)

        # The 2D with normalized coordinates (2 channels)
        x, y = self.generate_grid(h, w)

        xgrid = x.view(1, 1, h, w)
        ygrid = y.view(1, 1, h, w)

        # Apply pooling to the first channel only!
        pool_out, ind = self.pool(features[:, 0, :, :].unsqueeze(1))

        # Gather the all channels based on the index of the first channel pooling
        i = Variable(ind.data.view(1, 1, -1).repeat(1, feat_channels, 1))
        f = features.view(1, feat_channels, -1)
        gp = torch.gather(f, 2, i).view(1, feat_channels, pool_out.size(2), pool_out.size(3))

        # Gather the x- and y coordinates from the indices returned by the pooling layer
        x1 = xgrid.view(1, -1)
        y1 = ygrid.view(1, -1)
        i = ind.data.view(1, -1)
        gx1 = torch.gather(x1, 1, i).view(1, 1, pool_out.size(2), pool_out.size(3))
        gy1 = torch.gather(y1, 1, i).view(1, 1, pool_out.size(2), pool_out.size(3))

        # Gathered x- and y coordinates tensor shape: [1, 1, pool1_h, pool1_w]
        assert gx1.size() == gy1.size() == pool_out.size()
        num_feat_in_frame = pool_out.size(2) * pool_out.size(3)

        # concatenate along channels
        lstm_input_tensor = torch.cat((gp, gx1, gy1), 1)
        lstm_input_size = lstm_input_tensor.size(1)

        # Re-arrange dimensions to: [1, ph, pw, channels]
        lstm_input_tensor = lstm_input_tensor.permute(0, 2, 3, 1).contiguous()
        # Present all features in the frame as a list
        lstm_input_tensor = lstm_input_tensor.view(num_feat_in_frame, -1)
        lstm_input_tensor = lstm_input_tensor.unsqueeze(0)

        # LSTM input shape [1, all_features, channels]
        if not state:
            h0 = Variable(torch.zeros(self.nlayers, 1, self.hidden)).cuda()
            c0 = Variable(torch.zeros(self.nlayers, 1, self.hidden)).cuda()
            init = (h0, c0)
        else:
            init = state

        # Forward frames
        _, state = self.lstm(lstm_input_tensor, init)

        # Construct end-of-frame token which contains the previous pose in the first elements.
        # The rest of the entries are zero.
        eof_token = Variable(torch.zeros(1, 1, lstm_input_size)).cuda()
        eof_token[0, 0, :prev_pose.size(1)] = prev_pose

        # Request output with end-of-frame token
        output, state = self.lstm(eof_token, state)
        output = output.squeeze(0)

        # Output shape: [1, hidden_size]
        assert output.size(1) == self.hidden
        prediction = self.fc(output)

        # Shape of prediction: [1, 7]
        return prediction, state

    def train(self, mode=True):
        self.lstm.train(mode)

    def eval(self):
        self.lstm.eval()

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters()) + list(self.feature_extraction.parameters())
        return params