import torch
import torch.nn as nn
from torch.autograd import Variable
import plots


class FullPose7DModel(nn.Module):

    def __init__(self, input_size, hidden=500, nlayers=3):
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
        pool_size = 50
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, return_indices=True)

        # LSTM
        lstm_input_size = out_channels + 2 + 1

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
        x = torch.linspace(-1, 1, w).view(1, -1).repeat(h, 1)
        y = torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w)
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

    def forward(self, input, return_keypoints=False):
        # Input shape: [sequence, channels, h, w]
        n = input.size(0)
        h, w = input.size(2), input.size(3)

        # Using batch mode to forward sequence
        # Feature shape: [sequence, feat_channels, h, w]
        print('Feature extraction')
        features = self.feature_extraction(input)
        feat_channels = features.size(1)

        print('Feature selection')

        # TODO: is copy of data needed here (or expand suffices?)
        # The 2D with normalized coordinates (2 channels)
        x, y = self.generate_grid(h, w)
        if input.is_cuda:
            x = x.cuda()
            y = y.cuda()

        xgrid = x.repeat(n, 1, 1, 1)
        ygrid = y.repeat(n, 1, 1, 1)

        # Apply pooling to the first channel only!
        pool_out, ind = self.pool(features[:, 0, :, :].unsqueeze(1))

        # Gather the all channels based on the index of the first channel pooling
        i = Variable(ind.data.view(n, 1, -1).repeat(1, feat_channels, 1))
        f = features.view(n, feat_channels, -1)
        gp = torch.gather(f, 2, i).view(n, feat_channels, pool_out.size(2), pool_out.size(3))

        # Gather the x- and y coordinates from the indices returned by the pooling layer
        x1 = xgrid.view(n, -1)
        y1 = ygrid.view(n, -1)
        i = ind.data.view(n, -1)
        gx1 = torch.gather(x1, 1, i).view(n, 1, pool_out.size(2), pool_out.size(3))
        gy1 = torch.gather(y1, 1, i).view(n, 1, pool_out.size(2), pool_out.size(3))

        # Gathered x- and y coordinates tensor shape: [sequence, 1, pool1_h, pool1_w]
        assert gx1.size() == gy1.size() == pool_out.size()
        num_feat_per_frame = pool_out.size(2) * pool_out.size(3)

        tgrid = Variable(torch.arange(0, n).view(n, 1, 1, 1).repeat(1, 1, pool_out.size(2), pool_out.size(3)))
        if input.is_cuda:
            tgrid = tgrid.cuda()

        # concatenate along channels
        lstm_input_tensor = torch.cat((gp, gx1, gy1, tgrid), 1)

        # Re-arrange dimensions to: [sequence, ph, pw, channels]
        lstm_input_tensor = lstm_input_tensor.permute(0, 2, 3, 1).contiguous()
        lstm_input_tensor = lstm_input_tensor.view(n * num_feat_per_frame, -1)
        lstm_input_tensor = lstm_input_tensor.unsqueeze(0)
        # LSTM input shape [1, all_features, channels]

        #print('Total sequence length: {:d}'.format(lstm_input_tensor.size(1)))
        #print('Num features per frame: {:d}'.format(num_feat_per_frame))

        h0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        c0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        if input.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        init = (h0, c0)

        #print('input lstm', lstm_input_tensor)
        print('Feature tracking (lstm)')
        outputs, _ = self.lstm(lstm_input_tensor, init)

        # Not all outputs are needed. Only the last output per frame.
        assert outputs.size(1) == n * num_feat_per_frame

        output_inds = torch.LongTensor(n)
        if input.is_cuda:
            output_inds = output_inds.cuda()
        torch.arange(num_feat_per_frame - 1, n * num_feat_per_frame, num_feat_per_frame, out=output_inds)

        # Do not select the first output (canonical coordinate frame)
        output_inds = output_inds[1:]

        #print('outputs: ', outputs.size())
        outputs = outputs.squeeze(0)
        #print('outputs: ', outputs.size())
        #print('arange:', output_inds)
        outputs = outputs.index_select(0, Variable(output_inds))

        #print(outputs)

        assert outputs.size(0) == n - 1
        predictions = self.fc(outputs)

        if return_keypoints:
            keypoints = torch.cat((gx1, gy1), 1).data
            return predictions, keypoints
        else:
            return predictions

    def train(self, mode=True):
        self.lstm.train(mode)

    def eval(self):
        self.lstm.eval()

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters()) + list(self.feature_extraction.parameters())
        return params