import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import plots


class FullPose7DModel(nn.Module):

    def __init__(self, hidden=500, nlayers=3):
        super(FullPose7DModel, self).__init__()

        # Per-pixel feature extraction (padding)
        self.feat_channels = 100

        # LSTM
        lstm_input_size = self.feat_channels + 2

        self.hidden = hidden
        self.nlayers = nlayers

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden,
            num_layers=self.nlayers,
            batch_first=True,
            #dropout=0.3
        )

        # The initial state of the LSTM is a learnable parameter
        self.h0 = Parameter(torch.zeros(self.nlayers, 1, self.hidden))
        self.c0 = Parameter(torch.zeros(self.nlayers, 1, self.hidden))
        self.eof_token = Variable(torch.zeros(1, lstm_input_size)).cuda()

        # Output transform
        self.fc = nn.Linear(self.hidden, 2)

        self.init_weights()

    def forward(self, input):
        # Input shape: [sequence, num_points, 2]

        #print('Input', input)

        n = input.size(0)

        feat_channels = self.feat_channels
        num_points = input.size(1)

        # tgrid = Variable(torch.arange(0, n).view(n, 1, 1).repeat(1, num_points, 1))
        # if input.is_cuda:
        #     tgrid = tgrid.cuda()

        # Each keypoint has a random (unique) identity, constant over time
        random_features = Variable(torch.rand(1, num_points, feat_channels).repeat(n, 1, 1))
        if input.is_cuda:
            random_features = random_features.cuda()

        # Concatenate along channels
        lstm_input_tensor = torch.cat((random_features, input), 2)

        # Re-arrange dimensions to: [sequence * num_points, channels]
        lstm_input_tensor = lstm_input_tensor.view(n * num_points, -1)

        # Split tensor into chunks, add a special end-of-frame token after each chunk
        #token = Variable(torch.zeros(1, lstm_input_tensor.size(1)))
        #if input.is_cuda:
        #    token = token.cuda()

        input_chunks = lstm_input_tensor.chunk(n, 0)
        input_chunks_and_tok = []
        for c in input_chunks:
            input_chunks_and_tok.append(c)
            input_chunks_and_tok.append(self.eof_token)

        lstm_input_tensor = torch.cat(input_chunks_and_tok, 0)

        # Add batch dimension
        lstm_input_tensor = lstm_input_tensor.unsqueeze(0)

        #print('LSTM input', lstm_input_tensor)

        # LSTM input shape [1, all_features + n, channels]
        #print('Total sequence length: {:d}'.format(lstm_input_tensor.size(1)))
        #print('Num features per frame: {:d}'.format(num_points))

        #h0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        #c0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        #if input.is_cuda:
            #h0 = h0.cuda()
            #c0 = c0.cuda()

        init = (self.h0, self.c0)

        #print('input lstm', lstm_input_tensor)
        #print('Feature tracking (lstm)')
        outputs, _ = self.lstm(lstm_input_tensor, init)

        # Not all outputs are needed. Only the last output per frame (at end-of-frame token)
        assert outputs.size(1) == n * (num_points + 1)
        output_inds = torch.LongTensor(n)
        if input.is_cuda:
            output_inds = output_inds.cuda()
        torch.arange(num_points, n * (num_points + 1), num_points + 1, out=output_inds)

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

        #print(self.eof_token.sum().data[0])

        return predictions

    def train(self, mode=True):
        self.lstm.train(mode)

    def eval(self):
        self.lstm.eval()

    def init_weights(self):
        #self.fc.weight.data.uniform_(-0.0, 0.0)
        #self.fc.bias.data.fill_(0)
        pass

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters()) + [self.h0, self.c0]
        return params