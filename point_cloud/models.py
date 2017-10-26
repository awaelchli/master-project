import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable


class MultiClass1DTranslationModel(nn.Module):

    def __init__(self, hidden=500, nlayers=3, num_features=1, classes=10, dropout=0.0):
        super(MultiClass1DTranslationModel, self).__init__()

        self.num_classes = classes

        # LSTM
        self.feat_channels = num_features
        lstm_input_size = self.feat_channels + 2

        self.hidden = hidden
        self.nlayers = nlayers

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden,
            num_layers=self.nlayers,
            batch_first=True,
            dropout=dropout
        )

        # The initial state of the LSTM is a learnable parameter
        self.h0 = Parameter(torch.zeros(self.nlayers, 1, self.hidden))
        self.c0 = Parameter(torch.zeros(self.nlayers, 1, self.hidden))

        # The end-of-frame token is learnable
        self.eof_token = Parameter(torch.zeros(1, lstm_input_size))

        # Output transform (classification)
        self.fc = nn.Linear(self.hidden, self.num_classes)

        self.init_weights()

    def forward(self, input):
        # Input shape: [sequence, num_points, 2]
        n = input.size(0)
        num_points = input.size(1)
        feat_channels = self.feat_channels

        # Each keypoint has a random (unique) identity, constant over time
        random_features = Variable(torch.rand(1, num_points, feat_channels).repeat(n, 1, 1)).cuda()

        # Concatenate along channels
        lstm_input_tensor = torch.cat((random_features, input), 2)

        # Re-arrange dimensions to: [sequence * num_points, channels]
        lstm_input_tensor = lstm_input_tensor.view(n * num_points, -1)

        # Split tensor into chunks, add a special end-of-frame token after each chunk
        input_chunks = lstm_input_tensor.chunk(n, 0)
        input_chunks_and_tok = []
        for c in input_chunks:
            input_chunks_and_tok.append(c)
            input_chunks_and_tok.append(self.eof_token)

        lstm_input_tensor = torch.cat(input_chunks_and_tok, 0)

        # Add batch dimension for LSTM
        lstm_input_tensor = lstm_input_tensor.unsqueeze(0)

        # LSTM input shape: [1, all_features + n, channels]
        init = (self.h0, self.c0)
        outputs, _ = self.lstm(lstm_input_tensor, init)

        # Not all outputs are needed. Only the last output per frame (at end-of-frame token)
        assert outputs.size(1) == n * (num_points + 1)
        output_inds = torch.LongTensor(n).cuda()
        torch.arange(num_points, n * (num_points + 1), num_points + 1, out=output_inds)

        # Do not select the first output (canonical coordinate frame)
        output_inds = output_inds[1:]

        outputs = outputs.squeeze(0)
        outputs = outputs.index_select(0, Variable(output_inds))

        assert outputs.size(0) == n - 1
        predictions = self.fc(outputs)

        return predictions

    def train(self, mode=True):
        self.lstm.train(mode)

    def eval(self):
        self.lstm.eval()

    def init_weights(self):
        pass

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters()) + [self.h0, self.c0, self.eof_token]
        return params


class Binary1DTranslationModel(nn.Module):

    def __init__(self, hidden=500, nlayers=3, num_features=1, dropout=0.0):
        super(Binary1DTranslationModel, self).__init__()

        # LSTM
        self.feat_channels = num_features
        lstm_input_size = self.feat_channels + 2

        self.hidden = hidden
        self.nlayers = nlayers

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden,
            num_layers=self.nlayers,
            batch_first=True,
            dropout=dropout
        )

        # The initial state of the LSTM is a learnable parameter
        self.h0 = Parameter(torch.zeros(self.nlayers, 1, self.hidden))
        self.c0 = Parameter(torch.zeros(self.nlayers, 1, self.hidden))

        # The end-of-frame token is learnable
        self.eof_token = Parameter(torch.zeros(1, lstm_input_size))

        # Output transform (binary classification)
        self.fc = nn.Linear(self.hidden, 2)

        self.init_weights()

    def forward(self, input):
        # Input shape: [sequence, num_points, 2]
        n = input.size(0)
        num_points = input.size(1)
        feat_channels = self.feat_channels

        # Each keypoint has a random (unique) identity, constant over time
        random_features = Variable(torch.rand(1, num_points, feat_channels).repeat(n, 1, 1)).cuda()

        # Concatenate along channels
        lstm_input_tensor = torch.cat((random_features, input), 2)

        # Re-arrange dimensions to: [sequence * num_points, channels]
        lstm_input_tensor = lstm_input_tensor.view(n * num_points, -1)

        # Split tensor into chunks, add a special end-of-frame token after each chunk
        input_chunks = lstm_input_tensor.chunk(n, 0)
        input_chunks_and_tok = []
        for c in input_chunks:
            input_chunks_and_tok.append(c)
            input_chunks_and_tok.append(self.eof_token)

        lstm_input_tensor = torch.cat(input_chunks_and_tok, 0)

        # Add batch dimension for LSTM
        lstm_input_tensor = lstm_input_tensor.unsqueeze(0)

        # LSTM input shape: [1, all_features + n, channels]
        init = (self.h0, self.c0)
        outputs, _ = self.lstm(lstm_input_tensor, init)

        # Not all outputs are needed. Only the last output per frame (at end-of-frame token)
        assert outputs.size(1) == n * (num_points + 1)
        output_inds = torch.LongTensor(n).cuda()
        torch.arange(num_points, n * (num_points + 1), num_points + 1, out=output_inds)

        # Do not select the first output (canonical coordinate frame)
        output_inds = output_inds[1:]

        outputs = outputs.squeeze(0)
        outputs = outputs.index_select(0, Variable(output_inds))

        assert outputs.size(0) == n - 1
        predictions = self.fc(outputs)

        return predictions

    def train(self, mode=True):
        self.lstm.train(mode)

    def eval(self):
        self.lstm.eval()

    def init_weights(self):
        pass

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters()) + [self.h0, self.c0, self.eof_token]
        return params