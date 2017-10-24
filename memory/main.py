import argparse
import sys

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=10)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--look_back', type=int, default=5)
parser.add_argument('--bptt', type=int, default=50)
parser.add_argument('--sequence', type=int, default=100000)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()


class Model(nn.Module):
    """ The model consists of two parts: An LSTM and a fully-connected layer.
        At each time step, the model reads one input symbol and outputs a symbol from the past.
        It is trained to remember the input at a fixed number of time steps back in time.
    """
    def __init__(self, hidden_size, num_layers):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            num_layers=num_layers,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 2)
        self.init_weights()

    def forward(self, input, state=None):
        if not state:
            state = self.init_hidden_state()

        output, state = self.lstm(input, state)
        output = self.fc(output.view(-1, self.lstm.hidden_size))
        return output, state

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.parameters())
        return params

    def init_hidden_state(self):
        state = (Variable(torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size)).cuda(),
                 Variable(torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size)).cuda())
        return state

    def init_weights(self):
        pass


model = Model(args.hidden, args.layers)
model.cuda()

optimizer = torch.optim.Adam(model.get_parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
criterion.cuda()


def train():

    dataset = get_dataset(args.sequence, args.bptt, args.look_back)
    state = None

    model.train()
    for i, (batch, targets) in enumerate(dataset):
        optimizer.zero_grad()

        output, state = model.forward(batch.unsqueeze(0), state)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        state = repack_state(state)

        grad_norm = gradient_norm()
        accuracy = torch.sum(classify(output.data) == targets.data) / args.bptt
        print('[{:d}/{:d}] Loss: {:.4f}, Accuracy: {:.4f}, Grad. Norm: {:.4f}'.format(
            i + 1, args.sequence // args.bptt, loss.data[0], accuracy, grad_norm)
        )


def test():

    dataset = get_dataset(args.sequence, args.bptt, args.look_back)
    state = None
    avg_accuracy = AverageMeter()

    model.eval()
    for i, (batch, targets) in enumerate(dataset):

        batch.volatile = True
        output, state = model.forward(batch.unsqueeze(0), state)

        accuracy = torch.sum(classify(output.data) == targets.data) / args.bptt
        avg_accuracy.update(accuracy)

    print('Accuracy on testset: {:.4f}'.format(avg_accuracy.average))


def get_dataset(sequence_length, bptt, look_back):
    sequence = get_sequence(sequence_length + look_back)

    for i in range(look_back, sequence_length - bptt, bptt):

        batch = sequence[i: i + bptt].view(-1, 1)
        targets = sequence[i - look_back: i + bptt - look_back].long()

        yield Variable(batch), Variable(targets)


def get_sequence(length):
    sequence = torch.rand(length) < 0.5
    return sequence.float().cuda()


def repack_state(state):
    return Variable(state[0].data), Variable(state[1].data)


def get_batch(n):
    sequence = torch.rand(1, n, 1) < 0.5
    return Variable(sequence).float().cuda()


def classify(output):
    return torch.max(output, 1)[1]


def gradient_norm():
    params = model.get_parameters()
    parameters = list(filter(lambda p: p.grad is not None, params))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2

    total_norm = total_norm ** (1. / 2)
    return total_norm


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value):
        self.sum += value
        self.count += 1


try:
    print()
    print('Training...')
    train()
except KeyboardInterrupt:
    print('Training stopped.')
    print('Testing...')
    test()
