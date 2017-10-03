import torch
from torch.autograd import Variable
from cuda_functional import SRU, SRUCell

# input has length 20, batch size 32 and dimension 128
x = Variable(torch.FloatTensor(20, 32, 128).cuda())

input_size, hidden_size = 128, 128

rnn = SRU(
    input_size, hidden_size,
    num_layers = 2,          # number of stacking RNN layers
    dropout = 0.0,           # dropout applied between RNN layers
    rnn_dropout = 0.0,       # variational dropout applied on linear transformation
    use_tanh = 1,            # use tanh or identity activation
    bidirectional = False    # bidirectional RNN ?
).cuda()

output, hidden = rnn(x)      # forward pass

# output is (length, batch size, hidden size * number of directions)
# hidden is (layers, batch size, hidden size * number of directions)