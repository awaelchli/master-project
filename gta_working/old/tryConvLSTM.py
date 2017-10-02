from poseConvLSTM import PoseConvLSTM
from convolution_lstm import ConvLSTM
from torch.autograd import Variable
import torch


clstm = ConvLSTM(3, [16, 16, 16, 32], 3)


out, internal = clstm.forward(Variable(torch.rand(1, 3, 200, 200)))


print('Output', out.size())
for i in internal:
    print('h', i[0].size())
    print('c', i[1].size())

out, internal = clstm.forward(Variable(torch.rand(1, 3, 200, 200)), internal)

print('Output', out.size())
for i in internal:
    print('h', i[0].size())
    print('c', i[1].size())
