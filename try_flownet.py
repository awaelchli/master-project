from flownet.models.FlowNetS import flownets
import torch
from torch.autograd import Variable


model = flownets('../data/Pretrained Models/flownets_pytorch.pth')

model2 = torch.nn.Sequential(
    model.conv1,
    model.conv2,
    model.conv3,
    model.conv3_1,
    model.conv4,
    model.conv4_1,
    model.conv5,
    model.conv5_1,
    model.conv6,
    model.conv6_1
)

out = model2(Variable(torch.zeros(1, 6, 200, 200)))

print(out.size())


