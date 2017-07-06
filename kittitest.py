from vggLSTM import VGGLSTM
from dataimport import KITTI
from torchvision import models
from torch.autograd import Variable
import torch

root_dir = '../data/KITTI odometry/grayscale/sequences'
pose_dir = '../data/KITTI odometry/poses/'

sequence = KITTI.Sequence(root_dir, pose_dir, sequence_number = 2)

#t = sequence.__getitem__(2)

#print(t)

dataloader = KITTI.DataLoader(sequence, batch_size = 1, shuffle = True, num_workers = 4)

# for i, sample in enumerate(dataloader):
#     print(sample)


# Determine output size of vgg given KITTI images as input


#vgg = models.vgg11(pretrained=True).features
#vggC = models.vgg11(pretrained=True).classifier
#print(vgg)
#print(vggC)

# VGG without classifier
# Input tensor dimensions: [batch, channels, height, width]
# Output tensor dimensions: [batch, 512, 11, 38]
vgg = models.vgg19(pretrained=True).features


# Transform all images in the sequence to features for the LSTM
sequence = []
for i, sample in enumerate(dataloader):
    print('looping')
    input = Variable(sample['image'])

    output = vgg(input)
    # Reshape output to size [batch, 1, features]
    sequence.append(output.view(output.size(0), 1, output.size(1) * output.size(2) * output.size(3)))

# Concatenate sequence to one tensor of dimensions [batch, sequence, features]
input_lstm = torch.cat(tuple(sequence), 1)
print('Input sequence to LSTM', input_lstm.size())


# Feed the sequence to LSTM
lstm = VGGLSTM(input_size=214016, nhidden=4096, nlayers=2)

hidden = lstm.init_hidden(1)
input_lstm = Variable(input_lstm)

out, hidden = lstm(input_lstm, hidden)
print(out.size())
print(hidden.size())


