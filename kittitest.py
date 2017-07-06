from vggRNN import VGGRNN
from dataimport import KITTI
from torchvision import models
from torch.autograd import Variable
import torch

root_dir = '../data/KITTI odometry/grayscale/sequences'
pose_dir = '../data/KITTI odometry/poses/'

sequence = KITTI.Sequence(root_dir, pose_dir, sequence_number = 2)

#t = sequence.__getitem__(2)

#print(t)

dataloader = KITTI.DataLoader(sequence, batch_size = 4, shuffle = True, num_workers = 4)

# for i, sample in enumerate(dataloader):
#     print(sample)


# Determine output size of vgg given KITTI images as input


#vgg = models.vgg11(pretrained=True).features
#vggC = models.vgg11(pretrained=True).classifier
#print(vgg)
#print(vggC)

vggRNN = VGGRNN(nhidden=4096, nlayers=2)
if torch.cuda.is_available():
    vggRNN.cuda()

hidden = vggRNN.init_hidden(4)

for i, sample in enumerate(dataloader):
    input = sample['image'].cuda()
    print(input.size())

    input_var = Variable(input)
    #out = vgg(input_var)
    #print(out.size())

    out, hidden = vggRNN(input, hidden)
    print(out.size())
    print(hidden.size())

