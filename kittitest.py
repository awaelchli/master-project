from dataimport import KITTI

root_dir = '~/data/KITTI odometry/grayscale/sequences'
pose_dir = '~/data/KITTI odometry/poses/'

sequence = KITTI.Sequence(root_dir, pose_dir, sequence_number = 2)

t = sequence.__getitem__(2)

print(t)

dataloader = KITTI.DataLoader(sequence, batch_size = 4, shuffle = True, num_workers = 4)

# for i, sample in enumerate(dataloader):
#     print(sample)


# Determine output size of vgg given KITTI images as input
from torchvision import models
from torch.autograd import Variable

vgg = models.vgg11(pretrained=True).features
print(vgg)

for i, sample in enumerate(dataloader):
    input = sample['image']
    print(input.size())

    input_var = Variable(input)
    #out = vgg(input_var)
    #print(out.size())