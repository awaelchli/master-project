from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torchvision.transforms import ToTensor

from GTAV_old import Sequence

folder = '/home/adrian/Desktop/Grand Theft Auto V 08.12.2017 - 18.04.57.03/'

transform = ToTensor()
s1 = Sequence(folder, transform)

d = SequentialSampler(s1)
mysampler = BatchSampler(d, 10, drop_last=True)


l1 = DataLoader(s1, batch_size=10, sampler=None, drop_last=True)

#for i, (a, b, c) in enumerate(l1):
    #print(a)

