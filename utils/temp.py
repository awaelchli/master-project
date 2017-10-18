from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision.transforms import ToTensor

import GTAV

coll = GTAV.SequenceCollection('/media/Data/adrian/Datasets/GTA V/unzipped/walking/test/data/', transform=ToTensor())

for s in coll:
    samp = GTAV.StridedSampler(s, stride=2)
    batchsamp = BatchSampler(samp,batch_size=2, drop_last=True)
    loader = DataLoader(dataset=s,shuffle=False,batch_sampler=batchsamp, num_workers=0,pin_memory=False)

for i, (x, y, z) in enumerate(loader):
    print(y)
    print(z)