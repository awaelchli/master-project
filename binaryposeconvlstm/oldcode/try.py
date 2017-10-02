import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataimport.ImageNet import PoseImageNet


datadir = '../../data/ImageNet/ILSVRC2012'

traindir = os.path.join(datadir, 'train')
valdir = os.path.join(datadir, 'train')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    PoseImageNet(traindir, transform=transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize,
    ])),
    batch_size=1, shuffle=True,
    num_workers=4, pin_memory=False)

val_loader = torch.utils.data.DataLoader(
    PoseImageNet(valdir, transform=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #normalize,
    ])),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=False)


for i, sample in enumerate(train_loader):
    print(sample)