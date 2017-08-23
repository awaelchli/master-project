from ImageNet import FOLDERS, BinaryPoseSequenceGenerator
from torchvision import transforms
import os
import random

traindir = FOLDERS['training']
n = 5
sequence = 10
angle = 50
step = 5
zplane = 0.7

out_folder = os.path.join('out', 'dataset_visualization')
os.makedirs(out_folder)

# Image pre-processing
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
])

# After homography is applied to image
transform2 = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_set = BinaryPoseSequenceGenerator(
    traindir,
    sequence_length=sequence,
    max_angle=angle,
    step_angle=step,
    z_plane=zplane,
    transform1=transform1,
    transform2=transform2)

# Export some examples from the generated dataset
train_set.visualize = out_folder
inds = random.sample(range(len(train_set)), n)
for i in inds:
    _ = train_set[i]
