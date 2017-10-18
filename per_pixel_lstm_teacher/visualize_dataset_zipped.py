from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import shutil
from GTAV_old import ZippedSequence, FOLDERS, Loop
import time

sequence = 60
num_sequences = 15
output_folder = 'out/dataset_visualization_zipped'

if os.path.isdir(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)


# Image pre-processing
transform = transforms.Compose([
    transforms.Scale(320),
    transforms.CenterCrop((320, 448)),
    transforms.ToTensor(),
])

# Sequence transform
seq_transform = transforms.Compose([
    #RandomSequenceReversal(),
    Loop(40, 60),
])

train_set = ZippedSequence(
    '../data/GTA V/walking/test/Grand Theft Auto V 08.13.2017 - 19.53.40.04.zip',
    sequence,
    image_transform=transform,
    sequence_transform=seq_transform,
    return_filename=False,
)

dataloader = DataLoader(
    train_set,
    batch_size=1,
    pin_memory=False,
    shuffle=False,
    num_workers=1)

g = enumerate(dataloader)

# Benchmark
start = time.time()
for i in range(num_sequences):
    next(g)

elapsed = time.time() - start
print('Load benchmark: {:.4f} seconds'.format(elapsed))

g = enumerate(dataloader)
for i in range(num_sequences):
    idx, (images, poses) = next(g)

    images.squeeze_(0)
    poses.squeeze_(0)

    for i, image in enumerate(images):
        name = '{}-{}.png'.format(idx, i)
        save_image(image, os.path.join(output_folder, name))