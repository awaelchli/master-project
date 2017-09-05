from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import shutil
from GTAV import Subsequence, FOLDERS

traindir = FOLDERS['standing']['training']
image_size = 256
sequence = 25
num_sequences = 5
output_folder = 'out/dataset_visualization'

if os.path.isdir(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)


# Image pre-processing
transform = transforms.Compose([
    transforms.Scale(image_size),
    transforms.ToTensor(),
])

train_set = Subsequence(
    data_folder=traindir['data'],
    pose_folder=traindir['pose'],
    sequence_length=sequence,
    transform=transform,
    max_size=None,
)

dataloader = DataLoader(
    train_set,
    batch_size=1,
    pin_memory=False,
    shuffle=False,
    num_workers=1)

g = enumerate(dataloader)
for i in range(num_sequences):
    idx, (images, poses) = next(g)

    images.squeeze_(0)
    poses.squeeze_(0)

    for i, image in enumerate(images):
        name = '{}-{}.png'.format(idx, i)
        save_image(image, os.path.join(output_folder, name))