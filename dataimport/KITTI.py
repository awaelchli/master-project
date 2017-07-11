import os.path as path
import glob
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class Sequence(Dataset):
    """KITTI Dataset"""

    def __init__(self, root_dir, pose_dir, sequence_number, transform = None, eye = 0):
        """
        :param root_dir: Path to 'sequences' folder
        :param pose_dir: Path to 'poses' folder
        :param sequence_number: Integer number of the sequence
        :param eye: Left eye (0) or right eye (1) of stereo camera
        """

        self.root_dir = root_dir
        self.pose_dir = pose_dir
        self.sequence_number = sequence_number
        self.transform = transform

        string_number = '{:02d}'.format(sequence_number)
        self.images_dir = path.join(root_dir, string_number, 'image_{}'.format(eye))
        self.pose_file = path.join(self.pose_dir, '{}.txt'.format(string_number))
        self.poses = read_6D_poses(self.pose_file)

    def __len__(self):
        return len(self.get_file_list())

    def __getitem__(self, idx):
        img_name = self.get_file_list()[idx]

        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        sample = (image, self.poses[idx])
        return sample

    def get_file_list(self):
        return glob.glob(path.join(self.images_dir, '*.png'))


def read_6D_poses(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()

    poses = []
    for line in lines:
        vector = torch.FloatTensor([float(s) for s in line.split()]).view(1, 6)
        poses.append(vector)

    return poses


