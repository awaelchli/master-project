import os.path as path
import os
import glob
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Sequence(Dataset):
    """KITTI Dataset"""

    def __init__(self, root_dir, pose_dir, sequence_number, eye = 0):
        """
        :param root_dir: Path to 'sequences' folder
        :param pose_dir: Path to 'poses' folder
        :param sequence_number: Integer number of the sequence
        :param eye: Left eye (0) or right eye (1) of stereo camera
        """

        self.root_dir = root_dir
        self.pose_dir = pose_dir
        self.sequence_number = sequence_number

        string_number = '{:02d}'.format(sequence_number)
        self.images_dir = path.join(root_dir, string_number, 'image_{}'.format(eye))
        self.pose_file = path.join(self.pose_dir, '{}.txt'.format(string_number))
        self.poses = self.read_poses()

    def __len__(self):
        return len(self.get_file_list())

    def __getitem__(self, idx):
        img_name = self.get_file_list()[idx]
        gray_image = torch.from_numpy(io.imread(img_name)).float()

        # In case the image is grayscale
        height = gray_image.size()[0]
        width = gray_image.size()[1]
        gray_image.resize_(1, height, width)
        image = gray_image.expand(3, height, width)

        sample = {'image': image, 'pose': self.poses[idx]}
        return sample

    def get_file_list(self):
        return glob.glob(path.join(self.images_dir, '*.png'))

    def read_poses(self):
        with open(self.pose_file, 'r') as f:
            lines = f.readlines()

        poses = []
        for line in lines:
            vector = torch.FloatTensor([float(s) for s in line.split()])
            poses.append(vector.view(3, 4))

        return poses
