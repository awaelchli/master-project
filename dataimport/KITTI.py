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


class KITTISequence(Dataset):
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
        self.pose_file = path.join(self.pose_dir, string_number, '.txt')

    def __len__(self):
        return len(self.__get_file_list())

    def __getitem__(self, idx):
        img_name = self.__get_file_list()[idx]
        image = io.imread(img_name)

        self.read_poses()

        landmarks = landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
        landmarks = landmarks.reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_file_list(self):
        return glob.glob(path.join(self.images_dir, '*.png'))

    def read_poses(self):
        with open(self.pose_file, 'r') as f:
            lines = f.readlines()



def load_sequence(path_to_sequences, number, eye = 0):
    string_number ='{:02d}'.format(number)
    p = path.join(path_to_sequences, string_number, 'image_{}'.format(eye))

    print(p)


