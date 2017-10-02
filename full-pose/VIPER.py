import glob
import os
from math import radians
from os import path
from zipfile import ZipFile

import matplotlib.pyplot as plt
plt.switch_backend('agg') # For machines without display (e.g. cluster)
import numpy as np
import torch
from PIL import Image
from scipy import interpolate
from torch.utils.data import Dataset
from transforms3d.euler import euler2quat
from transforms3d.quaternions import rotate_vector, qinverse, qmult
from torch.utils.data import ConcatDataset
import random

FOLDERS = {
    'training': {
        'data': '',
        'pose': '',
    },
    'testing': {
        'data': '',
        'pose': '',
    },
    'validation': {
        'data': '',
        'pose': '',
    },
}

IMAGE_EXTENSION = 'jpg'
POSE_FILE_EXTENSION = 'txt'


class Sequence(Dataset):

    def __init__(self, data_folder, pose_folder, sequence_length, transform=None, max_size=None, return_filename=False):
        self.sequence_length = sequence_length
        self.transform = transform
        self.index = build_subsequence_index(data_folder, pose_folder, sequence_length)
        self.return_filename = return_filename

        if max_size and max_size < len(self):
            self.index = self.index[:max_size]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        filenames, poses = self.index[idx]
        images = [self.load_image(file).unsqueeze(0) for file in filenames]
        image_sequence = torch.cat(images, dim=0)

        # Convert raw pose (from text file) to 7D pose vectors (translation + quaternion)
        positions = get_positions(poses)
        quaternions = get_quaternions(poses)
        rel_positions, rel_quaternions = to_relative_pose(positions, quaternions)
        pose_vectors = encode_poses(rel_positions, rel_quaternions)
        pose_sequence = torch.from_numpy(pose_vectors).float()

        if self.return_filename:
            return image_sequence, pose_sequence, filenames
        else:
            return image_sequence, pose_sequence

    def load_image(self, filename):
        image = Image.open(filename).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image