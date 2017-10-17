import glob
import os
from math import radians
from os import path

import numpy as np
import torch
from PIL import Image
from scipy import interpolate
from torch.utils.data import Dataset
from transforms3d.euler import euler2quat


FOLDERS = {
    'walking': {
        'training': {
            'data': '../data/GTA V/walking/train/data/',
            'pose': '../data/GTA V/walking/train/poses/'
        },
        'validation': {
            'data': '../data/GTA V/walking/test/data/',
            'pose': '../data/GTA V/walking/test/poses/',
        },
        'test': {
            'data': '../data/GTA V/walking/test/data/',
            'pose': '../data/GTA V/walking/test/poses/',
        }
    },
    'standing': {
        'training': {
            'data': '../data/GTA V/standing/train/data/',
            'pose': '../data/GTA V/standing/train/poses/'
        },
        'validation': {
            'data': '../data/GTA V/standing/test/data/',
            'pose': '../data/GTA V/standing/test/poses/',
        },
        'test': {
            'data': '../data/GTA V/standing/test/data/',
            'pose': '../data/GTA V/standing/test/poses/',
        }
    },
}

IMAGE_EXTENSION = 'jpg'
POSE_FILE_EXTENSION = 'txt'


class Sequence(Dataset):

    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

        # The folder should contain exactly one pose file
        text_files = glob.glob(os.path.join(folder, '*.{}'.format(POSE_FILE_EXTENSION)))
        assert len(text_files) == 1
        self.pose_file = text_files[0]

        self.filenames, self.poses = read_filenames_and_poses(self.folder, self.pose_file)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = self.load_image(filename)
        pose = self.poses[idx]

        # Convert raw pose (from text file) to 7D pose vector (translation + quaternion)
        position = get_position(pose)
        quaternion = get_quaternion(pose)
        pose_vector = torch.from_numpy(encode_pose(position, quaternion)).float()

        return image, pose_vector, filename

    def load_image(self, filename):
        image = Image.open(filename).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def read_filenames_and_poses(sequence_dir, pose_file):
    all_times, all_poses = read_from_text_file(pose_file)
    interpolator = interpolate.interp1d(all_times, all_poses, kind='nearest', axis=0, copy=True,
                                        fill_value='extrapolate', assume_sorted=True)

    # Sort filenames according to time given in filename
    filenames = [path.join(sequence_dir, f)
                 for f in glob.glob(os.path.join(sequence_dir, '*.{}'.format(IMAGE_EXTENSION)))]

    time_from_filename = lambda x: int(path.splitext(path.basename(x))[0])
    filenames.sort(key=time_from_filename)

    # Interpolate at times given by filename
    query_times = [time_from_filename(f) for f in filenames]
    poses_interpolated = [interpolator(t) for t in query_times]

    return filenames, np.array(poses_interpolated)


def get_position(pose_array):
    return pose_array[0:3]


def get_quaternion(pose_array):
    angles = pose_array[3:6]

    # Order saved in file: pitch roll yaw
    # Apply rotations in order: yaw, pitch, roll
    quaternion = euler2quat(radians(angles[2]), radians(angles[0]), radians(angles[1]), 'rzxy')
    return np.array(quaternion)


def encode_pose(position, quaternion):
    return np.concatenate((position, quaternion))


def read_from_text_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = [s.split() for s in lines]

    times = np.array([float(line[1]) for line in lines])
    poses = np.array([[float(x) for x in line[2:8]] for line in lines])

    return times, poses

