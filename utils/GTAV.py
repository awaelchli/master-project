import glob
import os
from math import radians, ceil
from os import path

import numpy as np
import torch
from PIL import Image
from scipy import interpolate
from torch.utils.data import Dataset
from transforms3d.euler import euler2quat
from pose_transforms import relative_to_first_pose
from torch.utils.data.sampler import SequentialSampler

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


class StridedSampler(SequentialSampler):
    """Samples elements sequentially, always in the same order, but with a given stride (skipping elements).

    Arguments:
        data_source (Dataset): dataset to sample from
        stride (int): stride to apply when sampling
        max_size (int): clips the dataset such that there are at most max_size samples
    """

    def __init__(self, data_source, stride=1, max_size=None):
        super(StridedSampler, self).__init__(data_source)
        self.stride = stride
        self.max_size = max_size
        self.indices = self.get_indices()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def get_indices(self):
        indices = range(0, len(self.data_source), self.stride)
        if self.max_size:
            indices = indices[:self.max_size]
        return indices


class SequenceCollection(object):
    """ A collection of GTA sequences. Each sequence in the iterator is a Dataset object. """

    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.folders = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)]

    def __len__(self):
        return len(os.listdir(self.root_folder))

    def __iter__(self):
        for folder in self.folders:
            sequence = Sequence(
                folder=folder,
                transform=self.transform
            )
            yield sequence


class Sequence(Dataset):

    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

        self.pose_file = find_pose_file(folder)
        self.filenames, self.poses = read_filenames_and_poses(self.folder, self.pose_file)

        # Convert raw poses (from text file) to 7D pose vectors (translation + quaternion)
        positions = get_positions(self.poses)
        quaternions = get_quaternions(self.poses)

        positions, quaternions = relative_to_first_pose(positions, quaternions)
        self.poses = torch.from_numpy(np.concatenate((positions, quaternions), 1)).float()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = self.load_image(filename)
        pose = self.poses[idx]

        return image, pose, filename

    def load_image(self, filename):
        image = Image.open(filename).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def find_pose_file(folder):
    # The folder should contain exactly one pose file
    text_files = glob.glob(os.path.join(folder, '*.{}'.format(POSE_FILE_EXTENSION)))
    assert len(text_files) == 1, 'No pose file found in sequence folder {}'.format(folder)
    return text_files[0]


def read_filenames_and_poses(sequence_dir, pose_file):
    all_times, all_poses = read_from_text_file(pose_file)
    interpolator = interpolate.interp1d(all_times, all_poses, kind='nearest', axis=0, copy=True,
                                        fill_value='extrapolate', assume_sorted=True)

    # Sort filenames according to time given in filename
    filenames = glob.glob(os.path.join(sequence_dir, '*.{}'.format(IMAGE_EXTENSION)))

    time_from_filename = lambda x: int(path.splitext(path.basename(x))[0])
    filenames.sort(key=time_from_filename)

    # Interpolate at times given by filename
    query_times = [time_from_filename(f) for f in filenames]
    poses_interpolated = [interpolator(t) for t in query_times]

    return filenames, np.array(poses_interpolated)


def get_positions(pose_array):
    return pose_array[:, 0:3]


def euler_to_quaternion(angles):
    # Order saved in file: pitch roll yaw
    # Apply rotations in order: yaw, pitch, roll
    quaternion = euler2quat(radians(angles[2]), radians(angles[0]), radians(angles[1]), 'rzxy')
    return np.array(quaternion)


def get_quaternions(pose_array):
    quaternions = [euler_to_quaternion(row[3:6]) for row in pose_array]
    return np.array(quaternions)


def encode_pose(position, quaternion):
    return np.concatenate((position, quaternion))


def read_from_text_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = [s.split() for s in lines]

    times = np.array([float(line[1]) for line in lines])
    poses = np.array([[float(x) for x in line[2:8]] for line in lines])

    return times, poses

