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


class Subsequence(Dataset):

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


def build_subsequence_index(root_folder, ground_truth_folder, sequence_length):
    # Makes small chunks out of the large sequences
    index = build_folder_index(root_folder, ground_truth_folder)
    subsequences = []
    for filenames, poses in index:
        # Note: If the last sequence ends up to have only a single image, it gets dropped
        remainder = len(filenames) % sequence_length
        end = len(filenames) - 1 if remainder == 1 else len(filenames)

        subsequence = [(filenames[i:i + sequence_length], poses[i:i + sequence_length])
                       for i in range(0, end, sequence_length)]
        subsequences.extend(subsequence)

    return subsequences


def build_folder_index(root_folder, ground_truth_folder):
    sequence_dirs = [path.join(root_folder, d) for d in os.listdir(root_folder)
                     if path.isdir(path.join(root_folder, d))]

    pose_files = glob.glob(os.path.join(ground_truth_folder, '*.{}'.format(POSE_FILE_EXTENSION)))

    sequence_dirs.sort()
    pose_files.sort()

    index = [build_index_for_folder_sequence(sequence_dir, pose_file)
             for pose_file, sequence_dir in zip(pose_files, sequence_dirs)]

    return index


def build_index_for_folder_sequence(sequence_dir, pose_file):
    all_times, all_poses = read_from_text_file(pose_file)
    interpolator = interpolate.interp1d(all_times, all_poses, kind='nearest', axis=0, copy=True,
                                        fill_value='extrapolate', assume_sorted=True)

    # Sort filenames according to time given in filename
    filenames = [path.join(sequence_dir, f) for f in os.listdir(sequence_dir)]
    time_from_filename = lambda x: int(path.splitext(path.basename(x))[0])
    filenames.sort(key=time_from_filename)

    # Interpolate at times given by filename
    query_times = [time_from_filename(f) for f in filenames]
    poses_interpolated = [interpolator(t) for t in query_times]

    return filenames, np.array(poses_interpolated)


def encode_poses(positions, quaternions):
    # positions: n x 3 array
    # quaternions: n x 4 array
    vectors = [encode_pose(p, q) for p, q in zip(positions, quaternions)]
    return np.concatenate(vectors)


def encode_pose(position, quaternion):
    return np.concatenate((position, quaternion)).reshape(1, -1)


def read_from_text_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = [s.split() for s in lines]

    times = np.array([float(line[1]) for line in lines])
    poses = np.array([[float(x) for x in line[2:8]] for line in lines])

    return times, poses


def get_positions(pose_array):
    return pose_array[:, 0:3]


def get_quaternions(pose_array):
    angles = pose_array[:, 3:6]

    # Order saved in file: pitch roll yaw
    # Apply rotations in order: yaw, pitch, roll
    quaternions = [euler2quat(radians(eul[2]), radians(eul[0]), radians(eul[1]), 'rzxy') for eul in angles]
    return np.array(quaternions)


def get_camera_optical_axes(quaternions):
    return np.array([rotate_vector([0, 1, 0], q) for q in quaternions])


def to_relative_pose(translations, quaternions):
    t1 = translations[0]
    q1_inv = qinverse(quaternions[0])

    rel_translations = [rotate_vector(t - t1, q1_inv) for  t in translations]
    rel_quaternions = [qmult(q1_inv, q) for q in quaternions]

    return np.array(rel_translations), np.array(rel_quaternions)


def plot_camera_path_2D(file, resolution=1.0, show_rot=True, output='path.pdf'):
    assert 0 < resolution <= 1
    step = int(1 / resolution)

    _, poses = read_from_text_file(file)
    positions = get_positions(poses)[::step]
    quaternions = get_quaternions(poses)[::step]

    # Convert to relative pose
    positions, quaternions = to_relative_pose(positions, quaternions)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    plt.clf()
    plt.plot(x, y)

    if show_rot:
        y_axes = get_camera_optical_axes(quaternions)
        u, v, w = y_axes[:, 0], y_axes[:, 1], y_axes[:, 2]

        plt.quiver(x, y, u, v, units='xy', scale_units='xy', scale=0.5, width=0.05)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.axis('equal')

    plt.savefig(output, bbox_inches='tight')


def visualize_predicted_path(predictions, targets, output_file, resolution=1.0, show_rot=True):
    assert 0 < resolution <= 1
    step = int(1 / resolution)

    positions1 = predictions[::step, :3]
    positions2 = targets[::step, :3]

    quaternions1 = predictions[::step, 3:]
    quaternions2 = targets[::step, 3:]

    x1, y1, z1 = positions1[:, 0], positions1[:, 1], positions1[:, 2]
    x2, y2, z2 = positions2[:, 0], positions2[:, 1], positions2[:, 2]

    plt.clf()

    plt.subplot(121)
    plt.plot(x1, y1, 'b', label='Prediction')
    plt.plot(x2, y2, 'r', label='Ground Truth')

    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.axis('equal')
    plt.title('Top view')

    plt.subplot(122)
    plt.plot(y1, z1, 'b', label='Prediction')
    plt.plot(y2, z2, 'r', label='Ground Truth')

    plt.legend()
    plt.ylabel('z')
    plt.xlabel('y')
    plt.axis('equal')
    plt.title('Side view (height)')

    # if show_rot:
    #     y_axes1 = get_camera_optical_axes(quaternions1)
    #     y_axes2 = get_camera_optical_axes(quaternions2)
    #
    #     u1, v1, w1 = y_axes1[:, 0], y_axes1[:, 1], y_axes1[:, 2]
    #     u2, v2, w2 = y_axes2[:, 0], y_axes2[:, 1], y_axes2[:, 2]
    #
    #     plt.quiver(x1, y1, u1, v1, units='xy', scale_units='xy', scale=0.9, width=0.01, color='b')
    #     plt.quiver(x2, y2, u2, v2, units='xy', scale_units='xy', scale=0.9, width=0.01, color='r')


    plt.savefig(output_file, bbox_inches='tight')






########################################
# ZIP
########################################


def concat_zip_dataset(folders, sequence_length, image_transform=None, sequence_transform=None, return_filename=True, max_size=None):
    datasets = []
    current_size = 0
    for folder in folders:
        zip_files = [file for file in glob.glob(os.path.join(folder, '*.zip'))]
        for zip_file in zip_files:
            sequence = ZippedSequence(zip_file, sequence_length, image_transform, sequence_transform, return_filename)
            datasets.append(sequence)

            current_size += len(sequence)
            if max_size and current_size >= max_size:
                return ConcatDataset(datasets)

    return ConcatDataset(datasets)


def read_poses_from_zip(zip_file):
    with ZipFile(zip_file, 'r') as archive:
        with archive.open('poses.txt') as f:
            lines = [s.split() for s in f.readlines()]

    times = np.array([float(line[1]) for line in lines])
    poses = np.array([[float(x) for x in line[2:8]] for line in lines])

    return times, poses


def build_zipfile_index(zipfile):
    all_times, all_poses = read_poses_from_zip(zipfile)
    interpolator = interpolate.interp1d(all_times, all_poses, kind='nearest', axis=0, copy=True,
                                        fill_value='extrapolate', assume_sorted=True)

    # Sort filenames according to time given in filename
    with ZipFile(zipfile, 'r') as archive:
        filenames = [i.filename for i in archive.infolist() if not i.is_dir() and i.filename.endswith('.jpg')]

    time_from_filename = lambda x: int(path.splitext(path.basename(x))[0])
    filenames.sort(key=time_from_filename)

    # Interpolate at times given by filename
    query_times = [time_from_filename(f) for f in filenames]
    poses_interpolated = [interpolator(t) for t in query_times]
    poses = np.array(poses_interpolated)

    assert len(filenames) == poses.shape[0]
    return filenames, poses


def split_zipfile_index(index, sequence_length):
    # Makes small chunks out of the large sequence
    filenames, poses = index
    # Note: If the last sequence ends up to have only a single image, it gets dropped
    remainder = len(filenames) % sequence_length
    end = len(filenames) - 1 if remainder == 1 else len(filenames)

    subsequences = [(filenames[i:i + sequence_length], poses[i:i + sequence_length])
                   for i in range(0, end, sequence_length)]

    return subsequences


class ZippedSequence(Dataset):

    def __init__(self, zip_file, sequence_length, image_transform=None, sequence_transform=None, return_filename=True):
        self.zip_file = zip_file
        self.image_transform = image_transform
        self.sequence_transform = sequence_transform
        self.return_filename = return_filename
        index = build_zipfile_index(zip_file)
        self.sequence_length = len(index[0]) if sequence_length == 0 else min(sequence_length, len(index[0]))
        self.subsequences = split_zipfile_index(index, self.sequence_length)
        # Format of subsequences: [(filenames1, poses1), (filenames2, poses2), ...]

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        filenames, poses = self.subsequences[idx]
        images = [im.unsqueeze(0) for im in self.load_images(filenames)]
        image_sequence = torch.cat(images, dim=0)

        # Convert raw pose (from text file) to 7D pose vectors (translation + quaternion)
        positions = get_positions(poses)
        quaternions = get_quaternions(poses)

        if self.sequence_transform:
            image_sequence = self.sequence_transform(image_sequence)
            positions = self.sequence_transform(positions)
            quaternions = self.sequence_transform(quaternions)

        rel_positions, rel_quaternions = to_relative_pose(positions, quaternions)
        pose_vectors = encode_poses(rel_positions, rel_quaternions)
        pose_sequence = torch.from_numpy(pose_vectors).float()

        if self.return_filename:
            return image_sequence, pose_sequence, [os.path.join(self.zip_file, f) for f in filenames]
        else:
            return image_sequence, pose_sequence

    def load_images(self, filenames):
        images = []
        with ZipFile(self.zip_file) as archive:
            for entry in filenames:
                with archive.open(entry) as file:
                    image = Image.open(file).convert('RGB')
                    if self.image_transform:
                        image = self.image_transform(image)
                    images.append(image)

        return images


class RandomSequenceReversal(object):
    """ A transform that reverses a sequence randomly with probability 0.5. """

    def __call__(self, sequence):
        assert isinstance(sequence, torch.Tensor) or isinstance(sequence, np.ndarray), \
            'Sequence reversal only works for torch tensors or numpy arrays.'

        if random.uniform(0, 1) > 0.5:
            return sequence

        if isinstance(sequence, np.ndarray):
            return sequence[::-1]

        elif isinstance(sequence, torch.Tensor):
            # Currently, pytorch does not support negative step size for slicing
            idx = [i for i in range(sequence.size(0) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            return sequence.index_select(0, idx)
