import glob
import os
from math import radians
from os import path

import matplotlib.pyplot as plt
plt.switch_backend('agg') # For machines without display (e.g. cluster)
import numpy as np
import torch
from PIL import Image
from scipy import interpolate
from torch.utils.data import Dataset
from transforms3d.euler import euler2quat
from transforms3d.quaternions import rotate_vector, qinverse, qmult

FOLDERS = {
    'walking': {
        'training': {
            'data': '../data/GTA V/walking/train/data/',
            'pose': '../data/GTA V/walking/train/poses/'
        },
        'validation':{
            'data': '../data/GTA V/walking/test/data/',
            'pose': '../data/GTA V/walking/test/poses/',
        },
        'test': {
            'data': '../data/GTA V/walking/test/data/',
            'pose': '../data/GTA V/walking/test/poses/',
        }
    }
}

IMAGE_EXTENSION = 'jpg'
POSE_FILE_EXTENSION = 'txt'


class Subsequence(Dataset):

    def __init__(self, data_folder, pose_folder, sequence_length, transform=None, max_size=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.index = build_subsequence_index(data_folder, pose_folder, sequence_length)

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
        reminder = len(filenames) % sequence_length
        end = len(filenames) - 1 if reminder == 1 else len(filenames)

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


def plot_camera_path_2D(file, resolution=1.0, show_rot=True):
    assert 0 < resolution <= 1
    step = 1 / resolution

    _, poses = read_from_text_file(file)
    positions = get_positions(poses)[::step]
    quaternions = get_quaternions(poses)[::step]

    # Convert to relative pose
    positions, quaternions = to_relative_pose(positions, quaternions)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    plt.plot(x, y)

    if show_rot:
        y_axes = get_camera_optical_axes(quaternions)
        u, v, w = y_axes[:, 0], y_axes[:, 1], y_axes[:, 2]

        plt.quiver(x, y, u, v, units='xy', scale_units='xy', scale=0.5, width=0.05)

    plt.axis('equal')
    plt.show()


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
    plt.plot(x1, y1, 'b', label='Prediction')
    plt.plot(x2, y2, 'r', label='Ground Truth')

    if show_rot:
        y_axes1 = get_camera_optical_axes(quaternions1)
        y_axes2 = get_camera_optical_axes(quaternions2)

        u1, v1, w1 = y_axes1[:, 0], y_axes1[:, 1], y_axes1[:, 2]
        u2, v2, w2 = y_axes2[:, 0], y_axes2[:, 1], y_axes2[:, 2]

        plt.quiver(x1, y1, u1, v1, units='xy', scale_units='xy', scale=0.9, width=0.01, color='b')
        plt.quiver(x2, y2, u2, v2, units='xy', scale_units='xy', scale=0.9, width=0.01, color='r')

    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')

    plt.axis('equal')
    plt.savefig(output_file, bbox_inches='tight')

#s = r'E:\Rockstar Games\Grand Theft Auto V\08.12.2017 - 18.04.57.txt'
#plot_camera_path_2D(s, 0.07)

#index = build_subsequence_index(r'H:\Datasets\GTAV Dataset\data', r'H:\Datasets\GTAV Dataset\poses', 2)
#print(index[0])

# 3D plot:
#fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(x, y, z)


# Dataset class
# transform = transforms.Compose([
#     transforms.Scale(200),
#     transforms.ToTensor(),
#     #transforms.Normalize()
# ])
# trainset = Subsequence('../' + FOLDERS['training']['data'], '../' + FOLDERS['training']['pose'], 10, transform)
#
# print(trainset[0])
