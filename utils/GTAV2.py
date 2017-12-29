import glob
import os
import os.path as path
from scipy import interpolate
import math

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from pose_transforms import relative_to_first_pose_matrix, matrix_to_euler_pose_vector, euler_pose_vector_to_matrix, matrix_to_quaternion_pose_vector
from pose_evaluation import relative_quaternion_rotation_error
import matplotlib.pyplot as plt

FOLDERS = {
    'train': '../data/GTAV/train/',
    'val': '../data/GTAV/val/',
    'test': '../data/GTAV/test/',
}

BLACK_LIST = {
    'train': [],
    'val': [],
    'test': [],
}

IMAGE_EXTENSION = 'jpg'
POSE_FILE_EXTENSION = 'txt'
IMAGE_SUBFOLDER = 'data'
POSE_SUBFOLDER = 'poses'


class Subsequence(Dataset):

    def __init__(self, folder, sequence_length, overlap=0, transform=None, max_size=None, sequence_name=None, relative_pose=True):
        self.folder = folder
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.transform = transform
        self.sequence_name = sequence_name
        self.relative_pose = relative_pose

        assert 0 <= overlap < sequence_length
        self.index = self.build_index()

        if max_size:
            # Clip dataset
            assert 0 < max_size < len(self.index)
            self.index = self.index[:max_size]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sequence_sample = self.index[idx]
        images = [s.get_image(self.transform).unsqueeze_(0) for s in sequence_sample]
        filenames = [s.file for s in sequence_sample]

        poses = [s.get_pose() for s in sequence_sample]
        matrix_poses = [raw_pose_to_matrix(p) for p in poses]
        if self.relative_pose:
            matrix_poses = relative_to_first_pose_matrix(matrix_poses)

        #poses_6d = [matrix_to_quaternion_pose_vector(m) for m in matrix_poses]
        poses_6d = [matrix_to_euler_pose_vector(m) for m in matrix_poses]

        image_sequence = torch.cat(tuple(images), 0)
        pose_sequence = torch.cat(tuple(poses_6d), 0)
        return image_sequence, pose_sequence, filenames

    def get_sequence_names(self):
        if self.sequence_name:
            # A specific sequence is requested
            full_path = os.path.join(self.folder, IMAGE_SUBFOLDER, self.sequence_name)
            assert os.path.exists(full_path), 'Requested sequence does not exist: {}'.format(dir)
            dirs = [self.sequence_name]
        else:
            # All sequences are requested
            dirs = os.listdir(os.path.join(self.folder, IMAGE_SUBFOLDER))
            dirs.sort()

            # Remove items from blacklist
            for key in BLACK_LIST.keys():
                if key in self.folder:
                    dirs = [d for d in dirs if d not in BLACK_LIST[key]]

        return dirs

    def get_image_search_path(self, sequence_name):
        search = '*.{}'.format(IMAGE_EXTENSION)
        search_path = path.join(self.folder, IMAGE_SUBFOLDER, sequence_name, search)
        return search_path

    def get_pose_filename(self, sequence_name):
        return path.join(self.folder, POSE_SUBFOLDER, '{}.{}'.format(sequence_name, POSE_FILE_EXTENSION))

    def build_index(self):
        chunks = []
        sequence_names = self.get_sequence_names()
        for name in sequence_names:
            chunks.extend(self.index_sequence(name))
        return chunks

    def index_sequence(self, sequence_name):
        #search_path = self.get_image_search_path(sequence_name)
        #filenames = glob.glob(search_path)
        #filenames.sort()
        filenames, poses = self.read_filenames_and_poses(sequence_name)
        print('Sequence {}: {:d} files found.'.format(sequence_name, len(filenames)))

        sequence_beginnings = range(0, len(filenames), self.sequence_length - self.overlap)
        sequence_beginnings = sequence_beginnings[0: len(sequence_beginnings) - 1]

        #pose_matrices = read_matrices(self.get_pose_filename(sequence_name))

        chunks = []
        for i, j in enumerate(sequence_beginnings):
            chunk_files = filenames[j: j + self.sequence_length]
            p = poses[j: j + self.sequence_length]

            chunk = [ImageSample(file, sequence_name, pose)
                     for file, pose in zip(chunk_files, p)]
            chunks.append(chunk)

        return chunks

    def read_filenames_and_poses(self, sequence_name):
        pose_file = self.get_pose_filename(sequence_name)
        all_times, all_poses = read_from_text_file(pose_file)
        interpolator = interpolate.interp1d(all_times, all_poses, kind='nearest', axis=0, copy=True,
                                            fill_value='extrapolate', assume_sorted=True)

        # Sort filenames according to time given in filename
        filenames = glob.glob(self.get_image_search_path(sequence_name))

        time_from_filename = lambda x: int(path.splitext(path.basename(x))[0])
        filenames.sort(key=time_from_filename)

        # Interpolate at times given by filename
        query_times = [time_from_filename(f) for f in filenames]
        poses_interpolated = [interpolator(t) for t in query_times]

        return filenames, np.array(poses_interpolated)

    def print_statistics(self):
        sequence_names = self.get_sequence_names()
        all_rotation_angles = []
        all_translation_norms = []

        for name in sequence_names:
            _, raw_poses = self.read_filenames_and_poses(name)
            matrices = [raw_pose_to_matrix(p) for p in raw_poses]
            quaternion_pose_vecs = [matrix_to_quaternion_pose_vector(m) for m in matrices]
            quaternion_pose_vecs = torch.cat(quaternion_pose_vecs, 0)
            positions = quaternion_pose_vecs[:, :3]
            quats = quaternion_pose_vecs[:, 3:]

            quats1 = quats[:-1, :]
            quats2 = quats[1:, :]

            positions1 = positions[:-1, :]
            positions2 = positions[1:, :]

            translation_norms = torch.norm(positions1 - positions2, p=2, dim=1)
            rotation_angles = relative_quaternion_rotation_error(quats1, quats2)

            all_rotation_angles += rotation_angles
            all_translation_norms += list(translation_norms)

        #print(sum(math.isnan(x) for x in all_rotation_angles))

        avg_rotation = sum([x for x in all_rotation_angles if not math.isnan(x)]) / len(all_rotation_angles)
        avg_translation = sum(all_translation_norms) / len(all_translation_norms)

        max_rotation = max(all_rotation_angles)
        max_translation = max(all_translation_norms)

        min_rotation = min(all_rotation_angles)
        min_translation = min(all_translation_norms)


        print('Rotation angle between consecutive frames (AVG / MIN / MAX): {:.4f} / {:.4f} / {:.4f} degrees'.format(avg_rotation, min_rotation, max_rotation))
        print('Distance travelled between consecutive frames (AVG / MIN / MAX): {:.4f} / {:.4f} / {:.4f} meters'.format(avg_translation, min_translation, max_translation))


def read_from_text_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = [s.split() for s in lines]

    times = np.array([float(line[1]) for line in lines])
    poses = np.array([[float(x) for x in line[2:8]] for line in lines])

    return times, poses


def raw_pose_to_matrix(raw_pose):
    position = raw_pose[0:3]
    euler = raw_pose[3:6]
    rad_euler = np.radians(np.array([euler[2], euler[0], euler[1]]))
    pose = np.concatenate((position, rad_euler))

    return euler_pose_vector_to_matrix(torch.from_numpy(pose).float().view(1, -1), mode='rzxy')


class ImageSample(object):

    def __init__(self, file, sequence_label, pose):
        self.file = file
        self.sequence_label = sequence_label
        self.pose = pose

    def get_image(self, transform=None):
        image = Image.open(self.file).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    def get_pose(self):
        return self.pose


def visualize_predicted_path(predictions, targets, output_file, resolution=1.0):
    assert 0 < resolution <= 1
    step = int(1 / resolution)
    marker_freq = int(0.1 * len(predictions))
    ms = 5

    folder, name = path.split(output_file)
    fname1 = path.join(folder, 'bird-' + name)
    fname2 = path.join(folder, 'both-' + name)

    positions1 = predictions[::step, :3]
    positions2 = targets[::step, :3]

    x1, y1, z1 = positions1[:, 0], positions1[:, 1], positions1[:, 2]
    x2, y2, z2 = positions2[:, 0], positions2[:, 1], positions2[:, 2]

    #z1 = -z1
    #z2 = -z2

    plt.clf()
    #fig = plt.gcf()
    #fig.suptitle('Marker every {:d} frames'.format(marker_freq))

    plt.plot(x2, y2, 'ro-', label='Ground Truth', markevery=marker_freq, markersize=ms)
    plt.plot(x1, y1, 'bo-', label='Prediction', markevery=marker_freq, markersize=ms)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.axis('equal')
    plt.title("Bird's-eye view")

    plt.legend(loc=2)
    plt.savefig(fname1, bbox_inches='tight')



    plt.clf()
    #fig = plt.gcf()
    #fig.suptitle('Marker every {:d} frames'.format(marker_freq))

    plt.subplot(121)
    plt.plot(x2, y2, 'ro-', label='Ground Truth', markevery=marker_freq, markersize=ms)
    plt.plot(x1, y1, 'bo-', label='Prediction', markevery=marker_freq, markersize=ms)

    #plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.axis('equal')
    plt.title("Bird's-eye view")

    plt.subplot(122)
    plt.plot(y2, z2, 'ro-', label='Ground Truth', markevery=marker_freq, markersize=ms)
    plt.plot(y1, z1, 'bo-', label='Prediction', markevery=marker_freq, markersize=ms)

    plt.legend(loc=2)
    plt.ylabel('y')
    plt.xlabel('z')
    plt.axis('equal')
    plt.title('Side view (height)')

    plt.savefig(fname2, bbox_inches='tight')



# import torchvision.transforms as transforms
#
# transform = transforms.Compose([
#             transforms.Scale(320),
#             transforms.CenterCrop((320, 448)),
#             transforms.ToTensor(),
#         ])
#
# s = Subsequence(FOLDERS['train'], 5, overlap=0, transform=transform, max_size=None, sequence_name=None, relative_pose=True)

