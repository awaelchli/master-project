import glob
import os.path as path

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import read_matrices, to_relative_poses, matrix_to_pose_vector

FOLDERS = {
    'color': '../data/KITTI/color/sequences',
    'grayscale': '../data/KITTI/grayscale/sequences',
    'poses':  '../data/KITTI/poses/',
    'stereo_subfolder': {
        'grayscale': {
            0: 'image_0',
            1: 'image_1'
        },
        'color': {
            0: 'image_2',
            1: 'image_3'
        }
    }
}

SEQUENCES = {
    'training': [0, 1, 2, 5, 6, 7, 9, 9],
    'validation': [4],
    'test': [3, 10]
}

IMAGE_EXTENSION = 'png'
FILE_POSE_EXTENSION = 'txt'


def get_root_folder(grayscale=False):
    if grayscale:
        return FOLDERS['grayscale']
    else:
        return FOLDERS['color']


def get_pose_folder():
    return FOLDERS['poses']


def get_stereo_subfolder(grayscale=False, eye=0):
    stereo = FOLDERS['stereo_subfolder']
    if grayscale:
        return stereo['grayscale'][eye]
    else:
        return stereo['color'][eye]


class Subsequence(Dataset):

    def __init__(self, sequence_length, transform=None, grayscale=False, eye=0, sequence_numbers=list(range(0, 10))):
        self.transform = transform
        self.is_grayscale = grayscale
        self.eye = eye
        self.sequence_length = sequence_length
        self.index = build_index(sequence_length, sequence_numbers, grayscale, eye)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sequence_sample = self.index[idx]
        images = [s.get_image(self.transform).unsqueeze_(0) for s in sequence_sample]

        matrix_poses = [s.get_pose_matrix() for s in sequence_sample]
        matrix_poses = to_relative_poses(matrix_poses)
        poses_6d = [matrix_to_pose_vector(m) for m in matrix_poses]

        image_sequence = torch.cat(tuple(images), 0)
        pose_sequence = torch.cat(tuple(poses_6d), 0)
        return image_sequence, pose_sequence


def read_pose_strings(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()
    return lines


def sequence_number_to_string(number):
    return '{:02d}'.format(number)


def get_image_search_path(sequence_name, is_grayscale, eye):
    image_subfolder = get_stereo_subfolder(is_grayscale, eye)
    search = '*.{}'.format(IMAGE_EXTENSION)
    search_path = path.join(get_root_folder(is_grayscale), sequence_name, image_subfolder, search)
    return search_path


def get_pose_filename(sequence_name):
    return path.join(get_pose_folder(), '{}.{}'.format(sequence_name, FILE_POSE_EXTENSION))


def index_sequence(chunk_size, sequence_number, is_grayscale, eye):
    """ Returns a list of image sample chunks for a given KITTI sequence """

    sequence_name = sequence_number_to_string(sequence_number)
    search_path = get_image_search_path(sequence_name, is_grayscale, eye)
    filenames = glob.glob(search_path)
    filenames.sort()

    print('Sequence {}: {:d} files found.'.format(sequence_name, len(filenames)))

    sequence_beginnings = range(0, len(filenames), chunk_size)
    sequence_beginnings = sequence_beginnings[0: len(sequence_beginnings) - 1]

    pose_matrices = read_matrices(get_pose_filename(sequence_name))

    chunks = []
    for i, j in enumerate(sequence_beginnings):
        chunk_files = filenames[j: j + chunk_size]
        poses = pose_matrices[j: j + chunk_size]

        chunk = [ImageSample(file, sequence_name, pose, eye=eye)
                 for file, pose in zip(chunk_files, poses)]
        chunks.append(chunk)

    return chunks


def build_index(chunk_size, sequence_numbers, grayscale=False, eye=0):
    chunks = []
    for number in sequence_numbers:
        chunks.extend(index_sequence(chunk_size, number, grayscale, eye))
    return chunks


class ImageSample(object):

    def __init__(self, file, sequence_label, pose_matrix, eye=0):
        self.file = file
        self.sequence_label = sequence_label
        self.pose_matrix = pose_matrix

    def get_image(self, transform=None):
        image = Image.open(self.file).convert('RGB')
        if transform is not None:
            image = transform(image)
        return image

    def get_pose_matrix(self):
        return self.pose_matrix



