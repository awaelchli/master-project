import os
import os.path as path
import glob
import torch
from dataimport.utils import read_matrices, to_relative_poses, matrix_to_pose_vector
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from shutil import copyfile, rmtree


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

        string_number = sequence_number_to_string(sequence_number)
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


class Subsequence(Dataset):

    def __init__(self, sequence_length, sequences_dir, pose_dir, transform=None,
                 eye=0, sequence_numbers=list(range(0, 10))):
        self.sequences_dir = sequences_dir
        self.pose_dir = pose_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.index = build_index(sequence_length, sequences_dir, pose_dir,
                                 sequence_numbers, eye)

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

#
# def read_6D_poses(pose_file):
#     with open(pose_file, 'r') as f:
#         lines = f.readlines()
#
#     poses = []
#     for line in lines:
#         vector = torch.FloatTensor([float(s) for s in line.split()]).view(1, 6)
#         poses.append(vector)
#
#     return poses

def sequence_number_to_string(number):
    return '{:02d}'.format(number)

def index_sequence(chunk_size, sequences_root_dir, sequence_number, pose_dir, eye):
    """ Returns a list of image sample chunks for a given KITTI sequence """

    sequence_name = sequence_number_to_string(sequence_number)
    search_path = path.join(sequences_root_dir, sequence_name, 'image_{:d}'.format(eye), '*.png')
    filenames = glob.glob(search_path)
    filenames.sort()

    print('{:d} files found.'.format(len(filenames)))

    sequence_beginnings = range(0, len(filenames), chunk_size)
    sequence_beginnings = sequence_beginnings[0: len(sequence_beginnings) - 1]

    pose_matrices = read_matrices(path.join(pose_dir, '{}.txt'.format(sequence_name)))

    chunks = []
    for i, j in enumerate(sequence_beginnings):
        chunk_files = filenames[j: j + chunk_size]
        poses = pose_matrices[j: j + chunk_size]

        chunk = [ImageSample(file, sequence_name, pose, eye=eye)
                 for file, pose in zip(chunk_files, poses)]
        chunks.append(chunk)

    return chunks


def build_index(chunk_size, sequences_root_dir, pose_dir, sequence_numbers, eye=0):
    chunks = []
    for number in sequence_numbers:
        chunks.extend(index_sequence(chunk_size, sequences_root_dir, number, pose_dir, eye))
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


def convert_folder(sequence_dir, pose_dir, new_root, new_sequence_length, eye=0):
    if os.path.isdir(new_root):
        rmtree(new_root)
        print('Removed existing folder.')

    os.mkdir(new_root)

    sequence_index = 0
    for sequence_name in next(os.walk(sequence_dir))[1]:
        print('Converting sequence {}'.format(sequence_name))
        search_path = path.join(sequence_dir, sequence_name, 'image_{:d}'.format(eye), '*.png')
        print(search_path)
        filenames = glob.glob(search_path)
        filenames.sort()

        print('{:d} files found.'.format(len(filenames)))

        sequence_beginnings = range(0, len(filenames), new_sequence_length)
        sequence_beginnings = sequence_beginnings[0 : len(sequence_beginnings) - 1]

        pose_matrices = read_matrices(path.join(pose_dir, '{}.txt'.format(sequence_name)))

        for i, j in enumerate(sequence_beginnings):
            folder = path.join(new_root, '{:06d}'.format(sequence_index))
            os.mkdir(folder)
            print('Moving images [{:d}/{:d}]'.format(i + 1, len(sequence_beginnings)))

            for k, name in enumerate(filenames[j : j + new_sequence_length]):
                new_name = '{:04d}{}'.format(k, path.splitext(name)[1])
                new_name = path.join(folder, new_name)
                copyfile(name, new_name)

            # Write one pose file per mini-sequence
            poses = pose_matrices[j : j + new_sequence_length]
            poses = to_relative_poses(poses)

            with open(path.join(folder, 'poses.txt'), 'w') as f:
                for pose in poses:
                    f.write(' '.join([str(e) for e in pose.view(12)]))
                    f.write('\n')

            sequence_index += 1
