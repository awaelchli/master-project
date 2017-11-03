import glob
import os
import os.path as path

import torch
from PIL import Image
from torch.utils.data import Dataset

from pose_transforms import relative_to_first_pose_matrix, matrix_to_euler_pose_vector
import matplotlib.pyplot as plt

FOLDERS = {
    'train': '../data/VIPER/train/',
    'val': '../data/VIPER/val/',
    'test': '../data/VIPER/test/',
}

BLACK_LIST = {
    'train': ['045', '048', '049'],
    'val': ['043', '042', '029'],
    'test': []
}

IMAGE_EXTENSION = 'jpg'
POSE_FILE_EXTENSION = 'txt'
IMAGE_SUBFOLDER = 'img'
POSE_SUBFOLDER = 'poses'


class Subsequence(Dataset):

    def __init__(self, folder, sequence_length, overlap=0, transform=None, max_size=None):
        self.folder = folder
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.transform = transform

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

        matrix_poses = [s.get_pose_matrix() for s in sequence_sample]
        matrix_poses = relative_to_first_pose_matrix(matrix_poses)
        #poses_6d = [matrix_to_quaternion_pose_vector(m) for m in matrix_poses]
        poses_6d = [matrix_to_euler_pose_vector(m) for m in matrix_poses]

        image_sequence = torch.cat(tuple(images), 0)
        pose_sequence = torch.cat(tuple(poses_6d), 0)
        return image_sequence, pose_sequence, filenames

    def get_sequence_names(self):
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
        search_path = self.get_image_search_path(sequence_name)
        filenames = glob.glob(search_path)
        filenames.sort()
        print('Sequence {}: {:d} files found.'.format(sequence_name, len(filenames)))

        sequence_beginnings = range(0, len(filenames), self.sequence_length - self.overlap)
        sequence_beginnings = sequence_beginnings[0: len(sequence_beginnings) - 1]

        pose_matrices = read_matrices(self.get_pose_filename(sequence_name))

        chunks = []
        for i, j in enumerate(sequence_beginnings):
            chunk_files = filenames[j: j + self.sequence_length]
            poses = pose_matrices[j: j + self.sequence_length]

            chunk = [ImageSample(file, sequence_name, pose)
                     for file, pose in zip(chunk_files, poses)]
            chunks.append(chunk)

        return chunks


def read_matrices(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()

    matrices = []
    for line in lines:
        line = line.split(',')
        line = line[1:] # Ignore frame number

        vector = torch.Tensor([float(s) for s in line])
        matrix = vector.view(3, 4)
        matrices.append(matrix)

    return matrices


class ImageSample(object):

    def __init__(self, file, sequence_label, pose_matrix):
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


def visualize_predicted_path(predictions, targets, output_file, resolution=1.0):
    assert 0 < resolution <= 1
    step = int(1 / resolution)

    positions1 = predictions[::step, :3]
    positions2 = targets[::step, :3]

    x1, y1, z1 = positions1[:, 0], positions1[:, 1], positions1[:, 2]
    x2, y2, z2 = positions2[:, 0], positions2[:, 1], positions2[:, 2]

    z1 = -z1
    z2 = -z2

    plt.clf()

    plt.subplot(121)
    plt.plot(x2, z2, 'ro-', label='Ground Truth')
    plt.plot(x1, z1, 'bo-', label='Prediction')

    plt.legend()
    plt.ylabel('z')
    plt.xlabel('x')
    plt.axis('equal')
    plt.title('Bird view')

    plt.subplot(122)

    plt.plot(z2, y2, 'ro-', label='Ground Truth')
    plt.plot(z1, y1, 'bo-', label='Prediction')

    plt.legend()
    plt.ylabel('y')
    plt.xlabel('z')
    plt.axis('equal')
    plt.title('Side view (height)')

    plt.savefig(output_file, bbox_inches='tight')


# from torchvision.transforms import ToTensor
# t = ToTensor()
#
# s = Subsequence('/media/adrian/Data/adrian/Datasets/VIPER/train', 10, overlap=2, transform=t)
# print(s)
#
# from torch.utils.data.dataloader import DataLoader
#
#
# d = DataLoader(s)
#
# for i, (a, b, c) in enumerate(d):
#     if i < 10:
#         print(b)