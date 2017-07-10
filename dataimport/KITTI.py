import os
import os.path as path
import glob
import torch
import numpy as np
from dataimport.utils import matrix_to_pose_vector
from skimage import io
from torch.utils.data import Dataset, DataLoader


class Sequence(Dataset):
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
        self.pose_file = path.join(self.pose_dir, '{}.txt'.format(string_number))
        self.poses = read_6D_poses(self.pose_file)

    def __len__(self):
        return len(self.get_file_list())

    def __getitem__(self, idx):
        img_name = self.get_file_list()[idx]
        gray_image = torch.from_numpy(io.imread(img_name)).float()

        # In case the image is grayscale
        height = gray_image.size()[0]
        width = gray_image.size()[1]
        gray_image.resize_(1, height, width)
        image = gray_image.expand(3, height, width)

        sample = (image, self.poses[idx])
        return sample

    def get_file_list(self):
        return glob.glob(path.join(self.images_dir, '*.png'))


def read_matrix_poses(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()

    poses = []
    for line in lines:
        vector = np.array([float(s) for s in line.split()])
        matrix = vector.reshape(3, 4)
        poses.append(matrix_to_pose_vector(matrix))

    return poses


def read_6D_poses(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()

    poses = []
    for line in lines:
        vector = torch.FloatTensor([float(s) for s in line.split()]).view(1, 6)
        poses.append(vector)

    return poses


def convert_pose_files(pose_dir, new_pose_dir):
    file_list = glob.glob(path.join(pose_dir, '*.txt'))
    if not os.path.isdir(new_pose_dir):
        os.mkdir(new_pose_dir)

    for file in file_list:
        poses = read_matrix_poses(file)
        with open(path.join(new_pose_dir, path.basename(file)), 'w') as f:
            for pose in poses:
                f.write(' '.join([str(e) for e in pose.view(6)]))
                f.write('\n')


