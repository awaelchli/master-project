import os
import os.path as path
import torch
import numpy as np
import glob
from transforms3d.quaternions import mat2quat, quat2axangle


def matrix_to_pose_vector(matrix):
    quaternion = mat2quat(matrix[:, [0, 1, 2]])
    axis, angle = quat2axangle(quaternion)

    translation = torch.from_numpy(matrix[:, 3]).contiguous().view(1, 3)
    orientation = torch.from_numpy(axis[[0, 1]]).view(1, 2)
    angle = torch.DoubleTensor([[angle]])

    pose = torch.cat((translation, orientation, angle), 1)

    return pose


def read_matrix_poses(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()

    poses = []
    for line in lines:
        vector = np.array([float(s) for s in line.split()])
        matrix = vector.reshape(3, 4)
        poses.append(matrix_to_pose_vector(matrix))

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
