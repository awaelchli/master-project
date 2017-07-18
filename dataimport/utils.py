import os
import os.path as path
import torch
import numpy as np
import glob
from transforms3d.quaternions import mat2quat, quat2axangle


def matrix_to_pose_vector(matrix):
    matrix = matrix.numpy()
    quaternion = mat2quat(matrix[:, 0 : 3])
    axis, angle = quat2axangle(quaternion)

    translation = torch.from_numpy(matrix[:, 3]).contiguous().view(1, 3)
    orientation = torch.from_numpy(axis[[0, 1]]).view(1, 2)
    angle = torch.FloatTensor([[angle]])

    pose = torch.cat((translation.float(), orientation.float(), angle), 1)
    return pose


def read_matrix_poses(pose_file):
    matrices = read_matrices(pose_file)
    return list(map(matrix_to_pose_vector, matrices))


def read_matrices(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()

    matrices = []
    for line in lines:
        vector = torch.FloatTensor([float(s) for s in line.split()])
        matrix = vector.view(3, 4)
        matrices.append(matrix)

    return matrices


# def convert_pose_files(pose_dir, new_pose_dir):
#     file_list = glob.glob(path.join(pose_dir, '*.txt'))
#     if not os.path.isdir(new_pose_dir):
#         os.mkdir(new_pose_dir)
#
#     for file in file_list:
#         poses = read_matrix_poses(file)
#         with open(path.join(new_pose_dir, path.basename(file)), 'w') as f:
#             for pose in poses:
#                 f.write(' '.join([str(e) for e in pose.view(6)]))
#                 f.write('\n')


def to_relative_poses_old(matrices):
    # Given: R(1->world), R(2->world), t1, t2
    #
    # R(1->2) = R(world->2) * R(1->world)
    #         = R(2->world)^(-1) * R(1->world)
    #
    # t(1->2) = R(2->world)^(-1) * (t1 - t2)
    rotations = [m[:, 0 : 3] for m in matrices]
    translations = [m[:, 3] for m in matrices]

    rot1 = rotations[0]
    t1 = translations[0]

    rel_matrices = []
    for r, t in zip(rotations, translations):
        r_inv = r.t()
        r_rel = torch.mm(r_inv, rot1)
        t_rel = torch.mv(r_inv, t1 - t)
        rel_matrices.append(torch.cat((r_rel, t_rel), 1))

    return rel_matrices


def to_relative_poses(matrices):
    # Given: R(1->world), R(2->world), t1, t2
    #
    # R(2->1) = R(world->1) * R(2->world)
    #         = R(1->world)^(-1) * R(2->world)
    #
    # t(2->1) = R(1->world)^(-1) * (t2 - t1)
    rotations = [m[:, 0 : 3] for m in matrices]
    translations = [m[:, 3] for m in matrices]

    rot1_inv = rotations[0].t() # For rotations: inverse = transpose
    t1 = translations[0]

    rel_matrices = []
    for r, t in zip(rotations, translations):
        r_rel = torch.mm(rot1_inv, r)
        t_rel = torch.mv(rot1_inv, t - t1)
        rel_matrices.append(torch.cat((r_rel, t_rel), 1))

    return rel_matrices
