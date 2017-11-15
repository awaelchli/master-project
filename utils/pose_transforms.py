from transforms3d.euler import euler2quat, mat2euler, euler2mat
from transforms3d.quaternions import rotate_vector, qinverse, qmult, mat2quat
import numpy as np
import torch


def relative_to_first_pose(translations, quaternions):
    """ Makes all translations and quaternions in the list relative to the first translation/quaternion. """

    t1 = translations[0]
    q1_inv = qinverse(quaternions[0])

    rel_translations = [rotate_vector(t - t1, q1_inv) for t in translations]
    rel_quaternions = [qmult(q1_inv, q) for q in quaternions]

    return np.array(rel_translations), np.array(rel_quaternions)


def relative_to_first_pose_matrix(matrices):
    """ Makes all pose matrices in the list relative to the first pose.
        Each matrix in the list is expected to be a 3 x 4 torch tensor encoding rotation in the first 3 x 3 block
        and translation in the last column.
        Given a pair of matrices, the formulas to convert the poses is as follows:

        Given: R(1->world), R(2->world), t1, t2

        R(2->1) = R(world->1) * R(2->world)
                = R(1->world)^(-1) * R(2->world)

        t(2->1) = R(1->world)^(-1) * (t2 - t1)
    """
    rotations = [m[:, 0:3] for m in matrices]
    translations = [m[:, 3] for m in matrices]

    rot1_inv = rotations[0].t()  # For rotations: inverse = transpose
    t1 = translations[0]

    rel_matrices = []
    for r, t in zip(rotations, translations):
        r_rel = torch.mm(rot1_inv, r)
        t_rel = torch.mv(rot1_inv, t - t1)
        rel_matrices.append(torch.cat((r_rel, t_rel), 1))

    return rel_matrices


def matrix_to_quaternion_pose_vector(matrix):
    """ Converts a 3 x 4 pose matrix to a 1 x 7 pose vector.
        The rotational part is converted to a unit quaternion occupying the first four elements of the output vector.
        The translation is copied from the last column of the matrix.
    """
    matrix = matrix.numpy()
    quaternion = torch.from_numpy(mat2quat(matrix[:, 0:3])).float().view(1, 4)
    translation = torch.from_numpy(matrix[:, 3]).contiguous().float().view(1, 3)
    pose = torch.cat((translation, quaternion), 1)
    return pose


def matrix_to_euler_pose_vector(matrix):
    """ Converts a 3 x 4 pose matrix to a 1 x 6 pose vector.
        The first three elements of the pose vector are the translations, followed by three euler angles for the
        orientation.
    """
    matrix = matrix.numpy()
    angles = torch.Tensor(mat2euler(matrix[:, 0:3], axes='rxyz')).view(1, 3)
    translation = torch.from_numpy(matrix[:, 3]).contiguous().float().view(1, 3)
    pose = torch.cat((translation, angles), 1)
    return pose


def euler_pose_vector_to_matrix(pose):
    """
    :param pose: 1 x 6 torch vector, (translation, angles)
    :return: 3 x 4 pose matrix (rotation, translation)
    """
    t = pose[:, :3]
    angles = pose[:, 3:].view(-1).numpy()
    rot = torch.from_numpy(euler2mat(angles[0], angles[1], angles[2], axes='rxyz')).float()
    matrix = torch.cat((rot, t.view(3, 1)), 1)
    return matrix


def euler_to_quaternion(angles):
    assert angles.size(1) == 3
    quaternions = [euler2quat(a[0], a[1], a[2], axes='rxyz') for a in angles.numpy()]
    quaternions = torch.Tensor(quaternions)
    return quaternions


def relative_previous_pose_matrix(matrices):
    """ Makes all pose matrices in the list relative to the previous pose.
        Each matrix in the list is expected to be a 3 x 4 torch tensor encoding rotation in the first 3 x 3 block
        and translation in the last column.
        Given a pair of matrices, the formulas to convert the poses is as follows:

        Given: R(1->world), R(2->world), t1, t2

        R(2->1) = R(world->1) * R(2->world)
                = R(1->world)^(-1) * R(2->world)

        t(2->1) = R(1->world)^(-1) * (t2 - t1)
    """
    rotations = [m[:, 0:3] for m in matrices]
    translations = [m[:, 3] for m in matrices]

    rel_matrices = []
    r_prev, t_prev = rotations[0], translations[0]
    for r, t in zip(rotations, translations):
        r_rel = torch.mm(r_prev.t(), r)
        t_rel = torch.mv(r_prev.t(), t - t_prev)
        rel_matrices.append(torch.cat((r_rel, t_rel), 1))
        r_prev = r
        t_prev = t

    return rel_matrices


def relative_previous_pose_to_relative_first_matrix(matrices):
    """
    """
    rotations = [m[:, 0:3] for m in matrices]
    translations = [m[:, 3] for m in matrices]

    rel_matrices = []
    r_prev, t_prev = rotations[0], translations[0]
    for r, t in zip(rotations, translations):
        r_rel = torch.mm(r_prev, r)
        t_rel = torch.mv(r_prev, t) + t_prev
        rel_matrices.append(torch.cat((r_rel, t_rel), 1))
        r_prev = r_rel
        t_prev = t_rel

    return rel_matrices
