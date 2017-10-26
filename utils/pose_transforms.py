from transforms3d.euler import euler2quat
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

    rot1_inv = rotations[0].t() # For rotations: inverse = transpose
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
    quaternion = torch.from_numpy(mat2quat(matrix[:, 0 : 3])).float().view(1, 4)
    translation = torch.from_numpy(matrix[:, 3]).contiguous().float().view(1, 3)
    pose = torch.cat((translation, quaternion), 1)
    return pose