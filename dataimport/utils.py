import torch
from transforms3d.quaternions import mat2quat, quat2axangle


def matrix_to_pose_vector(matrix):
    quaternion = mat2quat(matrix[:, [0, 1, 2]])
    axis, angle = quat2axangle(quaternion)

    translation = torch.from_numpy(matrix[:, 3]).contiguous().view(1, 3)
    orientation = torch.from_numpy(axis[[0, 1]]).view(1, 2)
    angle = torch.DoubleTensor([[angle]])

    pose = torch.cat((translation, orientation, angle), 1)

    return pose
