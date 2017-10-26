from pose_transforms import euler_to_quaternion
import torch
from math import degrees


def relative_quaternion_rotation_error(predictions, target):
    """ The input quaternions are assumed to be normalized. """
    assert predictions.size() == target.size()
    assert predictions.size(1) == target.size(1) == 4

    rel_angles = torch.acos(2 * (predictions * target).sum(1) ** 2 - 1)
    return [degrees(a) for a in rel_angles.view(-1)]


def relative_euler_rotation_error(predictions, target):
    q1 = euler_to_quaternion(predictions)
    q2 = euler_to_quaternion(target)
    return relative_quaternion_rotation_error(q1, q2)