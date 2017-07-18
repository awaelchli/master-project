from transforms3d import axangles
import torch
from math import radians


def axis_angle_to_matrix(axis, angle):
    rot = axangles.axangle2mat(axis.numpy(), angle)
    return torch.from_numpy(rot)


def to_affine(rotation, translation):
    return torch.cat((rotation, translation), 1)


def from_affine(matrix):
    return matrix[0:3, 0:3], matrix[0:3, 3]


def x_axis():
    return torch.FloatTensor([1, 0, 0])


def y_axis():
    return torch.FloatTensor([0, 1, 0])


def z_axis():
    return torch.FloatTensor([0, 0, 1])


def rotX(angle):
    return axis_angle_to_matrix(x_axis(), radians(angle))


def rotY(angle):
    return axis_angle_to_matrix(y_axis(), radians(angle))


def rotZ(angle):
    return axis_angle_to_matrix(z_axis(), radians(angle))