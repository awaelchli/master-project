import torch
from math import pi
from math import radians
from math import tan
from time import sleep
import random
import matplotlib.pyplot as plt
from utils import to_relative_poses, matrix_to_quaternion_pose_vector


def distribute_points_on_sphere(num_points):
    points = torch.zeros(4, num_points)

    # Z coordinate
    z = points[2].uniform_(-1, 1)

    # X- and Y coordinate
    theta = torch.zeros(num_points).uniform_(0, 2 * pi)
    points[0] = torch.sqrt(1 - z ** 2) * torch.cos(theta)
    points[1] = torch.sqrt(1 - z ** 2) * torch.sin(theta)

    # W coordinate (homogeneous)
    points[3].fill_(1)

    return points


def camera_matrix(position=(0, 0, 0), up=(0, 1, 0), look_at=(0, 0, 1)):
    pos = torch.Tensor(position).view(-1, 1)
    up = torch.Tensor(up).view(-1, 1)
    look = torch.Tensor(look_at).view(-1, 1)

    z = (pos - look).renorm(2, 1, 1)
    x = torch.cross(up, z).renorm(2, 1, 1)
    y = torch.cross(z, x).renorm(2, 1, 1)

    mat = torch.zeros(4, 4)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = pos
    mat[3, 3] = 1

    return mat.inverse()


def projection_matrix(fov, aspect):
    fov = radians(fov)

    mat = torch.zeros(4, 4)

    mat[0, 0] = 1 / (aspect * tan(fov / 2))
    mat[1, 1] = 1 / tan(fov / 2)
    mat[2, 2] = 1 # (near + far) / (near - far)
    mat[2, 3] = -2# 2 * near * far / (near - far)
    mat[3, 2] = -1

    return mat


def translation_matrix(vector):
    t = torch.eye(4, 4)
    t[0:3, 3] = torch.Tensor(vector)
    return t


def homogeneous_division(points):
    points[0] /= points[3]
    points[1] /= points[3]
    points[2] /= points[3]
    points[3].fill_(1)
    return points


def viewport_matrix():
    pass


def project_points_to_screen(points, transform):
    return homogeneous_division(torch.mm(transform, points))


def animate_z_translation(init_camera_matrix, projection_matrix, frames=20, num_points=50):
    points = distribute_points_on_sphere(num_points)

    feature_tracks = torch.zeros(frames, num_points, 2)
    matrices = []
    c = init_camera_matrix
    z_step = 0.2

    for i in range(frames):

        if random.uniform(0, 1) < 0.5:
            z_step *= -1

        # Apply the new matrix after translation
        c = torch.mm(translation_matrix((0, 0, z_step)), c)
        #print(c[2, 3])
        transform = torch.mm(projection_matrix, c)
        proj_points = project_points_to_screen(points, transform)

        # Collect 2D points for each frame
        feature_tracks[i, :, 0] = proj_points[0]
        feature_tracks[i, :, 1] = proj_points[1]

        # Collect matrices to convert the pose later
        matrices.append(c[:3])

    # Convert all matrices to relative pose
    matrices = to_relative_poses(matrices)
    poses = [matrix_to_quaternion_pose_vector(m) for m in matrices]
    poses = torch.cat(poses, 0)

    assert feature_tracks.size(0) == frames
    assert feature_tracks.size(1) == num_points
    assert feature_tracks.size(2) == 2
    assert poses.size(0) == frames

    # Output shape for feature tracks: [frames, num_points, 2]
    # Output shape for poses: [frames, 7]
    return feature_tracks, poses


if __name__ == '__main__':

    c = camera_matrix(position=(0, 0, 5), look_at=(0, 0, -10))
    p = projection_matrix(60, 1)
    feature_tracks, _ = animate_z_translation(c, p, frames=20, num_points=50)

    plt.ion()
    for i in range(20):

        plt.clf()
        x = feature_tracks[i, :, 0].numpy()
        y = feature_tracks[i, :, 1].numpy()
        plt.axis('equal')
        plt.axis([-1, 1, -1, 1])
        plt.scatter(x, y)
        plt.show()
        plt.pause(0.01)

