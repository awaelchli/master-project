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
    top = tan(fov / 2)
    right = aspect * top

    width = 1
    height = 1

    mat = torch.zeros(4, 4)

    mat[0, 0] = 2 * right / width
    mat[0, 2] = right
    mat[1, 1] = 2 * top / height
    mat[1, 2] = top
    mat[2, 2] = 1
    mat[3, 3] = 1

    return mat.inverse()


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


def animate_translation(init_camera_matrix, projection_matrix, points=None, frames=20, num_points=50, max_step=0.2, p_turn=0.5):
    assert 0 <= p_turn <= 1
    assert frames > 0 and num_points > 0

    if points is None:
        points = distribute_points_on_sphere(num_points)

    feature_tracks = torch.zeros(frames, num_points, 2)
    matrices = []
    binary_pose = []
    c = init_camera_matrix

    for i in range(frames):

        step = random.uniform(0, max_step)
        if random.uniform(0, 1) < p_turn:
            step *= -1

        # Apply the new matrix after translation
        c = torch.mm(translation_matrix((step, 0, 0)), c)
        #print(c[2, 3])
        transform = torch.mm(projection_matrix, c)
        proj_points = project_points_to_screen(points, transform)

        # Collect 2D points for each frame
        feature_tracks[i, :, 0] = proj_points[0]
        feature_tracks[i, :, 1] = proj_points[1]

        # Collect matrices to convert the pose later
        matrices.append(c[:3])
        binary_pose.append(0 if step < 0 else 1)

    # Convert all matrices to relative pose
    matrices = to_relative_poses(matrices)
    poses = [matrix_to_quaternion_pose_vector(m) for m in matrices]
    poses = torch.cat(poses, 0)


    assert feature_tracks.size(0) == frames
    assert feature_tracks.size(1) == num_points
    assert feature_tracks.size(2) == 2
    assert poses.size(0) == frames

    binary_poses = torch.LongTensor(binary_pose)

    # Output shape for feature tracks: [frames, num_points, 2]
    # Output shape for poses: [frames, 7]
    # Output shape for binary_poses: [frames, 1]
    return feature_tracks, poses, binary_poses


if __name__ == '__main__':

    c = camera_matrix(position=(0, 0, 5), look_at=(0, 0, -10))
    p = projection_matrix(60, 1)
    feature_tracks, _ = animate_translation(c, p, frames=1000, num_points=8, max_step=1, p_turn=0.5)

    plt.ion()
    for i in range(20):

        plt.clf()
        x = feature_tracks[i, :, 0].numpy()
        y = feature_tracks[i, :, 1].numpy()
        plt.axis('equal')
        plt.axis([-5, 5, -5, 5])
        plt.scatter(x, y)
        plt.show()
        plt.pause(0.01)
