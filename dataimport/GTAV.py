from transforms3d.euler import euler2axangle, euler2quat
from transforms3d.quaternions import rotate_vector, qinverse, qmult
from math import radians
from scipy import interpolate
import numpy as np
import os
from os import path
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def build_index(root_folder, ground_truth_folder):
    sequence_dirs = [path.join(root_folder, d) for d in os.listdir(root_folder)
                     if path.isdir(path.join(root_folder, d))]

    pose_files = glob.glob(os.path.join(ground_truth_folder, '*.txt'))

    sequence_dirs.sort()
    pose_files.sort()

    index = [build_index_for_sequence(sequence_dir, pose_file)
             for pose_file, sequence_dir in zip(pose_files, sequence_dirs)]

    return index


def build_index_for_sequence(sequence_dir, pose_file):
    all_times, all_poses = read_from_text_file(pose_file)
    interpolator = interpolate.interp1d(all_times, all_poses, kind='nearest', axis=0, copy=True, bounds_error=True,
                                        assume_sorted=True)

    # Sort filenames according to time given in filename
    filenames = [path.join(sequence_dir, f) for f in os.listdir(sequence_dir)]
    time_from_filename = lambda x: int(path.splitext(path.basename(x))[0])
    filenames.sort(key=time_from_filename)

    # Interpolate at times given by filename
    query_times = [time_from_filename(f) for f in filenames]
    poses_interpolated = [interpolator(t) for t in query_times]

    return filenames, poses_interpolated


def read_from_text_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = [s.split() for s in lines]

    times = np.array([float(line[1]) for line in lines])
    poses = np.array([[float(x) for x in line[2:8]] for line in lines])

    return times, poses


def get_translations(pose_array):
    return pose_array[:, 0:3]


def get_quaternions(pose_array):
    angles = pose_array[:, 3:6]

    # Order saved in file: pitch roll yaw
    # Apply rotations in order: yaw, pitch, roll
    quaternions = [euler2quat(radians(eul[2]), radians(eul[0]), radians(eul[1]), 'rzxy') for eul in angles]
    return np.array(quaternions)


def get_camera_optical_axes(quaternions):
    return np.array([rotate_vector([0, 1, 0], q) for q in quaternions])


def to_relative_pose_quat(translations, quaternions):
    t1 = translations[0]
    q1_inv = qinverse(quaternions[0])

    rel_translations = []
    rel_quaternions = []
    for t, q in zip(translations, quaternions):
        t_rel = rotate_vector(t - t1, q1_inv)
        q_rel = qmult(q1_inv, q)

        rel_translations.append(t_rel)
        rel_quaternions.append(q_rel)

    return np.array(rel_translations), np.array(rel_quaternions)


def plot_camera_path_2D(file, resolution=1.0, show_rot=True):
    assert 0 < resolution <= 1
    step = 1 / resolution

    _, poses = read_from_text_file(file)
    translations = get_translations(poses)[::step]
    quaternions = get_quaternions(poses)[::step]

    # Convert to relative pose
    translations, quaternions = to_relative_pose_quat(translations, quaternions)

    x, y, z = translations[:, 0], translations[:, 1], translations[:, 2]
    plt.plot(x, y)

    if show_rot:
        y_axes = get_camera_optical_axes(quaternions)

        u, v, w = y_axes[:, 0], y_axes[:, 1], y_axes[:, 2]

        norm2d = np.sqrt(u ** 2 + v ** 2)
        u = u / norm2d
        v = v / norm2d

        plt.quiver(x, y, u, v, units='xy', scale_units='xy', scale=0.3, width=0.05)

    plt.show()

s = r'E:\Rockstar Games\Grand Theft Auto V\08.12.2017 - 18.04.57.txt'
plot_camera_path_2D(s, 0.07)

# 3D plot:
#fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(x, y, z)
