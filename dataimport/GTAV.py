#from transforms3d.euler import euler2axangle
from math import radians
from scipy import interpolate
import numpy as np
import os
from os import path
import glob


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


r = 'D:/ShadowPlay/Grand Theft Auto V/'
print(build_index(r, r))

#axis, angle = euler2axangle(-yaw, pitch, roll, 'ryxz')
