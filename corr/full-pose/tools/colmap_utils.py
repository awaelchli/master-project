import os
import numpy as np


def read_colmap_predictions(file):
    # Format of COLMAP file "images.txt":
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    # POINTS2D[] as (X, Y, POINT3D_ID)

    with open(file, 'r') as f:
        lines = f.readlines()

    # First four lines are comments
    lines = lines[4:]

    # Every other line contains feature points (not needed)
    assert len(lines) % 2 == 0
    lines = lines[::2]

    # Convert every line to a tuple
    # Format: qw, qx, qy, qz, tx, ty, tz, timestamp
    lines = [line.split() for line in lines]
    lines = [
        (float(s[1]), float(s[2]), float(s[3]), float(s[4]),    # Quaternion
         float(s[5]), float(s[6]), float(s[7]),                 # Position
         int(os.path.splitext(s[9])[0])                         # Timestamp
        )
        for s in lines
    ]

    # Sort according to filename/timestamp
    def timestamp_key(tuple):
        return tuple[-1]

    lines.sort(key=timestamp_key)

    print(lines)

    matrix = np.array(lines)
    poses = matrix[:, 0:7]
    poses = poses[:, [4, 5, 6, 0, 1, 2, 3]]
    times = matrix[:, 7]

    # Row format of poses:
    # tx, ty, tz, qw, qx, qy, qz
    return times, poses



times, poses = read_colmap_predictions(r'C:\Users\Adrian\Desktop\testsequence2\images.txt')
print(poses)
print(times)