#from transforms3d.euler import euler2axangle
from math import radians
from scipy import interpolate
import numpy as np


def read_from_text_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = [s.split() for s in lines]

    times = np.array([float(line[1]) for line in lines])
    poses = np.array([[float(x) for x in line[2:8]] for line in lines])

    return times, poses


pitch = radians(0)
roll = radians(0)
yaw = radians(0)

# Y axis is inverted
#axis, angle = euler2axangle(-yaw, pitch, roll, 'ryxz')
times, poses = read_from_text_file('camera-track.txt')

interpolator = interpolate.interp1d(times, poses, kind='nearest', axis=0, copy=True, bounds_error=True, assume_sorted=True)

print(interpolator(910))
print(interpolator(1100))
print(interpolator(1008))