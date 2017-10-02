from pyquaternion import Quaternion
import numpy as np

rot = np.eye(3)
print(rot)

q = Quaternion(matrix=rot)
print(q)

print(q.axis)
print(q.angle)

q = Quaternion(axis=[2,0,0], angle=0.1)
print(q)
print(q.axis)