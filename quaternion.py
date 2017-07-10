from pyquaternion import Quaternion
import numpy as np

rot = np.eye(3)
print(rot)

q = Quaternion(matrix=rot)
print(q)