from transforms3d.euler import euler2quat
from transforms3d.quaternions import rotate_vector, qinverse, qmult
import numpy as np


def relative_to_first_pose(translations, quaternions):
    """ Makes all translations and quaternions in the list relative to the first translation/quaternion. """

    t1 = translations[0]
    q1_inv = qinverse(quaternions[0])

    rel_translations = [rotate_vector(t - t1, q1_inv) for t in translations]
    rel_quaternions = [qmult(q1_inv, q) for q in quaternions]

    return np.array(rel_translations), np.array(rel_quaternions)