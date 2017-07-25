import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp
from skimage import data
from skimage.transform import ProjectiveTransform
from math import cos, sin, radians, pi


def homography_roty(angle):
    theta = radians(angle)

    f = 1 # focal length

    K = np.array([[f * width, 0, width / 2],
                  [0, f * height, height / 2],
                  [0, 0, 1]])

    K_inv = np.linalg.inv(K)

    rotY = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [sin(theta), 0, 1]])

    matrix = np.matmul(np.matmul(K, rotY), K_inv)
    print(matrix)
    return matrix


image = data.camera()
height, width = image.shape

warped1 = warp(image, ProjectiveTransform(matrix=homography_roty(10)))
warped2 = warp(image, ProjectiveTransform(matrix=homography_roty(20)))
warped3 = warp(image, ProjectiveTransform(matrix=homography_roty(45)), output_shape=(1*height, 1*width))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.imshow(warped1, cmap=plt.cm.gray, interpolation='nearest')
ax2.imshow(warped2, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(warped3, cmap=plt.cm.gray, interpolation='nearest')
plt.show()



def camera_matrix(f, width, height):
    K = np.array([[f * width, 0, width/2, 0],
                     [0, f * height, height/2, 0],
                     [0, 0, 1, 0]])

    return K


def rot_plane(z, angle):
    theta = radians(angle)
    t = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, -z],
                  [0, 0, 0, 1]])

    rot = np.array([[cos(theta), 0, sin(theta), 0],
                    [0, 1, 0, 0],
                    [-sin(theta), 0, cos(theta), 0],
                    [0, 0, 0, 1]])

    t_inv = np.linalg.inv(t)

    return np.matmul(np.matmul(t_inv, rot), t)




rot = rot_plane(2, 10)

