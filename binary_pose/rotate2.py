import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp
from skimage import data
from skimage.transform import ProjectiveTransform
from math import cos, sin, radians, pi
from matplotlib.widgets import Slider


def homography_roty(angle, w, h):
    theta = radians(angle)
    z = 1

    to3D = np.array([[1/w, 0, -0.5],
                     [0, 1/h, -0.5],
                     [0, 0,    0],
                     [0, 0,    1]])

    to2D = np.array([[w, 0, w/2, 0],
                     [0, h, h/2, 0],
                     [0, 0,   1, 0]])

    t = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])

    rot = np.array([[1/cos(theta), 0, sin(theta), 0],
                    [0, 1, 0, 0],
                    [-sin(theta), 0, cos(theta), 0],
                    [0, 0, 0, 1]])

    matrix = to2D.dot(t.dot(rot.dot(to3D)))
    matrix = to2D.dot(t).dot(rot).dot(to3D)
    return matrix


def homography_scale(s, w, h):
    shift_center = np.array([[1, 0, -w/2],
                             [0, 1, -h/2],
                             [0, 0, 1]])

    scale = np.array([[s, 0, 0],
                      [0, s, 0],
                      [0, 0, 1]])

    inv_shift_center = np.array([[1, 0, w/2],
                                 [0, 1, h/2],
                                 [0, 0, 1]])

    return np.matmul(np.matmul(inv_shift_center, scale), shift_center)


def rotate_image(image, angle):
    h, w = image.shape
    return warp(image, ProjectiveTransform(matrix=homography_roty(angle, w, h)))


def determine_scale(hom, w, h):
    points = np.array([[0, 0],
                    [0, 1],
                    [1, 1]])

    transformed = np.matmul(hom, points)
    transformed[0, :] /= transformed[2, :]
    transformed[1, :] /= transformed[2, :]
    transformed[2, :] = np.ones((1, 2))

    scale = np.linalg.norm(transformed[:, 0] - transformed[:, 1], 2)
    print(scale)
    return scale


def rotate_and_scale(image, angle):
    h, w = image.shape
    hom1 = homography_roty(angle, w, h)
    s = determine_scale(hom1, w, h)
    hom2 = homography_scale(1, w, h)
    hom = np.matmul(hom2, hom1)
    return warp(image, ProjectiveTransform(hom))


def update(val):
    ax1.imshow(rotate_and_scale(image, val), cmap=plt.cm.gray, interpolation='nearest')

image = data.camera()
height, width = image.shape

hom = homography_roty(45, width, height)
determine_scale(hom, width, height)

fig, (ax1, ax3) = plt.subplots(1, 2)

ax1.imshow(rotate_image(image, 0), cmap=plt.cm.gray, interpolation='nearest')

slider = Slider(ax3, 'Angle', -90, 90, valinit=0)
slider.on_changed(update)

plt.show()


