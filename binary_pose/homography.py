import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp, warp_coords
from skimage import data
from skimage.transform import ProjectiveTransform
from math import cos, sin, radians, pi, tan, atan
from matplotlib.widgets import Slider


def apply_homography(image, homography=np.eye(3, 3), interpolation_order=0):
    h = np.linalg.inv(homography)
    warped = warp(image, ProjectiveTransform(h), order=interpolation_order)
    return warped


def homography_roty(angle, w, h, z=1):
    theta = radians(angle)

    to3D = np.array([[1/w,  0,    -0.5],
                     [0,    1/h,  -0.5],
                     [0,    0,       0],
                     [0,    0,       1]])

    to2D = np.array([[w,    0,  w/2,   0],
                     [0,    h,  h/2,   0],
                     [0,    0,    1,   0]])

    t = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])

    rot = np.array([[cos(theta), 0, sin(theta), 0],
                    [0, 1, 0, 0],
                    [-sin(theta), 0, cos(theta), 0],
                    [0, 0, 0, 1]])

    return to2D.dot(t).dot(rot).dot(to3D)


# def homography_scale(s, w, h):
#     shift_center = np.array([[1, 0, -w/2],
#                              [0, 1, -h/2],
#                              [0, 0, 1]])
#
#     scale = np.array([[s, 0, 0],
#                       [0, s, 0],
#                       [0, 0, 1]])
#
#     inv_shift_center = np.array([[1, 0, w/2],
#                                  [0, 1, h/2],
#                                  [0, 0, 1]])
#
#     return np.matmul(np.matmul(inv_shift_center, scale), shift_center)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    hom = homography_roty(angle, w, h)
    #warp_corners(hom, w, h)
    return apply_homography(image, hom)


def warp_corners(homography, w, h):
    corners = np.array([[0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]])
    t = ProjectiveTransform(homography)
    warped_corners = t(corners)
    print(warped_corners)


# def determine_scale(hom, w, h):
#     points = np.array([[0, 0],
#                     [0, 1],
#                     [1, 1]])
#
#     transformed = np.matmul(hom, points)
#     transformed[0, :] /= transformed[2, :]
#     transformed[1, :] /= transformed[2, :]
#     transformed[2, :] = np.ones((1, 2))
#
#     scale = np.linalg.norm(transformed[:, 0] - transformed[:, 1], 2)
#     return scale


# def rotate_and_scale(image, angle):
#     h, w = image.shape
#     hom1 = homography_roty(angle, w, h)
#     s = determine_scale(hom1, w, h)
#     hom2 = homography_scale(s, w, h)
#     hom = np.matmul(hom2, hom1)
#     hom = np.linalg.inv(hom)
#     return warp(image, ProjectiveTransform(hom))


def update(val):
    ax1.imshow(rotate_image(image, val), cmap=plt.cm.gray)

image = data.camera()

fig, (ax1, ax3) = plt.subplots(1, 2)

ax1.imshow(rotate_image(image, 0), cmap=plt.cm.gray)

slider = Slider(ax3, 'Angle', -90, 90, valinit=0)
slider.on_changed(update)

plt.show()


