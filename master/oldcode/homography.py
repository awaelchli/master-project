import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import warp, warp_coords
from skimage import data
from skimage import util
from skimage.transform import ProjectiveTransform, rescale
from math import cos, sin, radians, pi, tan, atan, ceil, floor
from matplotlib.widgets import Slider


def apply_homography(image, homography=np.eye(3, 3), order=0):
    h = np.linalg.inv(homography)
    warped = warp(image, ProjectiveTransform(h), order=order)
    return warped


def homography_roty(angle, w, h, z=1.0):
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


def rotate_image(image, angle, z=1.0, scale=1.0, interp=0):
    h, w = image.shape[:2]
    hom = homography_roty(angle, w, h, z)
    warped = apply_homography(image, hom, order=interp)

    scale = 1 / determine_pixel_scale(hom, w, h)
    resized = rescale(warped, scale, order=interp)

    #points = warp_corners(hom, w, h)
    #xmin, xmax, ymin, ymax = inner_bounding_box(points, w, h)
    #s = util.crop(image, (ymin, ymax), (xmin, xmax), copy=True)
    return resized


def warp_corners(homography, w, h):
    corners = np.array([[0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]])
    t = ProjectiveTransform(homography)
    warped_corners = t(corners)
    return tuple(tuple(r) for r in warped_corners)


# def inner_bounding_box(points, w, h):
#     """
#     :param points: list of tuples of four points in clockwise order startin from the top-left corner.
#     :param w: width
#     :param h: height
#     :return:
#     """
#     assert len(points) == 4
#
#     xs = [p[0] for p in points]
#     ys = [p[1] for p in points]
#
#     x_min = ceil(max(min(xs), 0))
#     y_min = ceil(max(min(ys), 0))
#     x_max = floor(min(max(xs), w))
#     y_max = floor(min(max(ys), h))
#     print(x_min, x_max, y_min, y_max)
#     return x_min, x_max, y_min, y_max

def determine_pixel_scale(hom, w, h):
    transform = ProjectiveTransform(hom)
    points = np.array([[0, 0],
                       [0, h],
                       [w, 0],
                       [w, h]])

    new_points = transform(points)
    print(new_points)
    scale_left = abs(new_points[0, 1] - new_points[1, 1]) / h
    scale_right = abs(new_points[2, 1] - new_points[3, 1]) / h

    print(scale_left)
    print(scale_right)
    return max(scale_left, scale_right)


# def rotate_and_scale(image, angle):
#     h, w = image.shape
#     hom1 = homography_roty(angle, w, h)
#     s = determine_scale(hom1, w, h)
#     hom2 = homography_scale(s, w, h)
#     hom = np.matmul(hom2, hom1)
#     hom = np.linalg.inv(hom)
#     return warp(image, ProjectiveTransform(hom))

if __name__ == '__main__':

    def update(val):
        rotated = rotate_image(image, val, z=1)
        ax1.imshow(rotated, cmap=plt.cm.gray)

    image = data.camera()

    fig, (ax1, ax3) = plt.subplots(1, 2)

    ax1.imshow(rotate_image(image, 0), cmap=plt.cm.gray)

    slider = Slider(ax3, 'Angle', -90, 90, valinit=0)
    slider.on_changed(update)

    plt.show()


