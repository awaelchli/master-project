from torchvision.datasets import ImageFolder
import numpy as np
import random
from PIL import Image
from skimage.transform import ProjectiveTransform, rescale
from math import cos, sin, radians, floor


INTERPOLATION = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC]


class PoseImageNet(ImageFolder):

    def __init__(self, root, max_angle=45.0, z_plane=1.0, transform1=None, transform2=None):
        super().__init__(root, transform=transform1)
        self.max_angle = max_angle
        self.z_plane = z_plane
        self.transform2 = transform2

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        w, h = image.width, image.height

        angle, target = random_pose(self.max_angle)

        # Homography that rotates the image at a given depth
        hom = homography_roty(angle, w, h, self.z_plane)
        image = apply_homography(image, hom)

        # Rescale image such that no pixel is scaled up
        #image = compensate_homography_scale(image, hom)

        if self.transform2:
            image = self.transform2(image)

        return image, target


def random_pose(max_angle):
    angle = random_angle(max_angle)
    pose = int(angle > 0)
    return angle, pose


def random_angle(theta):
    return random.uniform(-theta / 2, theta / 2)


def random_interpolation_method():
    index = random.randint(0, len(INTERPOLATION) - 1)
    return INTERPOLATION[index]


def normalize(homography):
    return homography / homography[2, 2]


def homography2tuple(homography):
    elements = [el for el in np.nditer(homography)]
    return tuple(elements[:-1])


def apply_homography(image, homography):
    h = np.linalg.inv(homography)
    size = (image.width, image.height)
    data = homography2tuple(normalize(h))
    warped = image.transform(size, Image.PERSPECTIVE, data, random_interpolation_method())
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


def compensate_homography_scale(image, homography):
    w, h = image.width, image.height
    scale = 1 / determine_pixel_scale(homography, w, h)
    newsize = (floor(scale * image.width), floor(scale * image.height))
    resized = image.resize(newsize, random_interpolation_method())
    return resized


def determine_pixel_scale(hom, w, h):
    transform = ProjectiveTransform(hom)
    points = np.array([[0, 0],
                       [0, h],
                       [w, 0],
                       [w, h]])

    new_points = transform(points)
    scale_left = abs(new_points[0, 1] - new_points[1, 1]) / h
    scale_right = abs(new_points[2, 1] - new_points[3, 1]) / h

    return max(scale_left, scale_right)


