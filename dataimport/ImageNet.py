from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import os
import re
from PIL import Image
from skimage.transform import ProjectiveTransform, rescale
from math import cos, sin, radians, floor
from random import shuffle

FOLDERS = {
    'training': '../data/ImageNet/ILSVRC2012/train',
    'validation': '../data/ImageNet/ILSVRC2012/val',
    'test': '../data/ImageNet/ILSVRC2012/test',
}

INTERPOLATION = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC]
EXTENSIONS = ['jpeg', 'jpg', 'png']


class PoseGenerator(Dataset):

    def __init__(self, root, max_angle=45.0, z_plane=1.0, transform1=None, transform2=None, max_size=None):
        self.root = root
        self.max_angle = max_angle
        self.z_plane = z_plane
        self.transform1 = transform1
        self.transform2 = transform2
        self.filenames = find_images(root)
        if max_size and max_size < len(self.filenames):
            shuffle(self.filenames)
            self.filenames = self.filenames[:max_size]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert('RGB')
        if self.transform1:
            image = self.transform1(image)

        angle, target = random_pose(self.max_angle)
        image, hom = self.homography_transform(image, angle)
        # Rescale image such that no pixel is scaled up
        image = compensate_homography_scale(image, hom)

        if self.transform2:
            image = self.transform2(image)

        return image, target

    def homography_transform(self, image, angle):
        w, h = image.width, image.height
        # Homography that rotates the image at a given depth
        hom = homography_roty(angle, w, h, self.z_plane)
        image = apply_homography(image, hom)
        return image, hom

    def visualize_sample_transforms(self, index, output_folder):
        image = Image.open(self.filenames[index]).convert('RGB')
        if self.transform1:
            image = self.transform1(image)

        save_image(image, '{}-original'.format(index), 0, 1, output_folder)

        angle, target = random_pose(self.max_angle)
        image, hom = self.homography_transform(image, angle)
        image = compensate_homography_scale(image, hom)

        save_image(image, '{}-homography'.format(index), angle, target, output_folder)

        image = compensate_homography_scale(image, hom)

        save_image(image, '{}-rescale'.format(index), angle, target, output_folder)

        if self.transform2:
            image = self.transform2(image)

        save_image(image, '{}-post-transform'.format(index), angle, target, output_folder, is_torch_tensor=True)


def save_image(image, basename, angle, label, output_folder, is_torch_tensor=False):
    fname = '{}-angle={:.2f}-label={}.png'.format(basename, angle, label)

    if is_torch_tensor:
        tf = transforms.ToPILImage()
        image = tf(image)

    image.save(os.path.join(output_folder, fname))


def find_images(rootdir):
    # Match filenames with image extension
    regex = re.compile(r'\.({})$'.format('|'.join(EXTENSIONS)), re.IGNORECASE)

    file_list = []
    for root, dirs, files in os.walk(rootdir):
        l = [os.path.join(root, f) for f in files if re.search(regex, f)]
        file_list.extend(l)

    return file_list


def random_pose(angle):
    left = random.uniform(0, 1) < 0.5
    angle = -angle if left else angle
    pose = 0 if left else 1
    return angle, pose


def random_pose_range(total_angle):
    angle = random_angle(total_angle)
    pose = 0 if angle < 0 else 1
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


