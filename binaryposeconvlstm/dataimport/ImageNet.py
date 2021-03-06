from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
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


class DiscretePoseGenerator(Dataset):

    def __init__(self, root, max_angle=45.0, classes=3, z_plane=1.0, transform1=None, transform2=None, max_size=None):
        self.root = root
        self.max_angle = max_angle
        self.classes = classes
        self.z_plane = z_plane
        self.transform1 = transform1
        self.transform2 = transform2
        self.filenames = find_images(root)
        self.visualize = None
        if max_size and max_size < len(self.filenames):
            shuffle(self.filenames)
            self.filenames = self.filenames[:max_size]

    def visualize(self, output_folder):
        self.visualize = output_folder

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        original = Image.open(self.filenames[index]).convert('RGB')
        if self.transform1:
            original = self.transform1(original)

        angle, target = random_pose_range(self.max_angle, self.classes)
        new, hom = self.homography_transform(original, angle)
        # Rescale image such that no pixel is scaled up
        # Make original image the same size
        new_downscaled = compensate_homography_scale(new, hom)
        original_downscaled = original.resize((new_downscaled.width, new_downscaled.height), random_interpolation_method())

        original_final = self.transform2(original_downscaled) if self.transform2 else original_downscaled
        new_final = self.transform2(new_downscaled) if self.transform2 else new_downscaled

        # Optional visualization
        if self.visualize:
            save_image(original, '{}a-ORIGINAL'.format(index), 0, 1, self.visualize)
            save_image(new, '{}b-NEW'.format(index), angle, target, self.visualize)
            save_image(original_downscaled, '{}c-ORIGINAL-DOWNSCALED'.format(index), angle, target, self.visualize)
            save_image(new_downscaled, '{}d-NEW-DOWNSCALED'.format(index), angle, target, self.visualize)
            save_image(original_final, '{}e-ORIGINAL-FINAL'.format(index), angle, target, self.visualize, is_torch_tensor=True)
            save_image(new_final, '{}f-NEW-FINAL'.format(index), angle, target, self.visualize, is_torch_tensor=True)

        original_final.unsqueeze_(0)
        new_final.unsqueeze_(0)
        images = torch.cat((original_final, new_final), 0)

        return images, target

    def homography_transform(self, image, angle):
        w, h = image.width, image.height
        # Homography that rotates the image at a given depth
        hom = homography_roty(angle, w, h, self.z_plane)
        image = apply_homography(image, hom)
        return image, hom









class BinaryPoseSequenceGenerator(Dataset):

    def __init__(self, root, sequence_length=10, max_angle=45.0, step_angle=5.0, z_plane=1.0, transform1=None, transform2=None, max_size=None):
        self.root = root
        self.sequence_length=sequence_length
        self.step_angle=step_angle
        self.max_angle = max_angle
        self.z_plane = z_plane
        self.transform1 = transform1
        self.transform2 = transform2
        self.filenames = find_images(root)
        self.visualize = None
        if max_size and max_size < len(self.filenames):
            shuffle(self.filenames)
            self.filenames = self.filenames[:max_size]

    def visualize(self, output_folder):
        self.visualize = output_folder

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        original = Image.open(self.filenames[index]).convert('RGB')
        #if self.transform1:
         #   original = self.transform1(original)

        current_angle = 0 #random_angle(self.max_angle)
        direction = -1 if random.uniform(0, 1) < 0.5 else 1
        prob = 0.25

        images = [original]
        angles = []
        turns = []
        for i in range(self.sequence_length - 1):
            direction = -direction if random.uniform(0, 1) < prob else direction
            new_angle = current_angle + direction * self.step_angle
            if not (-self.max_angle <= new_angle <= self.max_angle):
                # If rotation goes outside maximum angle, turn in other direction
                direction *= -1
                new_angle = current_angle + direction * self.step_angle

            current_angle = new_angle

            new, hom = self.homography_transform(original, current_angle)
            images.append(new)
            angles.append(current_angle)
            turns.append(0 if direction < 0 else 1)

            if self.visualize:
                save_image(new, '{}a-ROTATED-{}'.format(index, i), current_angle, turns[-1], self.visualize)


        # Rescale image such that no pixel is scaled up
        # Make original image the same size
        #new_downscaled = compensate_homography_scale(new, hom)
        #original_downscaled = original.resize((new_downscaled.width, new_downscaled.height), random_interpolation_method())

        if self.transform2:
            images = [self.transform2(img) for img in images]

        if self.visualize:
            for i, im in enumerate(images[1:]):
                save_image(im, '{}a-FINAL-{}'.format(index, i), current_angle, turns[i], self.visualize, is_torch_tensor=True)

        images = [img.unsqueeze(0) for img in images]
        images = torch.cat(images, 0)
        turns = torch.LongTensor(turns)

        assert images.size(0) == turns.size(0) + 1 == self.sequence_length
        return images, turns

    def homography_transform(self, image, angle):
        w, h = image.width, image.height
        # Homography that rotates the image at a given depth
        #hom = homography_roty(angle, w, h, self.z_plane)
        hom = homography_shiftx(angle, w, h, self.z_plane)
        image = apply_homography(image, hom)
        return image, hom













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


def random_pose_range(max_angle, classes = 3):
    bins = np.linspace(-max_angle, max_angle, classes + 1)
    angle = random_angle(max_angle)
    pose = int(np.digitize(angle, bins)) - 1

    assert 0 <= pose < classes
    return angle, pose


def random_angle(theta):
    return random.uniform(-theta, theta)


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


def homography_shiftx(tx, w, h, z=1.0):
    t = np.array([[1, 0, tx],
                  [0, 1, 0],
                  [0, 0, 1]])

    return t


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


