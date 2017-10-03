import os.path
import random
import glob
import math
from .listdataset import ListDataset
from scipy.ndimage import imread
import numpy as np
import flow_transforms

'''
Dataset routines for KITTI_flow, 2012 and 2015.
http://www.cvlibs.net/datasets/kitti/eval_flow.php
The dataset is not very big, you might want to only pretrain on it for flownet
EPE are not representative in this dataset because of the sparsity of the GT.
'''

def load_flow_from_png(png_path):
    return(imread(png_path)[:,:,0:2].astype(float) - 128)

def make_dataset(dir, occ=True, split = 0):
    '''Will search in training folder for folders 'flow_noc' or 'flow_occ' and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
    flow_dir = 'flow_occ' if occ else 'flow_noc'
    assert(os.path.isdir(os.path.join(dir,flow_dir)))
    img_dir = 'colored_0'
    if not os.path.isdir(os.path.join(dir,img_dir)):
        img_dir = 'image_2'
    assert(os.path.isdir(os.path.join(dir,img_dir)))

    images = []
    for flow_map in glob.iglob(os.path.join(dir,flow_dir,'*.png')):
        flow_map = os.path.basename(flow_map)
        root_filename = flow_map[:-7]
        flow_map = os.path.join(flow_dir,flow_map)
        img1 = os.path.join(img_dir,root_filename+'_10.png')
        img2 = os.path.join(img_dir,root_filename+'_11.png')
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue
        images.append([[img1,img2],flow_map])

    assert(len(images) > 0)
    random.shuffle(images)
    split_index = math.floor(len(images)*split/100)
    assert(split_index >= 0 and split_index <= len(images))
    return (images[:split_index], images[split_index+1:]) if split_index < len(images) else (images, [])

def KITTI_loader(root,path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)
    return [imread(img) for img in imgs],load_flow_from_png(flo)

def KITTI_occ(root, transform=None, target_transform=None,
                 co_transform=None, split = 80):
    train_list, test_list = make_dataset(root,True,split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform, loader=KITTI_loader)
    test_dataset = ListDataset(root, test_list, transform, target_transform, flow_transforms.CenterCrop((320,1216)), loader=KITTI_loader)

    return train_dataset, test_dataset

def KITTI_noc(root, transform=None, target_transform=None,
                 co_transform=None, split = 80):
    train_list, test_list = make_dataset(root,False,split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform, loader=KITTI_loader)
    test_dataset = ListDataset(root, test_list, transform, target_transform, flow_transforms.CenterCrop((320,1216)), loader=KITTI_loader)

    return train_dataset, test_dataset