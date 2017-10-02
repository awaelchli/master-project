import os.path
import random
import glob
import math
from .listdataset import ListDataset
from scipy.ndimage import imread
import numpy as np
import flow_transforms

'''
Dataset routines for MPI Sintel.
http://sintel.is.tue.mpg.de/
clean version imgs are without shaders, final version imgs are fully rendered
The dataset is not very big, you might want to only pretrain on it for flownet
'''

def make_dataset(dir, dataset_type = 'clean', split = 0, shuffle = True):
    training_dir = os.path.join(dir,'training')
    flow_dir = 'flow'
    assert(os.path.isdir(os.path.join(dir,flow_dir)))
    
    img_dir = dataset_type
    assert(os.path.isdir(os.path.join(dir,img_dir)))

    images = []
    for flow_map in glob.iglob(os.path.join(dir,flow_dir,'*','*.flo')):
        flow_map = os.path.relpath(flow_map,os.path.join(dir,flow_dir))
        root_filename = flow_map[:-8]
        frame_nb = int(flow_map[-8:-4])
        img1 = os.path.join(img_dir,root_filename+str(frame_nb).zfill(4)+'.png')
        img2 = os.path.join(img_dir,root_filename+str(frame_nb+1).zfill(4)+'.png')
        flow_map = os.path.join(flow_dir,flow_map)
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue
        images.append([[img1,img2],flow_map])

    assert(len(images) > 0)
    if shuffle:
        random.shuffle(images)
    split_index = math.floor(len(images)*split/100)
    assert(split_index >= 0 and split_index <= len(images))
    return images[:split_index], images[split_index+1:]

def mpi_sintel_clean(root, transform=None, target_transform=None,
                 co_transform=None, split = 80):
    train_list, test_list = make_dataset(root,'clean',split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset, test_dataset

def mpi_sintel_final(root, transform=None, target_transform=None,
                 co_transform=None, split = 80):
    train_list, test_list = make_dataset(root,'final',split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset, test_dataset

def mpi_sintel_both(root, transform=None, target_transform=None,
                 co_transform=None, split = 80):
    '''load images from both clean and final folders.
    We cannot shuffle input, because it would very likely cause data snooping
    for the clean and final frames are not that different'''
    train_list1, test_list1 = make_dataset(root,'clean',split,False)
    train_list2, test_list2 = make_dataset(root,'final',split,False)
    train_dataset = ListDataset(root, train_list1 + train_list2, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list1 + test_list2, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset, test_dataset

