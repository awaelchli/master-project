import shutil
import os.path as path
import os
import glob

main_folder = '/mnt/SSDdata/adrian/Datasets/GTAV/test'
data_subfolder = 'data'
pose_subfolder = 'poses'


image_folders = os.listdir(path.join(main_folder, data_subfolder))
pose_filenames = os.listdir(path.join(main_folder, pose_subfolder))
pose_filenames = [fn for fn in pose_filenames if fn.endswith('.txt')]

image_folders.sort()
pose_filenames.sort()


nF = len(image_folders)
nP = len(pose_filenames)
assert nF == nP, 'Number of folders ({:d}) does not match number of pose files {:d}'.format(nF, nP)

for i, (f, p) in enumerate(zip(image_folders, pose_filenames)):

    old_folder_filename = path.join(main_folder, data_subfolder, f)
    old_pose_filename = path.join(main_folder, pose_subfolder, p)

    new_folder_filename = path.join(main_folder, data_subfolder, '{:03d}'.format(i))
    new_pose_filename = path.join(main_folder, pose_subfolder, '{:03d}.txt'.format(i))

    shutil.move(old_folder_filename, new_folder_filename)
    shutil.move(old_pose_filename, new_pose_filename)



