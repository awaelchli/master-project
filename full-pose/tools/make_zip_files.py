from zipfile import ZipFile
import os
import glob
import argparse


def zip_all_folders(data_root, pose_root, destination):
    folders = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
    folders = [os.path.join(data_root, name) for name in folders]

    pose_files = glob.glob(os.path.join(pose_root, '*.txt'))

    assert len(folders) == len(pose_files)

    folders.sort()
    pose_files.sort()

    print('Found {} folders.'.format(len(folders)))

    for i, (folder, pose_file) in enumerate(zip(folders, pose_files)):
        out_name = '{}.zip'.format(os.path.basename(folder))
        out_name = os.path.join(destination, out_name)

        print('({:02d}) Zipping "{}" and pose file "{}" to "{}"'.format(
            i, os.path.basename(folder), os.path.basename(pose_file), out_name
        ))

        zip_one_folder(folder, pose_file, out_name)


def zip_one_folder(folder, pose_file, out_name):

    with ZipFile(out_name, 'w') as archive:
        def time_from_filename(x):
            return int(os.path.splitext(os.path.basename(x))[0])

        files = os.listdir(folder)
        files.sort(key=time_from_filename)

        for f in files:
            p = os.path.join(folder, f)
            archive.write(p, arcname=os.path.basename(f))

        # Add the pose file
        archive.write(pose_file, arcname='poses.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make Zip Files')
    parser.add_argument('--data', type=str, default=os.path.join('.', 'data'),
                        help='Root folder that contains all the subfolders for each video sequence.')
    parser.add_argument('--poses', type=str, default=os.path.join('.', 'poses'),
                        help='Root folder of the pose text files.')
    parser.add_argument('--dest', type=str, default='.',
                        help='Output directory for the zip files. Default is the current directory.')

    args = parser.parse_args()

    data_root = args.data
    pose_root = args.poses
    destination = args.dest

    if not os.path.isdir(destination):
        os.makedirs(destination)

    zip_all_folders(data_root, pose_root, destination)
