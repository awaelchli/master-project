import argparse
import glob
import os
import os.path as path

from GTAV_old import plot_camera_path_2D


def get_text_files(folder):
    return glob.glob(path.join(folder, '*.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./',
                        help='The folder that contains the camera track files.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder for the plots. By default, this is equal to the input folder.')

    args = parser.parse_args()

    if not args.output:
        args.output = args.folder

    files = get_text_files(args.folder)
    print('Found {} text files in {}'.format(len(files), args.folder))

    for filename in files:
        out = os.path.basename(os.path.splitext(filename)[0])
        out = os.path.join(args.output, '{}.svg'.format(out))

        print('Saving plot as {}'.format(out))

        plot_camera_path_2D(filename, resolution=1, show_rot=False, output=out)

    print('Done.')
