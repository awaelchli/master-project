from dataimport import utils
import os
import argparse


def main():
    print('Converting poses...')
    utils.convert_pose_files(args.pose_dir, args.pose_dir.rstrip(os.sep) + '_converted')
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_dir', type=str, default='../data/KITTI odometry/poses/',
                        help='Original pose folder from KITTI containing ground truth sequence poses.')

    args = parser.parse_args()
    main()