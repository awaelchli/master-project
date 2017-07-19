from dataimport import utils
import os
from dataimport.KITTI import convert_folder
import argparse


def main():
    print('Converting folder...')
    convert_folder(sequence_dir=args.sequence_dir, pose_dir=args.pose_dir,
                   new_root=args.new_root, new_sequence_length=args.length)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence_dir', type=str, default='../data/KITTI odometry/grayscale/sequences',
                        help='')
    parser.add_argument('pose_dir', type=str, default='../data/KITTI odometry/poses/',
                        help='Original pose folder from KITTI containing ground truth sequence poses.')
    parser.add_argument('new_root', type=str, default='../data/KITTI odometry/converted/',
                        help='')
    parser.add_argument('--length', type=int, default=10,
                        help='')

    args = parser.parse_args()
    main()