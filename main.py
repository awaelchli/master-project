import argparse
import os
import torch
import time
import shutil
from base import BaseExperiment
from binarypose import BinaryPoseCNN
from poseConvLSTM import KITTIPoseConvLSTM
from binaryposeConvLSTM import BinaryPoseConvLSTM
from binaryposeflow import BinaryPose
from base import CHECKPOINT_FILENAME


OUT_ROOT_FOLDER = 'out'
ARCHS = {
    # 'cnnlstm': BaseExperiment,
    'binarypose': BinaryPoseCNN,
    'kitticonvlstm': KITTIPoseConvLSTM,
    'binaryposeconvlstm': BinaryPoseConvLSTM,
    'binaryflownet': BinaryPose
}


def setup_environment():
    os.makedirs(OUT_ROOT_FOLDER, exist_ok=True)


def reset_folder(folder):
    # Wipe all existing data
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def get_main_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='Frequency of printed information during training')

    parser.add_argument('--name', type=str, default='unnamed',
                        help='Name of the experiment. Output files will be stored in this folder.')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)

    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_FILENAME,
                        help='The filename of the checkpoint. Only the name of the file must be given, not the path.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    return parser


def print_cuda_status():
    if torch.cuda.is_available():
        print('CUDA is available on this machine.')
    else:
        print('CUDA is not available on this machine.')


def print_elapsed_hours(elapsed):
    print('Elapsed time: {:.4f} hours.'.format(elapsed / 3600))


if __name__ == '__main__':
    print_cuda_status()
    setup_environment()

    main_parser = get_main_parser()
    subparsers = main_parser.add_subparsers()

    for name, c in ARCHS.items():
        sub = subparsers.add_parser(name, parents=[main_parser])
        c.submit_arguments(sub)
        sub.set_defaults(create=c)

    args = main_parser.parse_args()
    print(args)

    experiment_folder = os.path.join(OUT_ROOT_FOLDER, args.name)
    if args.train and not args.resume:
        reset_folder(experiment_folder)

    experiment = args.create(experiment_folder, args)

    start_epoch = 1
    checkpoint = None
    if args.resume:
        print('Loading checkpoint ...')
        checkpoint = experiment.load_checkpoint()
        experiment.restore_from_checkpoint(checkpoint)
        start_epoch = checkpoint['epoch'] + 1

    if args.train:
        print('Trainingset: {} samples'.format(experiment.trainingset_size))
        print('Validationset: {} samples'.format(experiment.validationset_size))
        print('Training for {} epochs ...'.format(args.epochs))
        start_time = time.time()

        for epoch in range(start_epoch, start_epoch + args.epochs):
            # Train for one epoch
            print('Epoch [{:d}/{:d}]'.format(epoch, start_epoch + args.epochs - 1))
            experiment.train()
            experiment.save_checkpoint(experiment.make_checkpoint())
            experiment.plot_performance()

        print(print_elapsed_hours(time.time() - start_time))

    if args.test:
        print('Testset: {} samples'.format(experiment.testset_size))
        print('Testing ...')
        start_time = time.time()
        checkpoint = experiment.load_checkpoint()
        experiment.restore_from_checkpoint(checkpoint)
        experiment.test()
        print(print_elapsed_hours(time.time() - start_time))
