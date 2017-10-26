import argparse
import os
import shutil
import time

import torch

from base import CHECKPOINT_FILENAME
from binarypose1Dtranslation import BinaryPose1DTranslation
from multiclass1Dtranslation import Multiclass1DTranslation

OUT_ROOT_FOLDER = 'out'
ARCHS = {
    'binarypose1Dtranslation': BinaryPose1DTranslation,
    'multiclass1Dtranslation': Multiclass1DTranslation,
}


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

    parser.add_argument('--print_freq', type=int, default=1,
                        help='Frequency of printed information during training')

    parser.add_argument('--input', type=str, default=None,
                        help='Input folder. All resources (such as checkpoint) will be read from this location.')
    parser.add_argument('--output', type=str, default='unnamed',
                        help='Output files will be stored in this folder.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the contents of the output folder if it exists without asking.')

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
    s = 'Elapsed time: {:.4f} hours.'.format(elapsed / 3600)
    print(s)
    return s


if __name__ == '__main__':
    print_cuda_status()

    main_parser = get_main_parser()
    subparsers = main_parser.add_subparsers()

    for name, c in ARCHS.items():
        sub = subparsers.add_parser(name, parents=[main_parser])
        c.submit_arguments(sub)
        sub.set_defaults(create=c)

    args = main_parser.parse_args()
    print(args)

    args.input = args.output if not args.input else args.input
    out_folder = os.path.join(OUT_ROOT_FOLDER, args.output)
    in_folder = os.path.join(OUT_ROOT_FOLDER, args.input)
    if not args.resume:
        if os.path.isdir(out_folder) and not args.overwrite:
            new_name = input('Output folder "{}" exists. Press ENTER to overwrite or type a name for a new folder.\n'
                             .format(args.output))
            args.output = new_name if new_name else args.output
            out_folder = os.path.join(OUT_ROOT_FOLDER, args.output)

        reset_folder(out_folder)

    experiment = args.create(in_folder, out_folder, args)
    experiment.print_info(args)

    start_epoch = 1
    checkpoint = None
    if args.resume:
        print('Loading checkpoint ...')
        checkpoint = experiment.load_checkpoint()
        experiment.restore_from_checkpoint(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        experiment.print_info('\n Resuming training at epoch {} \n'.format(start_epoch))

    if args.train:
        datasetinfo = 'Trainingset: {} samples\n' \
                      'Validationset: {} samples'.format(experiment.trainingset_size, experiment.validationset_size)
        print(datasetinfo)
        experiment.print_info(datasetinfo)
        print('Training for {} epochs ...'.format(args.epochs))
        start_time = time.time()

        for epoch in range(start_epoch, start_epoch + args.epochs):
            # Train for one epoch
            print('Epoch [{:d}/{:d}]'.format(epoch, start_epoch + args.epochs - 1))
            start_epoch_time = time.time()
            experiment.train()
            elapsed_epoch = time.time() - start_epoch_time
            # Save latest checkpoint + copy with labeled with epoch
            experiment.save_checkpoint(experiment.make_checkpoint())
            shutil.copy(experiment.checkpoint_file, experiment.make_output_filename('checkpoint-epoch-{:03d}.pth.tar'.format(epoch)))
            experiment.plot_performance()
            print('Elapsed time for epoch: {:.4f} hours'.format(elapsed_epoch / 3600))

        print_elapsed_hours(time.time() - start_time)

    if args.test:
        datasetinfo = 'Testset: {} samples'.format(experiment.testset_size)
        print(datasetinfo)
        experiment.print_info(datasetinfo)

        print('Testing ...')
        start_time = time.time()
        checkpoint = experiment.load_checkpoint()
        experiment.restore_from_checkpoint(checkpoint)
        experiment.test()

        print_elapsed_hours(time.time() - start_time)