import argparse
import os
import torch
import time
from base import BaseExperiment, OUT_ROOT_FOLDER
from binarypose import BinaryPoseCNN


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
    ARCHS = {
        #'cnnlstm': BaseExperiment,
        'binarypose': BinaryPoseCNN,
    }

    os.makedirs(OUT_ROOT_FOLDER, exist_ok=True)

    main_parser = get_main_parser()
    subparsers = main_parser.add_subparsers()

    for name, c in ARCHS.items():
        sub = subparsers.add_parser(name, parents=[main_parser])
        c.submit_arguments(sub)
        sub.set_defaults(create=c)

    args = main_parser.parse_args()
    print(args)
    experiment = args.create(args)

    print_cuda_status()

    start_epoch = 1
    checkpoint = None
    if args.resume:
        print('Loading checkpoint ...')
        checkpoint = experiment.load_checkpoint()
        start_epoch = checkpoint['epoch'] + 1

    if args.train:
        print('Training for {} epochs ...'.format(args.epochs))
        start_time = time.time()

        for epoch in range(start_epoch, start_epoch + args.epochs):
            print('Epoch [{:d}/{:d}]'.format(epoch, start_epoch + args.epochs - 1))
            # Train for one epoch
            experiment.train(checkpoint)

        print(print_elapsed_hours(time.time() - start_time))
        # TODO: plotting
        #plots.plot_epoch_loss(train_loss, validation_loss, save=self.save_loss_plot)

    if args.test:
        print('Testing ...')
        start_time = time.time()
        checkpoint = experiment.load_checkpoint()
        experiment.test(checkpoint)
        print(print_elapsed_hours(time.time() - start_time))
