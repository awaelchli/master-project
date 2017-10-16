import torch
import torch.nn as nn
import cloud

import plots
from base import BaseExperiment, AverageMeter, Logger, CHECKPOINT_BEST_FILENAME
from models import MultiClass1DTranslationModel
import torch.nn.parallel
import random

class Multiclass1DTranslation(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--max_size', type=int, nargs=3, default=[10000, 100, 100],
                            help="""Clips training-, validation-, and testset at the given size. 
                            A zero signalizes that the whole dataset should be used.""")
        parser.add_argument('--sequence', type=int, default=10,
                            help='Length of sequence fed to the LSTM')
        parser.add_argument('--hidden', type=int, default=500,
                            help='Hidden size of the LSTM')
        parser.add_argument('--layers', type=int, default=3,
                            help='Number of layers in the LSTM')
        parser.add_argument('--keypoints', type=int, default=50,
                            help='Number of keypoints per frame')
        parser.add_argument('--classes', type=int, default=10,
                            help='Number of classes to quantize the translation')
        parser.add_argument('--features', type=int, default=1,
                            help='Number of features (channels) for encoding the identity of the points')

    def __init__(self, in_folder, out_folder, args):
        super(Multiclass1DTranslation, self).__init__(in_folder, out_folder, args)

        # Model
        self.model = MultiClass1DTranslationModel(
            hidden=args.hidden,
            nlayers=args.layers,
            num_features=args.features,
            classes=args.classes
        )
        params = self.model.get_parameters()

        if self.use_cuda:
            print('Moving model to GPU ...')
            self.model.cuda()

        self.optimizer = torch.optim.Adam(params, self.lr)
        self.loss_function = nn.CrossEntropyLoss()

        self.print_freq = args.print_freq
        self.sequence_length = args.sequence
        self.training_loss = []
        self.validation_loss = []

        self.train_logger.column('Epoch', '{:d}')
        self.train_logger.column('Training Loss', '{:.4f}')
        self.train_logger.column('Validation Loss (combined)', '{:.4f}')
        self.train_logger.column('Validation Loss (rotation)', '{:.4f}')
        self.train_logger.column('Validation Loss (translation)', '{:.4f}')

        self.print_info(self.model)
        self.gradient_logger = Logger(self.make_output_filename('gradient.log'))
        self.gradient_logger.column('Epoch', '{:d}')
        self.gradient_logger.column('Gradient Norm', '{:.4f}')

    def load_dataset(self, args):
        # Data is loaded into RAM
        print('Generating dataset. Loading to RAM...')

        train_size = args.max_size[0]
        val_size = args.max_size[1]
        test_size = args.max_size[2]

        dataloader_train = []
        dataloader_val = []
        dataloader_test = []

        max_step = 0.1
        bounds = (0, 1)
        turn_probability = 0

        points = None#cloud.distribute_points_on_sphere(args.keypoints)
        for i in range(train_size):
            c = cloud.camera_matrix(position=(0, 0, 5), look_at=(0, 0, -10))
            p = cloud.projection_matrix(60, 1)

            feature_tracks, classes = cloud.translation_for_classification(
                c, p,
                num_classes=args.classes,
                frames=args.sequence,
                num_points=args.keypoints,
                bounds=bounds,
            )
            dataloader_train.append((feature_tracks, classes))

        points = None#cloud.distribute_points_on_sphere(args.keypoints)
        for i in range(val_size):
            c = cloud.camera_matrix(position=(0, 0, 5), look_at=(0, 0, -10))
            p = cloud.projection_matrix(60, 1)

            feature_tracks, classes = cloud.translation_for_classification(
                c, p,
                num_classes=args.classes,
                frames=args.sequence,
                num_points=args.keypoints,
                bounds=bounds,
            )
            dataloader_val.append((feature_tracks, classes))

        dataloader_test = dataloader_val
        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        gradient_norm = AverageMeter()
        accuracy = AverageMeter()

        num_batches = len(self.trainingset)
        epoch = len(self.training_loss) + 1

        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        self.model.train()
        random.shuffle(self.trainingset)
        for i, (keypoints, poses) in enumerate(self.trainingset):

            input = self.to_variable(keypoints)
            target = self.to_variable(poses[1:])

            # Forward
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss_function(output, target)

            # Backward
            loss.backward()

            grad_norm = self.gradient_norm()
            self.optimizer.step()

            training_loss.update(loss.data[0])
            gradient_norm.update(grad_norm)
            accuracy.update(self.num_correct_predictions(output, target) / output.size(0))

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Sequence [{:d}/{:d}], '
                      'T. Loss: {: .4f} ({: .4f}), '
                      'Accuracy: {: .4f}, '
                      'Grad Norm: {: .4f}'
                      .format(i + 1, num_batches,
                              loss.data[0], training_loss.average,
                              accuracy.average,
                              grad_norm
                              )
                      )

        training_loss = training_loss.average
        self.training_loss.append(training_loss)

        # Validate after each epoch
        validation_loss, accuracy = self.validate()
        self.validation_loss.append(validation_loss)

        print('Accuracy on validation set: {: .4f}'.format(accuracy))

        self.gradient_logger.log(epoch, gradient_norm.average)

        # Save extra checkpoint for best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(self.make_checkpoint(), self.make_output_filename(CHECKPOINT_BEST_FILENAME))

    def validate(self):
        return self.test(dataloader=self.validationset)

    def test(self, dataloader=None):
        if not dataloader:
            dataloader = self.testset

        avg_loss = AverageMeter()
        accuracy = 0

        self.model.eval()
        for i, (keypoints, poses) in enumerate(dataloader):

            input = self.to_variable(keypoints, volatile=True)
            target = self.to_variable(poses[1:])

            output = self.model(input)
            loss = self.loss_function(output, target)
            avg_loss.update(loss.data[0])

            # Correct predictions in the batch
            accuracy += self.num_correct_predictions(output, target)

        accuracy /= len(dataloader) * (self.sequence_length - 1)
        avg_loss = avg_loss.average

        self.test_logger.print('Average loss on testset: {:.4f}'.format(avg_loss))
        self.test_logger.clear()
        self.test_logger.print('Accuracy on testset: {:.4f}'.format(accuracy))
        self.test_logger.print()

        return avg_loss, accuracy

    def num_correct_predictions(self, output, target):
        # argmax = predicted class
        _, ind = torch.max(output.data, 1)

        # Correct predictions in the batch
        return torch.sum(torch.eq(ind, target.data))

    def make_checkpoint(self):
        checkpoint = {
            'epoch': len(self.training_loss),
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return checkpoint

    def restore_from_checkpoint(self, checkpoint):
        self.training_loss = checkpoint['training_loss']
        self.validation_loss = checkpoint['validation_loss']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def num_parameters(self):
        return sum([p.numel() for p in self.model.get_parameters()])

    def plot_performance(self):
        checkpoint = self.load_checkpoint()
        plots.plot_epoch_loss(checkpoint['training_loss'], checkpoint['validation_loss'], save=self.save_loss_plot)

    def gradient_norm(self, params=None):
        if not params:
            params = self.model.get_parameters()
        parameters = list(filter(lambda p: p.grad is not None, params))
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2

        total_norm = total_norm ** (1. / 2)
        return total_norm