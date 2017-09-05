import time
from math import degrees
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from transforms3d.quaternions import qinverse, qmult, quat2axangle

import plots
from GTAV import Subsequence, visualize_predicted_path, FOLDERS
from base import BaseExperiment, AverageMeter, CHECKPOINT_BEST_FILENAME
from flownet.models.FlowNetS import flownets


class RotationModel(nn.Module):

    def __init__(self, input_size, fix_flownet=True):
        super(RotationModel, self).__init__()

        flownet = flownets('../data/Pretrained Models/flownets_pytorch.pth')
        self.flownet = flownet
        self.flownet.train(False)

        self.layers = torch.nn.Sequential(
            flownet.conv1,
            flownet.conv2,
            flownet.conv3,
            flownet.conv3_1,
            flownet.conv4,
            flownet.conv4_1,
            flownet.conv5,
            flownet.conv5_1,
            flownet.conv6,
            flownet.conv6_1,
        )
        self.fix_flownet = fix_flownet
        if fix_flownet:
            for param in self.layers.parameters():
                param.requires_grad = False

            for param in self.flownet.parameters():
                param.requires_grad = False

        self.hidden = 100
        self.nlayers = 1
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden,
            num_layers=self.nlayers,
            batch_first=True
        )

        # Output is a quaternion
        self.fc = nn.Linear(self.hidden, 4)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()

    def flownet_output_size(self, input_size):
        var = Variable(torch.zeros(1, 6, input_size[0], input_size[1]), volatile=True)
        if next(self.layers.parameters()).is_cuda:
            var = var.cuda()
        #out = self.layers(var)
        out = self.flownet(var)
        return out.size(0), out.size(1), out.size(2), out.size(3)

    def forward(self, input):
        # Input shape: [sequence, channels, h, w]
        n = input.size(0)
        first = input[:n-1]
        second = input[1:]

        # New shape: [sequence - 1, 2 * channels, h, w]
        pairs = torch.cat((first, second), 1)

        assert pairs.size(0) == n - 1

        # Using batch mode to forward sequence
        #out_pairs = self.layers(pairs)

        out_pairs = self.flownet(pairs)

        # Shape of out_pairs is: [n - 1, 2, h, w]
        out_means = out_pairs.mean(2).mean(3).view(-1, 2)
        out_means = out_means.unsqueeze(0)
        # Shape: [1, n - 1, 2]

        flows = []
        #for pair in pairs:
            #flows.append(self.flownet(pair.unsqueeze(0)))

        h0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        c0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        if input.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        outputs, _ = self.lstm(out_means, (h0, c0))
        predictions = self.fc(outputs.squeeze(0))

        return predictions, flows

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters())
        if not self.fix_flownet:
            params = list(self.layers.parameters()) + params
        return params


class RotationOnly(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--max_size', type=int, nargs=3, default=[0, 0, 0],
                            help="""Clips training-, validation-, and testset at the given size. 
                            A zero signalizes that the whole dataset should be used.""")
        parser.add_argument('--sequence', type=int, default=10,
                            help='Length of sequence fed to the LSTM')
        parser.add_argument('--image_size', type=int, nargs=1, default=256,
                            help='The shorter side of the images will be scaled to the given size.')
        parser.add_argument('--beta', type=float, default=1,
                            help='Balance weight for the translation loss')

    def __init__(self, in_folder, out_folder, args):
        super(RotationOnly, self).__init__(in_folder, out_folder, args)

        # Determine size of input images
        _, (tmp, _) = next(enumerate(self.trainingset))
        self.input_size = (tmp.size(3), tmp.size(4))

        # Model
        self.model = RotationModel(self.input_size, fix_flownet=True)

        if self.use_cuda:
            print('Moving model to GPU ...')
            self.model.cuda()

        params = self.model.get_parameters()
        self.optimizer = torch.optim.SGD(params, self.lr)

        self.beta = args.beta
        self.print_freq = args.print_freq
        self.sequence_length = args.sequence
        self.training_loss = []
        self.validation_loss = []

        self.train_logger.column('Epoch', '{:d}')
        self.train_logger.column('Training Loss', '{:.4f}')
        self.train_logger.column('Validation Loss', '{:.4f}')

        self.print_info(self.model)
        self.print_info('Input size: {} x {}'.format(self.input_size[0], self.input_size[1]))
        _, c, h, w = self.model.flownet_output_size(self.input_size)
        self.print_info('FlowNet output shape: {} x {} x {}'.format(c, h, w))
        self.print_info('Number of trainable parameters: {}'.format(self.num_parameters()))
        self.print_info('Average time to load sample sequence: {:.4f} seconds'.format(self.load_benchmark()))

    def load_dataset(self, args):
        traindir = FOLDERS['standing']['training']
        valdir = FOLDERS['standing']['validation']
        testdir = FOLDERS['standing']['test']



        # Image pre-processing
        transform = transforms.Compose([
            transforms.Scale(320),
            transforms.CenterCrop((320, 448)),
            transforms.ToTensor(),
        ])

        train_set = Subsequence(
            data_folder=traindir['data'],
            pose_folder=traindir['pose'],
            sequence_length=args.sequence,
            transform=transform,
            max_size=args.max_size[0],
        )

        val_set = Subsequence(
            data_folder=valdir['data'],
            pose_folder=valdir['pose'],
            sequence_length=args.sequence,
            transform=transform,
            max_size=args.max_size[1],
        )

        test_set = Subsequence(
            data_folder=testdir['data'],
            pose_folder=testdir['pose'],
            sequence_length=args.sequence,
            transform=transform,
            max_size=args.max_size[2],
        )

        dataloader_train = DataLoader(
            train_set,
            batch_size=1,
            pin_memory=self.use_cuda,
            shuffle=True,
            num_workers=args.workers)

        dataloader_val = DataLoader(
            val_set,
            batch_size=1,
            pin_memory=self.use_cuda,
            shuffle=False,
            num_workers=args.workers)

        dataloader_test = DataLoader(
            test_set,
            batch_size=1,
            pin_memory=self.use_cuda,
            shuffle=False,
            num_workers=args.workers)

        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        forward_backward_time = AverageMeter()
        num_batches = len(self.trainingset)

        epoch = len(self.training_loss) + 1
        #self.adjust_learning_rate(epoch)

        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        for i, (images, poses) in enumerate(self.trainingset):

            images.squeeze_(0)
            poses.squeeze_(0)

            # Only use rotational part of pose
            poses = poses[:, 3:]

            input = self.to_variable(images)
            target = self.to_variable(poses)

            self.optimizer.zero_grad()

            start = time.time()
            output, _ = self.model(input)
            output = self.normalize_output(output)

            #print('Prediction: ', output)
            #print('Target:     ', target)

            loss = self.loss_function(output, target[1:])
            loss.backward()
            self.optimizer.step()

            forward_backward_time.update(time.time() - start)

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Sample [{:d}/{:d}], Rotation Loss: {:.4f}'
                      .format(i + 1, num_batches, loss.data[0]))

            training_loss.update(loss.data[0])

        training_loss = training_loss.average
        self.training_loss.append(training_loss)

        # Validate after each epoch
        validation_loss = self.validate()
        self.validation_loss.append(validation_loss)

        self.train_logger.log(epoch, training_loss, validation_loss)

        # Save extra checkpoint for best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(self.make_checkpoint(), self.make_output_filename(CHECKPOINT_BEST_FILENAME))

        if epoch == 1:
            self.print_info('Average time for forward and backward operation: {:.4f} seconds'.format(
                forward_backward_time.average))

    def validate(self):
        avg_loss = AverageMeter()

        for i, (images, poses) in enumerate(self.validationset):

            images.squeeze_(0)
            poses.squeeze_(0)

            # Only use rotational part of pose
            poses = poses[:, 3:]

            input = self.to_variable(images, volatile=True)
            target = self.to_variable(poses, volatile=True)

            output, _ = self.model(input)
            output = self.normalize_output(output)

            loss = self.loss_function(output, target[1:])
            avg_loss.update(loss.data[0])

        avg_loss = avg_loss.average

        return avg_loss

    def test(self):
        avg_loss = AverageMeter()

        all_predictions = []
        all_targets = []
        rel_angle_error_over_time = []

        for i, (images, poses) in enumerate(self.testset):

            images.squeeze_(0)
            poses.squeeze_(0)

            # Only use rotational part of pose
            poses = poses[:, 3:]

            input = self.to_variable(images, volatile=True)
            target = self.to_variable(poses, volatile=True)

            output, _ = self.model(input)
            output = self.normalize_output(output)

            all_predictions.append(output.data)
            all_targets.append(target.data[1:])

            # A few sequences are shorter, don't add them for averaging
            tmp = self.relative_rotation_angles(output.data, target.data[1:])
            if len(tmp) == self.sequence_length - 1:
                rel_angle_error_over_time.append(tmp)

            loss = self.loss_function(output, target[1:])
            avg_loss.update(loss.data[0])


        # Average losses for rotation, translation and combined
        avg_loss = avg_loss.average

        all_predictions = torch.cat(all_predictions, 0)
        all_targets = torch.cat(all_targets, 0)
        rel_angle_error_over_time = torch.Tensor(rel_angle_error_over_time).mean(0).view(-1)

        # Relative rotation angle between estimated and target rotation
        rot_error_logger = self.make_logger('relative_rotation_angles.log')
        rot_error_logger.clear()
        rot_error_logger.column('Relative rotation angle between prediction and target', format='{:.4f}')
        pose_errors = self.relative_rotation_angles(all_predictions, all_targets)
        for err in pose_errors:
            rot_error_logger.log(err)

        # The distribution of relative rotation angle
        thresholds, cdf = self.error_distribution(torch.Tensor(pose_errors))
        plots.plot_error_distribution(thresholds, cdf, self.make_output_filename('rotation_error_distribution.pdf'))
        self.test_logger.clear()
        self.test_logger.print('Cumulative distribution of rotation error:')
        self.test_logger.print('Threshold: ' + ', '.join([str(t) for t in thresholds]))
        self.test_logger.print('Fraction:  ' + ', '.join([str(p) for p in cdf]))
        self.test_logger.print()
        self.test_logger.print('Average loss on testset: {:.4f}'.format(avg_loss))

        plots.plot_sequence_error(list(rel_angle_error_over_time), self.make_output_filename('average_rotation_error_over_time.pdf'))
        self.test_logger.clear()
        self.test_logger.print('Average relative rotation error over time:')
        self.test_logger.print(', '.join([str(i) for i in rel_angle_error_over_time]))
        self.test_logger.print()

        return avg_loss

    def loss_function(self, output, target):
        # Dimensions: [sequence_length, 7]
        sequence_length = output.size(0)

        #print(output)
        #print(target)
        q1 = output
        q2 = target

        assert q1.size(1) == q2.size(1) == 4

        #print('Q1, Q2')
        #print(q1)
        #print(q2)

        # Loss for rotation: dot product between quaternions
        loss1 = 1 - (q1 * q2).sum(1) ** 2
        loss1 = loss1.sum() / sequence_length

        return loss1

    def normalize_output(self, output):
        # Normalize quaternion in output to unit quaternion
        q = output
        q_norm = torch.norm(q, 2, dim=1).view(-1, 1)
        q = q / q_norm.expand_as(q)
        return q

    def relative_rotation_angles(self, predictions, targets):
        # Dimensions: [N, 7]
        q1 = predictions
        q2 = targets

        rel_angles = torch.acos(2 * (q1 * q2).sum(1) ** 2 - 1)

        return [degrees(a) for a in rel_angles.view(-1)]

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

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.5 ** (epoch // 5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def plot_performance(self):
        checkpoint = self.load_checkpoint()
        plots.plot_epoch_loss(checkpoint['training_loss'], checkpoint['validation_loss'], save=self.save_loss_plot)

    def error_distribution(self, errors, start=0.0, stop=10.0, step=1.0):
        thresholds = list(torch.arange(start, stop, step))
        n = torch.numel(errors)
        distribution = [torch.sum(errors <= t) / n for t in thresholds]
        return thresholds, distribution

    def load_benchmark(self):
        start = time.time()
        g = enumerate(self.trainingset)
        n = 10
        for i in range(n):
            next(g)

        elapsed = time.time() - start
        return elapsed / n
