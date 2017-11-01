import sys
sys.path.insert(0, 'utils/')
import time
from math import degrees
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from transforms3d.quaternions import qinverse, qmult, quat2axangle
from pose_evaluation import relative_euler_rotation_error
from torch.nn.parallel import data_parallel

import plots
from base import BaseExperiment, AverageMeter, Logger, CHECKPOINT_BEST_FILENAME
from flownet.models.FlowNetS import flownets
import KITTI, VIPER


class FullPose7DModel(nn.Module):

    def __init__(self, input_size, hidden=500, nlayers=3, dropout=0, fix_flownet=True):
        super(FullPose7DModel, self).__init__()

        flownet = flownets('../data/Pretrained Models/flownets_pytorch.pth')
        flownet.train(False)

        self.layers = flownet
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

        fout = self.flownet_output_size(input_size)
        self.hidden = hidden
        self.nlayers = nlayers

        self.lstm = nn.LSTM(
            input_size=fout[1] * fout[2] * fout[3],
            hidden_size=self.hidden,
            num_layers=self.nlayers,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden, 6)

        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()

    def flownet_output_size(self, input_size):
        var = Variable(torch.zeros(1, 6, input_size[0], input_size[1]), volatile=True)
        if next(self.layers.parameters()).is_cuda:
            var = var.cuda()
        out = self.layers(var)
        return out.size(0), out.size(1), out.size(2), out.size(3)

    def forward(self, input):
        # Input shape: [1, sequence, channels, h, w]
        input.squeeze_(0)

        n = input.size(0)
        first = input[:n-1]
        second = input[1:]

        # New shape: [sequence - 1, 2 * channels, h, w]
        pairs = torch.cat((first, second), 1)

        assert pairs.size(0) == n - 1

        # Using batch mode to forward sequence
        pairs = self.layers(pairs)

        h0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        c0 = Variable(torch.zeros(self.nlayers, 1, self.hidden))
        if input.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        init = (h0, c0)
        outputs, _ = self.lstm(pairs.view(1, n - 1, -1), init)

        outputs = self.drop(outputs)
        predictions = self.fc(outputs.squeeze(0))

        return predictions

    def train(self, mode=True):
        super(FullPose7DModel, self).train(mode)
        self.lstm.train(mode)

    def eval(self):
        super(FullPose7DModel, self).eval()
        self.lstm.eval()

    def get_parameters(self):
        params = list(self.lstm.parameters()) + list(self.fc.parameters()) + list(self.layers.parameters())
        return params


class FullPose7D(BaseExperiment):

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
        parser.add_argument('--hidden', type=int, default=500,
                            help='Hidden size of the LSTM')
        parser.add_argument('--layers', type=int, default=3,
                            help='Number of layers in the LSTM')
        parser.add_argument('--gpus', type=int, default=1)
        parser.add_argument('--overlap', type=int, default=0)
        parser.add_argument('--dataset', type=str, default='KITTI', choices=['KITTI', 'VIPER'])
        parser.add_argument('--dropout', type=float, default=0.0)

    def __init__(self, in_folder, out_folder, args):
        super(FullPose7D, self).__init__(in_folder, out_folder, args)

        # Determine size of input images
        _, (tmp, _, _) = next(enumerate(self.trainingset))
        self.input_size = (tmp.size(3), tmp.size(4))
        self.num_gpus = args.gpus

        # Model
        self.model = FullPose7DModel(
            self.input_size,
            hidden=args.hidden,
            nlayers=args.layers,
            dropout=args.dropout,
        )

        if self.use_cuda:
            print('Moving model to GPU ...')
            self.model.cuda()

        #params = self.model.get_parameters()
        #self.optimizer = torch.optim.Adagrad(params, self.lr)
        self.optimizer = torch.optim.Adagrad(
            [
                {'params': self.model.layers.parameters(), 'lr': 0.0001},
                {'params': self.model.lstm.parameters(), 'lr': self.lr},
                {'params': self.model.fc.parameters(), 'lr': self.lr}
            ]
            , self.lr
        )

        self.beta = args.beta
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
        self.print_info('Input size: {} x {}'.format(self.input_size[0], self.input_size[1]))
        _, c, h, w = self.model.flownet_output_size(self.input_size)
        self.print_info('FlowNet output shape: {} x {} x {}'.format(c, h, w))
        self.print_info('Number of trainable parameters: {}'.format(self.num_parameters()))
        self.print_info('Average time to load sample sequence: {:.4f} seconds'.format(self.load_benchmark()))

        self.gradient_logger = Logger(self.make_output_filename('gradient.log'))
        self.gradient_logger.column('Epoch', '{:d}')
        self.gradient_logger.column('Gradient Norm', '{:.4f}')

    def load_dataset(self, args):

        # Image pre-processing
        transform = transforms.Compose([
            #transforms.Scale(args.image_size),
            transforms.Scale(320),
            transforms.CenterCrop((320, 448)),
            transforms.ToTensor(),
        ])

        if args.dataset == 'KITTI':
            self.dataset = KITTI
            train_set = KITTI.Subsequence(
                sequence_length=args.sequence,
                overlap=args.overlap,
                transform=transform,
                sequence_numbers=KITTI.SEQUENCES['training']
            )

            val_set = KITTI.Subsequence(
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                sequence_numbers=KITTI.SEQUENCES['validation']
            )

            test_set = KITTI.Subsequence(
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                sequence_numbers=KITTI.SEQUENCES['test']
            )

        elif args.dataset == 'VIPER':
            self.dataset = VIPER
            train_set = VIPER.Subsequence(
                folder=VIPER.FOLDERS['train'],
                sequence_length=args.sequence,
                overlap=args.overlap,
                transform=transform,
                max_size=args.max_size[0]
            )

            val_set = VIPER.Subsequence(
                folder=VIPER.FOLDERS['val'],
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                max_size=args.max_size[1]
            )

            # Ground truth not available for test folder
            test_set = val_set
        else:
            raise RuntimeError('unkown dataset: {}'.format(args.dataset))


        dataloader_train = DataLoader(
            train_set,
            batch_size=args.gpus,
            pin_memory=False,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True
        )

        dataloader_val = DataLoader(
            val_set,
            batch_size=1,
            pin_memory=False,
            shuffle=False,
            num_workers=args.workers,
            drop_last=True
        )

        dataloader_test = DataLoader(
            test_set,
            batch_size=1,
            pin_memory=False,
            shuffle=False,
            num_workers=args.workers,
            drop_last=True
        )

        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        rotation_loss = AverageMeter()
        translation_loss = AverageMeter()
        forward_time = AverageMeter()
        backward_time = AverageMeter()
        loss_time = AverageMeter()
        gradient_norm = AverageMeter()
        epoch_start = time.time()

        num_batches = len(self.trainingset)
        first_epoch_loss = []

        epoch = len(self.training_loss) + 1
        #self.adjust_learning_rate(epoch)

        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        self.model.lstm.flatten_parameters()
        self.model.train()
        for i, (images, poses, fn) in enumerate(self.trainingset):

            #images.squeeze_(0)
            #poses.squeeze_(0)

            #print(poses[0, :5, :])

            input = self.to_variable(images)
            target = self.to_variable(poses)

            self.optimizer.zero_grad()

            # Forward
            #start = time.time()
            output = data_parallel(self.model, input, device_ids=range(self.num_gpus))
            output = torch.stack(output.chunk(self.num_gpus, 0))
            #forward_time.update(time.time() - start)

            # Loss function
            #start = time.time()
            loss, r_loss, t_loss = self.loss_function(output, target[:, 1:])
            #loss_time.update(time.time() - start)

            # Backward
            #start = time.time()
            loss.backward()
            #backward_time.update(time.time() - start)

            #torch.nn.utils.clip_grad_norm(list(self.model.lstm.parameters()), 200)

            grad_norm = self.gradient_norm()
            self.optimizer.step()

            training_loss.update(loss.data[0])
            rotation_loss.update(r_loss.data[0])
            translation_loss.update(t_loss.data[0])
            gradient_norm.update(grad_norm)

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Sequence [{:d}/{:d}], '
                      'Total Loss: {: .4f} ({: .4f}), '
                      'R. Loss: {: .4f} ({: .4f}), '
                      'T. Loss: {: .4f} ({: .4f}), '
                      'Grad Norm: {: .4f}'
                      .format(i + 1, num_batches,
                              loss.data[0], training_loss.average,
                              r_loss.data[0], rotation_loss.average,
                              t_loss.data[0], translation_loss.average,
                              grad_norm
                              )
                      )

            if loss.data[0] > 500:
                self.print_info('\n\nHIGH: {}\n\n'.format(fn[0]))
            # if epoch == 1:
            #     first_epoch_loss.append(loss.data[0])
            #     plots.plot_epoch_loss(first_epoch_loss, save=self.make_output_filename('first_epoch_loss.pdf'))

        training_loss = training_loss.average
        self.training_loss.append(training_loss)

        # Validate after each epoch
        validation_loss, validation_r_loss, validation_t_loss = self.validate()
        self.validation_loss.append(validation_loss)

        self.train_logger.log(epoch, training_loss, validation_loss, validation_r_loss, validation_t_loss)
        self.gradient_logger.log(epoch, gradient_norm.average)

        # Save extra checkpoint for best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(self.make_checkpoint(), self.make_output_filename(CHECKPOINT_BEST_FILENAME))

        if epoch == 1:
            #self.print_info('Average time for forward operation: {:.4f} seconds'.format(forward_time.average))
            #self.print_info('Average time for backward operation: {:.4f} seconds'.format(backward_time.average))
            #self.print_info('Average time for loss computation: {:.4f} seconds'.format(loss_time.average))
            s = 'Time for first epoch: {:.4f} minutes'.format((time.time() - epoch_start) / 60)
            print(s)
            self.print_info(s)

    def validate(self):
        return self.test(dataloader=self.validationset)

    def test(self, dataloader=None):
        if not dataloader:
            dataloader = self.testset

        avg_loss = AverageMeter()
        avg_rot_loss = AverageMeter()
        avg_trans_loss = AverageMeter()

        last_frame_predictions = []
        last_frame_targets = []
        rel_angle_error_over_time = []

        self.model.eval()
        for i, (images, poses, filenames) in enumerate(dataloader):

            #images.squeeze_(0)
            poses.squeeze_(0)

            input = self.to_variable(images, volatile=True)
            target = self.to_variable(poses, volatile=True)

            output = self.model(input)

            last_frame_predictions.append(output.data[-1].view(1, -1))
            last_frame_targets.append(target.data[-1].view(1, -1))

            # A few sequences are shorter, don't add them for averaging
            #tmp = relative_euler_rotation_error(output.data[:, 3:], target.data[1:, 3:])
            #if len(tmp) == self.sequence_length - 1:
            #    rel_angle_error_over_time.append(tmp)

            loss, r_loss, t_loss = self.loss_function(output.unsqueeze(0), target[1:].unsqueeze(0))
            avg_loss.update(loss.data[0])
            avg_rot_loss.update(r_loss.data[0])
            avg_trans_loss.update(t_loss.data[0])

            #print(filenames[0])

            # Visualize predicted path
            fn = filenames[0][0].replace(os.path.sep, '$$').replace('..', '')
            of = self.make_output_filename('path/a-{}--{:05}.png'.format(fn, i))
            of2 = self.make_output_filename('path/b-{}--{:05}.png'.format(fn, i))
            out_cpu = output.data.cpu().numpy()
            tar_cpu = target.data[1:].cpu().numpy()
            self.dataset.visualize_predicted_path(out_cpu, tar_cpu, of)
            plots.plot_xyz_error(out_cpu, tar_cpu, of2)

            if loss.data[0] > 500:
                print('\n\nHIGH LOSS: {:.4f}, at {}\n\n'.format(loss.data[0], filenames))

        #print(last_frame_predictions[:5])
        #print(last_frame_targets[:5])

        # Average losses for rotation, translation and combined
        avg_loss = avg_loss.average
        avg_rot_loss = avg_rot_loss.average
        avg_trans_loss = avg_trans_loss.average

        last_frame_predictions = torch.cat(last_frame_predictions, 0).cpu()
        last_frame_targets = torch.cat(last_frame_targets, 0).cpu()

        #rel_angle_error_over_time = torch.Tensor(rel_angle_error_over_time).mean(0).view(-1)

        # Relative rotation angle between estimated and target rotation
        rot_error_logger = self.make_logger('relative_rotation_angles_last_frame.log')
        rot_error_logger.clear()
        rot_error_logger.column('Relative rotation angle between prediction and target of last frame', format='{:.4f}')
        pose_errors = relative_euler_rotation_error(last_frame_predictions[:, 3:], last_frame_targets[:, 3:])
        for err in pose_errors:
            rot_error_logger.log(err)

        # The distribution of relative rotation angle
        thresholds, cdf = self.error_distribution(torch.Tensor(pose_errors))
        plots.plot_error_distribution(thresholds, cdf, self.make_output_filename('rotation_error_distribution_last_frame.pdf'))
        # self.test_logger.clear()
        # self.test_logger.print('Cumulative distribution of rotation error:')
        # self.test_logger.print('Threshold: ' + ', '.join([str(t) for t in thresholds]))
        # self.test_logger.print('Fraction:  ' + ', '.join([str(p) for p in cdf]))
        # self.test_logger.print()
        self.test_logger.print('Average combined loss on testset: {:.4f}'.format(avg_loss))
        self.test_logger.print('Average rotation loss on testset: {:.4f}'.format(avg_rot_loss))
        self.test_logger.print('Average translation loss on testset: {:.4f}'.format(avg_trans_loss))
        #
        # plots.plot_sequence_error(list(rel_angle_error_over_time), self.make_output_filename('average_rotation_error_over_time.pdf'))
        # self.test_logger.clear()
        # self.test_logger.print('Average relative rotation error over time:')
        # self.test_logger.print(', '.join([str(i) for i in rel_angle_error_over_time]))
        # self.test_logger.print()

        return avg_loss, avg_rot_loss, avg_trans_loss

    def loss_function(self, output, target):
        # Dimensions: [batch, sequence_length, 6]
        sequence_length = output.size(1)
        bs = output.size(0)

        #print(output)
        #print(target)

        t1 = output[:, :, :3]
        t2 = target[:, :, :3]
        e1 = output[:, :, 3:]
        e2 = target[:, :, 3:]

        assert e1.size(2) == e2.size(2) == 3


        c = torch.nn.MSELoss()

        # Loss for rotation: dot product between quaternions
        #loss1 = torch.norm(q1 - q2, 1, dim=1)
        #loss1 = 1 - (q1 * q2).sum(1) ** 2
        #loss1 = loss1.sum() / sequence_length
        loss1 = torch.norm(e1 - e2, 2, dim=2)


        # Loss for translation
        #t_diff = torch.norm(t1 - t2, 1, dim=1)
        #loss2 = t_diff
        #loss2 = loss2.sum() / sequence_length
        loss2 = torch.norm(t1 - t2, 2, dim=2)

        loss = self.beta * loss1 + loss2

        return loss.sum() / bs, loss1.sum() / bs, loss2.sum() / bs

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
        lr = self.lr * (0.5 ** (epoch // 5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def plot_performance(self):
        checkpoint = self.load_checkpoint()
        plots.plot_epoch_loss(checkpoint['training_loss'], checkpoint['validation_loss'], save=self.save_loss_plot)

    def error_distribution(self, errors, start=0.0, stop=180.0, step=1.0):
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
