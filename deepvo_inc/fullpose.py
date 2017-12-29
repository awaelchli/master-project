import sys

from model import FullPose7DModel

sys.path.insert(0, 'utils/')
import time
import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pose_evaluation import relative_euler_rotation_error, error_distribution, translation_error_per_meters, measure_distance_along_path, relative_rotation_error_per_meters_from_euler_pose
from torch.nn.parallel import data_parallel
import pose_transforms

import plots
from base import BaseExperiment, AverageMeter, Logger, CHECKPOINT_BEST_FILENAME
import KITTI, VIPER, GTAV2

# To re-produce results
torch.manual_seed(0)
random.seed(0)


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
        parser.add_argument('--dataset', type=str, default='KITTI', choices=['KITTI', 'VIPER', 'GTA'])
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--random_truncate', type=int, default=0)

    def __init__(self, in_folder, out_folder, args):
        super(FullPose7D, self).__init__(in_folder, out_folder, args)

        # Determine size of input images
        _, (tmp, _, _) = next(enumerate(self.trainingset))
        self.input_size = (tmp.size(3), tmp.size(4))
        self.num_gpus = args.gpus
        self.random_truncate = args.random_truncate
        self.args = args

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

        self.optimizer = torch.optim.Adagrad(
            [
                {'params': self.model.layers.parameters(), 'lr': 0.0001},
                {'params': self.model.lstm.parameters(), 'lr': self.lr},
                {'params': self.model.fc.parameters(), 'lr': self.lr}
            ],
            self.lr
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
                sequence_numbers=KITTI.SEQUENCES['training'],
                relative_pose=True,
            )

            val_set = KITTI.Subsequence(
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                sequence_numbers=KITTI.SEQUENCES['validation'],
                relative_pose=True,
            )

            test_set = KITTI.Subsequence(
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                sequence_numbers=KITTI.SEQUENCES['test'],
                relative_pose=True,
            )

        elif args.dataset == 'VIPER':
            self.dataset = VIPER
            train_set = VIPER.Subsequence(
                folder=VIPER.FOLDERS['train'],
                sequence_length=args.sequence,
                overlap=args.overlap,
                transform=transform,
                max_size=args.max_size[0],
                relative_pose=True
            )

            val_set = VIPER.Subsequence(
                folder=VIPER.FOLDERS['val'],
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                max_size=args.max_size[1],
                relative_pose=True
            )

            # Ground truth not available for test folder
            test_set = val_set

        elif args.dataset == 'GTA':
            self.dataset = GTAV2
            train_set = GTAV2.Subsequence(
                folder=GTAV2.FOLDERS['train'],
                sequence_length=args.sequence,
                overlap=args.overlap,
                transform=transform,
                max_size=args.max_size[0],
                relative_pose=True
            )

            val_set = GTAV2.Subsequence(
                folder=GTAV2.FOLDERS['val'],
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                max_size=args.max_size[1],
                relative_pose=True
            )

            test_set = GTAV2.Subsequence(
                folder=GTAV2.FOLDERS['test'],
                sequence_length=args.sequence,
                overlap=0,
                transform=transform,
                max_size=args.max_size[1],
                relative_pose=True
            )

        else:
            raise RuntimeError('Unkown dataset: {}'.format(args.dataset))


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
        gradient_norm = AverageMeter()
        epoch_start = time.time()
        epoch = len(self.training_loss) + 1
        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        if False:
            # Load new dataset with increased sequence length
            self.args.sequence += 1
            self.sequence_length = self.args.sequence
            self._trainingset, self._validationset, self._testset = self.load_dataset(self.args)

        num_batches = len(self.trainingset)
        self.model.lstm.flatten_parameters()
        self.model.train()
        for i, (images, poses, _) in enumerate(self.trainingset):

            if self.random_truncate:
                # Discard randomly sized tail of sequence
                clip = random.randint(self.sequence_length - self.random_truncate, self.sequence_length)
                images = images[:, :clip, :, :]
                poses = poses[:, :clip, :]

            # From relative pose to incremental pose
            # TODO: works only with one gpu
            poses.squeeze_(0)
            poses = self.convert_pose_to_incremental(poses)
            poses.unsqueeze_(0)

            input = self.to_variable(images)
            target = self.to_variable(poses)

            self.optimizer.zero_grad()

            # Forward
            output, _ = data_parallel(self.model, input, device_ids=range(self.num_gpus))
            output = torch.stack(output.chunk(self.num_gpus, 0))

            # Loss function
            loss, r_loss, t_loss = self.loss_function(output, target[:, 1:])

            # Backward
            loss.backward()
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
            s = 'Time for first epoch: {:.4f} minutes'.format((time.time() - epoch_start) / 60)
            print(s)
            self.print_info(s)

    def validate(self):
        return self.test(dataloader=self.validationset)

    def test(self, dataloader=None):
        if not dataloader:
            dataloader = self.testset

        self.test_logger.clear()
        avg_loss = AverageMeter()
        avg_rot_loss = AverageMeter()
        avg_trans_loss = AverageMeter()

        last_frame_predictions = []
        last_frame_targets = []
        all_filenames = []
        all_outputs = torch.Tensor(len(dataloader), self.sequence_length - 1, 6)
        all_targets = torch.Tensor(len(dataloader), self.sequence_length - 1, 6)
        all_original_targets = torch.Tensor(len(dataloader), self.sequence_length, 6)

        self.model.eval()
        for i, (images, poses, filenames) in enumerate(dataloader):

            orig_poses = poses.clone()

            # From relative pose to incremental pose
            poses.squeeze_(0)
            poses = self.convert_pose_to_incremental(poses)

            input = self.to_variable(images, volatile=True)
            target = self.to_variable(poses, volatile=True)

            output, _ = self.model(input)

            loss, r_loss, t_loss = self.loss_function(output.unsqueeze(0), target[1:].unsqueeze(0))
            avg_loss.update(loss.data[0])
            avg_rot_loss.update(r_loss.data[0])
            avg_trans_loss.update(t_loss.data[0])

            all_outputs[i] = output.data
            all_targets[i] = target.data[1:]
            all_original_targets[i] = orig_poses
            all_filenames.append(filenames)

        # Average losses for rotation, translation and combined
        avg_loss = avg_loss.average
        avg_rot_loss = avg_rot_loss.average
        avg_trans_loss = avg_trans_loss.average

        all_outputs_tensor = torch.stack(all_outputs)
        all_targets_tensor = torch.stack(all_targets)

        # Convert incremental pose back to relative pose
        all_converted_outputs = torch.stack([self.convert_pose_to_global(output) for output in all_outputs])

        # Evaluate rotation error on last frame
        last_frame_predictions = [outputs[-1].view(1, -1) for outputs in all_converted_outputs]
        last_frame_targets = [targets[-1].view(1, -1) for targets in all_original_targets]
        self.plot_last_frame_error(last_frame_predictions, last_frame_targets)

        # Translation- and rotation error per meter
        self.plot_translation_error_per_meter(all_converted_outputs, all_original_targets)
        self.plot_rotation_error_per_meter(all_converted_outputs, all_original_targets)

        # Translation- and rotation loss per meter (incremental)
        self.plot_translation_loss_per_meter(all_outputs_tensor, all_targets_tensor, all_targets_global=all_original_targets)
        self.plot_rotation_loss_per_meter(all_outputs_tensor, all_targets_tensor, all_targets_global=all_original_targets)

        # Losses on testset
        self.test_logger.print('Average combined loss on testset: {:.4f}'.format(avg_loss))
        self.test_logger.print('Average rotation loss on testset: {:.4f}'.format(avg_rot_loss))
        self.test_logger.print('Average translation loss on testset: {:.4f}'.format(avg_trans_loss))

        # Visualize predicted paths
        self.visualize_paths(all_converted_outputs, all_original_targets, all_filenames)

        # Loss w.r.t. global
        losses = [self.loss_function(o.unsqueeze(0), t.unsqueeze(0)) for (o, t) in zip(all_converted_outputs, all_original_targets)]
        mean_losses = torch.Tensor(losses).mean(0)
        self.test_logger.print('Average combined loss on testset (global pose): {:.4f}'.format(mean_losses[0]))
        self.test_logger.print('Average rotation loss on testset (global pose): {:.4f}'.format(mean_losses[1]))
        self.test_logger.print('Average translation loss on testset (global pose): {:.4f}'.format(mean_losses[2]))



        # avg. incremental errors
        all_targets_cat = torch.cat(tuple(all_targets), 0)
        all_outputs_cat = torch.cat(tuple(all_outputs), 0)
        avg_error, t, error_distr = self.avg_rotation_error_incremental(all_outputs_cat.cpu(), all_targets_cat)
        avg_t_error = self.avg_translation_error_incremental(all_outputs_cat.cpu(), all_targets_cat)
        #avg_rel_rot, avg_rel_transl = self.avg_incremental_rotation_translation_in_sequence(all_targets)
        #rmse_t = self.rmse_translation_error_incremental(all_outputs.cpu(), all_targets_inc)
        #rmse_r = self.rmse_euler_error_incremental(all_outputs.cpu(), all_targets_inc)
        self.test_logger.print('Avg. incremental error (m): {}'.format(avg_t_error))
        self.test_logger.print('Avg. incremental error (deg): {}'.format(avg_error))

        return avg_loss, avg_rot_loss, avg_trans_loss

    def avg_translation_error_incremental(self, inc_outputs, inc_targets):
        avg_t_error = torch.sum(torch.norm(inc_outputs[:, :3] - inc_targets[:, :3], p=2, dim=1)) / len(inc_targets)
        return avg_t_error

    def avg_rotation_error_incremental(self, inc_outputs, inc_targets):
        errors = relative_euler_rotation_error(inc_outputs[:, 3:], inc_targets[:, 3:])
        t, error_distr = error_distribution(torch.Tensor(errors), 0, 1, 0.01)
        avg_error = sum(errors) / len(errors)
        return avg_error, t, error_distr

    def loss_function(self, output, target, average=True):
        # Dimensions: [batch, sequence_length, 6]
        bs = output.size(0)
        t1 = output[:, :, :3]
        t2 = target[:, :, :3]
        e1 = output[:, :, 3:]
        e2 = target[:, :, 3:]

        assert e1.size(2) == e2.size(2) == 3

        loss1 = torch.norm(e1 - e2, 2, dim=2)
        loss2 = torch.norm(t1 - t2, 2, dim=2)

        loss = self.beta * loss1 + loss2

        if average:
            loss = loss.sum() / bs
            loss1 = loss1.sum() / bs
            loss2 = loss2.sum() / bs

        return loss, loss1, loss2

    def loss_function_squared(self, output, target, average=True):
        # Dimensions: [batch, sequence_length, 6]
        bs = output.size(0)
        t1 = output[:, :, :3]
        t2 = target[:, :, :3]
        e1 = output[:, :, 3:]
        e2 = target[:, :, 3:]

        assert e1.size(2) == e2.size(2) == 3

        loss1 = torch.norm(e1 - e2, 2, dim=2) ** 2
        loss2 = torch.norm(t1 - t2, 2, dim=2) ** 2

        loss = self.beta * loss1 + loss2

        if average:
            loss = loss.sum() / bs
            loss1 = loss1.sum() / bs
            loss2 = loss2.sum() / bs

        return loss, loss1, loss2

    def visualize_paths(self, all_outputs, all_targets, all_filenames):
        all_outputs = all_outputs.cpu()
        all_targets = all_targets.cpu()

        for i, (output, target, filenames) in enumerate(zip(all_outputs, all_targets, all_filenames)):
            fn = filenames[0][0].replace(os.path.sep, '$$').replace('..', '')
            of1 = self.make_output_filename('path/a-{}--{:05}.svg'.format(fn, i))
            of2 = self.make_output_filename('path/b-{}--{:05}.svg'.format(fn, i))
            out = output.numpy()
            tar = target.numpy()
            self.dataset.visualize_predicted_path(out, tar, of1)
            plots.plot_xyz_error(out, tar, of2)

    def plot_last_frame_error(self, last_frame_predictions, last_frame_targets):
        last_frame_predictions = torch.cat(last_frame_predictions, 0).cpu()
        last_frame_targets = torch.cat(last_frame_targets, 0).cpu()
        # Relative rotation angle between estimated and target rotation
        rot_error_logger = self.make_logger('relative_rotation_angles_last_frame.log')
        rot_error_logger.clear()
        rot_error_logger.column('Relative rotation angle between prediction and target of last frame', format='{:.4f}')
        pose_errors = relative_euler_rotation_error(last_frame_predictions[:, 3:], last_frame_targets[:, 3:])
        for err in pose_errors:
            rot_error_logger.log(err)

        # The distribution of relative rotation angle
        thresholds, cdf = error_distribution(torch.Tensor(pose_errors), start=0.0, stop=20, step=0.1)
        plots.plot_error_distribution(thresholds, cdf, self.make_output_filename('rotation_error_distribution_last_frame.svg'))
        self.test_logger.print('Cumulative distribution of rotation error (for last frame):')
        self.test_logger.print('Threshold: ' + ', '.join([str(t) for t in thresholds]))
        self.test_logger.print('Fraction:  ' + ', '.join([str(p) for p in cdf]))
        self.test_logger.print()

    def plot_translation_error_per_meter(self, all_outputs, all_targets):
        all_outputs = all_outputs.cpu()[:, :, :3]
        all_targets = all_targets.cpu()[:, :, :3]
        longest_distance = max(measure_distance_along_path(all_targets)[:, -1])
        increments, errors = translation_error_per_meters(all_outputs, all_targets, 0, longest_distance, 1)
        errors = errors.numpy()
        increments = increments.numpy()
        plots.plot_translation_error_per_meter(increments, errors, self.make_output_filename('translation_error_per_meter.svg'))

        self.test_logger.print('Average Translation error per meters')
        self.test_logger.print('Distance [m]: ' + ', '.join([str(t) for t in increments]))
        self.test_logger.print('Translation error [m]:  ' + ', '.join([str(p) for p in errors]))
        self.test_logger.print()

    def plot_rotation_error_per_meter(self, all_outputs, all_targets):
        all_outputs = all_outputs.cpu()
        all_targets = all_targets.cpu()
        longest_distance = max(measure_distance_along_path(all_targets[:, :, :3])[:, -1])
        increments, errors = relative_rotation_error_per_meters_from_euler_pose(all_outputs, all_targets, 0, longest_distance, 1)
        errors = errors.numpy()
        increments = increments.numpy()
        plots.plot_rotation_error_per_meter(increments, errors, self.make_output_filename('rotation_error_per_meter.svg'))

        self.test_logger.print('Average Relative Rotation error per meters')
        self.test_logger.print('Distance [m]: ' + ', '.join([str(t) for t in increments]))
        self.test_logger.print('Rotation error [deg]:  ' + ', '.join([str(p) for p in errors]))
        self.test_logger.print()

    def convert_pose_to_global(self, output):
        # Converts output of incremental poses to poses that are relative to the first frame
        # output: [sequence, 6]
        output = torch.cat((torch.zeros(1, 6), output), 0)
        matrices = [pose_transforms.euler_pose_vector_to_matrix(pose.view(1, -1)) for pose in output]
        matrices = torch.stack(matrices)
        matrices = pose_transforms.relative_previous_pose_to_relative_first_matrix(matrices)
        output = [pose_transforms.matrix_to_euler_pose_vector(m) for m in matrices]
        output = torch.cat(tuple(output), 0)
        return output

    def convert_pose_to_incremental(self, poses):
        # Converts poses in global coordinates to incremental poses (in coordinate frame of previous image)
        # poses: [sequence, 6]
        matrices = [pose_transforms.euler_pose_vector_to_matrix(pose.view(1, -1)) for pose in poses]
        matrices = torch.stack(matrices)
        matrices = pose_transforms.relative_previous_pose_matrix(matrices)
        poses = [pose_transforms.matrix_to_euler_pose_vector(m) for m in matrices]
        poses = torch.cat(tuple(poses), 0)
        return poses

    def plot_translation_loss_per_meter(self, all_outputs, all_targets, all_targets_global):
        all_outputs = all_outputs.cpu()[:, :, :3]
        all_targets = all_targets.cpu()[:, :, :3]

        dists = measure_distance_along_path(all_targets_global[:, :, :3])
        longest_distance = max(dists[:, -1])

        def t_err_func(o, t):
            z = torch.zeros(o.size(0), o.size(1), 3)
            _, _, err_t = self.loss_function(torch.cat((o, z), 2), torch.cat((t, z), 2), average=False)
            return err_t

        increments, errors = translation_error_per_meters(all_outputs, all_targets, 0, longest_distance, 1, err_func=t_err_func, dist=dists)
        errors = errors.numpy()
        increments = increments.numpy()
        plots.plot_translation_error_per_meter(increments, errors, self.make_output_filename('translation_loss_per_meter.svg'))

        self.test_logger.print('Average Translation loss per meters')
        self.test_logger.print('Distance [m]: ' + ', '.join([str(t) for t in increments]))
        self.test_logger.print('Translation loss:  ' + ', '.join([str(p) for p in errors]))
        self.test_logger.print()

    def plot_rotation_loss_per_meter(self, all_outputs, all_targets, all_targets_global):
        all_outputs = all_outputs.cpu()
        all_targets = all_targets.cpu()

        dists = measure_distance_along_path(all_targets_global[:, :, :3])
        longest_distance = max(dists[:, -1])

        def r_err_func(o, t):
            z = torch.zeros(o.size(0), o.size(1), 3)
            _, err_r, _ = self.loss_function(torch.cat((z, o), 2), torch.cat((z, t), 2), average=False)
            return err_r

        increments, errors = relative_rotation_error_per_meters_from_euler_pose(all_outputs, all_targets, 0, longest_distance, 1, err_func=r_err_func, dist=dists)
        errors = errors.numpy()
        increments = increments.numpy()
        plots.plot_rotation_error_per_meter(increments, errors, self.make_output_filename('rotation_loss_per_meter.svg'))

        self.test_logger.print('Average Rotation loss per meters')
        self.test_logger.print('Distance [m]: ' + ', '.join([str(t) for t in increments]))
        self.test_logger.print('Rotation loss:  ' + ', '.join([str(p) for p in errors]))
        self.test_logger.print()

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
