import time
from math import degrees
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from transforms3d.quaternions import qinverse, qmult, quat2axangle
import cloud

import plots
from GTAV import Subsequence, visualize_predicted_path, concat_zip_dataset, Loop, FOLDERS
from base import BaseExperiment, AverageMeter, Logger, CHECKPOINT_BEST_FILENAME
import loss_functions as lsf
from models import BinaryTranslationModel
import torch.nn.parallel

class FullPose7D(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--max_size', type=int, nargs=3, default=[10000, 100, 100],
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
        parser.add_argument('--keypoints', type=int, default=50,
                            help='Number of keypoints per frame')

    def __init__(self, in_folder, out_folder, args):
        super(FullPose7D, self).__init__(in_folder, out_folder, args)

        # Model

        self.model = BinaryTranslationModel(
            hidden=args.hidden,
            nlayers=args.layers,
        )
        params = self.model.get_parameters()

        #self.model = torch.nn.DataParallel(
        #    self.model,
        #    device_ids=[0, 1]
        #)


        if self.use_cuda:
            print('Moving model to GPU ...')
            self.model.cuda()


        #self.optimizer = torch.optim.Adagrad(params, self.lr)
        self.optimizer = torch.optim.Adam(params, self.lr)

        #print('Calculating translation scale...')
        #l1_scale, l2_scale = self.determine_translation_scale()
        #l1_scale, l2_scale = 15.698, 11.605
        #self.scale = 1.0 #l1_scale
        #print('Scale: ', self.scale)

        self.loss_function = nn.CrossEntropyLoss()

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
        #self.print_info('Input size: {} x {}'.format(self.input_size[0], self.input_size[1]))
        #_, c, h, w = self.model.flownet_output_size(self.input_size)
        #self.print_info('FlowNet output shape: {} x {} x {}'.format(c, h, w))
        #self.print_info('Number of trainable parameters: {}'.format(self.num_parameters()))
        #self.print_info('Average time to load sample sequence: {:.4f} seconds'.format(self.load_benchmark()))

        self.gradient_logger = Logger(self.make_output_filename('gradient.log'))
        self.gradient_logger.column('Epoch', '{:d}')
        self.gradient_logger.column('Gradient Norm', '{:.4f}')

    def load_dataset(self, args):
        # Data is loaded into RAM
        print('Generating dataset. Loading to RAM...')

        train_size = args.max_size[0]
        val_size = args.max_size[1]
        test_size = args.max_size[2]

        dataloader_train = []#torch.zeros(train_size, args.sequence, args.keypoints, 2)
        dataloader_val = []#torch.zeros(val_size, args.sequence, args.keypoints, 2)
        dataloader_test = []#torch.zeros(test_size, args.sequence, args.keypoints, 2)

        max_step = 0.2
        turn_probability = 0.5

        points = None#cloud.distribute_points_on_sphere(args.keypoints)
        for i in range(train_size):
            c = cloud.camera_matrix(position=(0, 0, 5), look_at=(0, 0, -10))
            p = cloud.projection_matrix(60, 1)

            feature_tracks, poses, bin = cloud.animate_translation(c, p, points=points, frames=args.sequence, num_points=args.keypoints, max_step=max_step, p_turn=turn_probability)
            dataloader_train.append((feature_tracks, poses, bin))

        points = None#cloud.distribute_points_on_sphere(args.keypoints)

        for i in range(val_size):
            c = cloud.camera_matrix(position=(0, 0, 5), look_at=(0, 0, -10))
            p = cloud.projection_matrix(60, 1)

            feature_tracks, poses, bin = cloud.animate_translation(c, p, points=points, frames=args.sequence, num_points=args.keypoints, max_step=max_step, p_turn=turn_probability)


            dataloader_val.append((feature_tracks, poses, bin))

        dataloader_test = dataloader_val
        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        rotation_loss = AverageMeter()
        translation_loss = AverageMeter()
        gradient_norm = AverageMeter()
        accuracy = AverageMeter()

        num_batches = len(self.trainingset)
        first_epoch_loss = []

        epoch = len(self.training_loss) + 1
        #self.adjust_learning_rate(epoch)

        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        self.model.train()
        for i, (keypoints, poses, binary_poses) in enumerate(self.trainingset):

            #keypoints.squeeze_(0)
            #poses.squeeze_(0)

            # Normalize scale of translation
            #poses[:, :3] /= self.scale

            input = self.to_variable(keypoints)
            #target = self.to_variable(poses[1:, 0].contiguous())
            target = self.to_variable(binary_poses[1:].contiguous())

            #print(target)

            self.optimizer.zero_grad()

            # Forward
            output = self.model(input)

            #print('Output', output)

            # Loss function
            #output = self.normalize_output(output)

            #loss = torch.abs(output - target)
            #loss = loss.sum() #/ self.sequence_length
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

        all_predictions = []
        all_targets = []

        accuracy = 0

        self.model.eval()
        for i, (keypoints, poses, binary_poses) in enumerate(dataloader):

            #keypoints.squeeze_(0)
            #poses.squeeze_(0)

            # Normalize scale of translation
            #poses[:, :3] /= self.scale

            input = self.to_variable(keypoints, volatile=True)
            target = self.to_variable(poses[1:], volatile=True)
            target = self.to_variable(binary_poses[1:].contiguous())

            output = self.model(input)

            # Correct predictions in the batch
            accuracy += self.num_correct_predictions(output, target)


            all_predictions.append(output.data)
            all_targets.append(target.data)



            loss = self.loss_function(output, target)
            avg_loss.update(loss.data[0])



        accuracy /= len(dataloader) * (self.sequence_length - 1)
        avg_loss = avg_loss.average

        self.test_logger.print('Average combined loss on testset: {:.4f}'.format(avg_loss))
        self.test_logger.clear()
        self.test_logger.print('Accuracy on testset: {:.4f}'.format(accuracy))
        self.test_logger.print()

        return avg_loss, accuracy

    def num_correct_predictions(self, output, target):
        # argmax = predicted class
        _, ind = torch.max(output.data, 1)

        # Correct predictions in the batch
        return torch.sum(torch.eq(ind, target.data))


    def normalize_output(self, output):
        # Normalize quaternion in output to unit quaternion
        t = output[:, :3]
        q = output[:, 3:]

        q_norm = torch.norm(q, 2, dim=1).view(-1, 1)
        q_norm = q_norm.expand_as(q)

        # Only divide by norm if non-zero
        valid = (q_norm > 0.00001).clone()

        q2 = q.clone()
        q2[valid] = q[valid] / q_norm[valid]

        # Check if quaternion on positive hemisphere
        cos_negative = (q2[:, 0] < 0).view(-1, 1).expand_as(q)
        q3 = q2
        q3[cos_negative] *= -1

        return torch.cat((t, q3), 1)

    # def relative_rotation_angles(self, predictions, targets):
    #     # Dimensions: [N, 7]
    #     q1 = predictions[:, 3:]
    #
    #     # Normalize output quaternion
    #     #q1_norm = torch.norm(q1, 2, dim=1, keepdim=True)
    #     #q1 = q1 / q1_norm.expand_as(q1)
    #
    #     # Convert to numpy
    #     q1 = q1.cpu().numpy()
    #     q2 = targets[:, 3:].cpu().numpy()
    #
    #     # Compute the relative rotation
    #     rel_q = [qmult(qinverse(r1), r2) for r1, r2 in zip(q1, q2)]
    #     rel_angles = [quat2axangle(q)[1] for q in rel_q]
    #     rel_angles = [degrees(a) for a in rel_angles]
    #     return rel_angles

    def relative_rotation_angles2(self, predictions, targets):
        # Dimensions: [N, 7]
        q1 = predictions[:, 3:]
        q2 = targets[:, 3:]

        # Normalize output quaternion
        #q1_norm = torch.norm(q1, 2, dim=1, keepdim=True)
        #q1 = q1 / q1_norm.expand_as(q1)

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

    def determine_translation_scale(self):
        largest_l2_norm = 0
        largest_l1_norm = 0
        for i, (_, poses, _) in enumerate(self.trainingset):
            translation = poses[0, :, :3]

            l1 = max(torch.norm(translation, p=1, dim=1))
            l2 = max(torch.norm(translation, p=2, dim=1))

            largest_l1_norm = l1 if l1 > largest_l1_norm else largest_l1_norm
            largest_l2_norm = l2 if l2 > largest_l2_norm else largest_l2_norm

            if i % 100 == 0:
                print('{:d} / {:d}'.format(i, len(self.trainingset)))
                print('Largest scale (L1): {:.3f}'.format(largest_l1_norm))
                print('Largest scale (L2): {:.3f}'.format(largest_l2_norm))

        return largest_l1_norm, largest_l2_norm
