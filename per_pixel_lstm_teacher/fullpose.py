import sys
sys.path.insert(0, 'utils/')
import time
from math import degrees, ceil

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms

import plots
#from GTAV import Subsequence, visualize_predicted_path, concat_zip_dataset, Loop, FOLDERS
from GTAV import SequenceCollection, FOLDERS
from sampler import StridedSampler
from base import BaseExperiment, AverageMeter, Logger, CHECKPOINT_BEST_FILENAME
import loss_functions as lsf
from model import FullPose7DModel


class FullPose7D(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--max_size', type=int, nargs=3, default=[0, 0, 0],
                            help="""Clips training-, validation-, and testset at the given size. 
                            A zero signalizes that the whole dataset should be used.""")
        parser.add_argument('--stride', type=int, default=1)
        parser.add_argument('--pool', type=int, default=50)
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

    def __init__(self, in_folder, out_folder, args):
        super(FullPose7D, self).__init__(in_folder, out_folder, args)

        # Model
        self.model = FullPose7DModel(
            hidden=args.hidden,
            nlayers=args.layers,
            pool_size=args.pool
        )

        if self.use_cuda:
            print('Moving model to GPU ...')
            self.model.cuda()

        params = self.model.get_parameters()
        #self.optimizer = torch.optim.Adagrad(params, self.lr)
        self.optimizer = torch.optim.Adam(params, self.lr)

        print('Calculating translation scale...')
        #l1_scale, l2_scale = self.determine_translation_scale()
        l1_scale, l2_scale = 15.698, 11.605
        self.scale = 1.0 #l1_scale
        print('Scale: ', self.scale)

        self.loss_function = lsf.InnerL1(args.beta)

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
        self.print_info('Number of trainable parameters: {}'.format(self.num_parameters()))
        #self.print_info('Average time to load sample sequence: {:.4f} seconds'.format(self.load_benchmark()))

        self.gradient_logger = Logger(self.make_output_filename('gradient.log'))
        self.gradient_logger.column('Epoch', '{:d}')
        self.gradient_logger.column('Gradient Norm', '{:.4f}')

    def load_dataset(self, args):
        print('Preparing data...')

        train_dir = '../data/GTA V/walking/train/data'
        test_dir = '../data/GTA V/walking/test/data'
        val_dir = test_dir

        h, w = 320, 448
        # Transforms for image pre-processing
        transform = transforms.Compose([
            transforms.Scale(h),
            transforms.CenterCrop((h, w)),
            transforms.ToTensor(),
        ])

        print('Features per frame: {:d}'.format(int(ceil(h / args.pool) * ceil(w / args.pool))))

        # Training data
        train_collection = SequenceCollection(train_dir, transform)
        train_dataloaders = []
        for sequence in train_collection:
            sampler = StridedSampler(sequence, stride=args.stride, max_size=args.max_size[0])
            batch_sampler = BatchSampler(sampler, batch_size=args.sequence, drop_last=True)
            loader = DataLoader(
                dataset=sequence,
                batch_sampler=batch_sampler,
                pin_memory=False,
                num_workers=args.workers
            )
            train_dataloaders.append(loader)

        # Test data
        test_collection = SequenceCollection(test_dir, transform)
        test_dataloaders = []
        for sequence in test_collection:
            sampler = StridedSampler(sequence, stride=args.stride, max_size=args.max_size[1])
            batch_sampler = BatchSampler(sampler, batch_size=args.sequence, drop_last=True)
            loader = DataLoader(
                dataset=sequence,
                batch_sampler=batch_sampler,
                pin_memory=False,
                num_workers=args.workers
            )
            test_dataloaders.append(loader)

        # TODO: change in final version
        val_dataloaders = test_dataloaders

        return train_dataloaders, val_dataloaders, test_dataloaders

    def train(self):
        training_loss = AverageMeter()
        rotation_loss = AverageMeter()
        translation_loss = AverageMeter()
        # forward_time = AverageMeter()
        # backward_time = AverageMeter()
        # loss_time = AverageMeter()
        gradient_norm = AverageMeter()

        epoch = len(self.training_loss) + 1
        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)
        num_sequences = len(self.trainingset)

        self.model.train()
        for i, dataloader in enumerate(self.trainingset):

            num_batches = len(dataloader)

            # Prepare LSTM state
            state = None

            for j, (images, poses, filenames) in enumerate(dataloader):

                input = self.to_variable(images)
                target = self.to_variable(poses)

                self.optimizer.zero_grad()

                prev_pose = Variable(torch.zeros(1, 7)).cuda()
                outputs = Variable(torch.zeros(target.size(0), 7)).cuda()
                for k in range(input.size(0)):
                    frame = input[k]
                    prediction, state = self.model(frame, prev_pose, state)
                    outputs[k] = prediction
                    # In training, feed the ground truth pose from the previous frame
                    prev_pose = target[k].view(1, -1)

                outputs = self.normalize_output(outputs)

                # Loss function
                # Ignore first output
                loss, r_loss, t_loss = self.loss_function(outputs[1:], target[1:])

                # Backward
                loss.backward()

                grad_norm = self.gradient_norm()
                self.optimizer.step()

                # Detach state from history
                state = (Variable(state[0].data).cuda(), Variable(state[1].data).cuda())

                training_loss.update(loss.data[0])
                rotation_loss.update(r_loss.data[0])
                translation_loss.update(t_loss.data[0])
                gradient_norm.update(grad_norm)

                # Print log info
                if (j + 1) % self.print_freq == 0:
                    print('Sequence [{:d}/{:d}], Batch [{:d}/{:d}], '
                          'Total Loss: {: .4f} ({: .4f}), '
                          'R. Loss: {: .4f} ({: .4f}), '
                          'T. Loss: {: .4f} ({: .4f}), '
                          'Grad Norm: {: .4f}'
                          .format(i + 1, num_sequences,
                                  j + 1, num_batches,
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

    def validate(self):
        return self.test(testset=self.validationset)

    def test(self, testset=None):
        if not testset:
            testset = self.testset

        avg_loss = AverageMeter()
        avg_rot_loss = AverageMeter()
        avg_trans_loss = AverageMeter()

        all_predictions = []
        all_targets = []
        rel_angle_error_over_time = []

        self.model.eval()
        for i, dataloader in enumerate(testset):

            state = None

            for j, (images, poses, filenames) in enumerate(dataloader):

                input = self.to_variable(images, volatile=True)
                target = self.to_variable(poses, volatile=True)

                prev_pose = Variable(torch.zeros(1, 7)).cuda()
                outputs = Variable(torch.zeros(target.size(0), 7)).cuda()
                for k in range(input.size(0)):
                    frame = input[k]

                    prediction, state = self.model(frame, prev_pose, state)

                    outputs[k] = prediction
                    # In testing, feed the predicted pose from the previous frame
                    prev_pose = prediction

                outputs = self.normalize_output(outputs)

                # Loss function
                # Ignore first output
                loss, r_loss, t_loss = self.loss_function(outputs[1:], target[1:])

                avg_loss.update(loss.data[0])
                avg_rot_loss.update(r_loss.data[0])
                avg_trans_loss.update(t_loss.data[0])

                #print(filenames[0])

                # Visualize predicted path
                # fn = filenames[0][0].replace(os.path.sep, '--').replace('..', '')
                # of1 = self.make_output_filename('path/{}--{:05}.png'.format(fn, i))
                # of2 = self.make_output_filename('axis/{}--{:05}.png'.format(fn, i))
                # p = output.data.cpu().numpy()
                # t = target.data.cpu().numpy()
                # visualize_predicted_path(p, t, of1, show_rot=False)
                # plots.plot_xyz_error(p, t, of2)
                #
                # # Visualize keypoints
                # filename = self.make_output_filename('keypoints/{:4d}.png'.format(i))
                # plots.plot_extracted_keypoints(images, keypoints.cpu(), save=filename)


        # Average losses for rotation, translation and combined
        avg_loss = avg_loss.average
        avg_rot_loss = avg_rot_loss.average
        avg_trans_loss = avg_trans_loss.average

        # all_predictions = torch.cat(all_predictions, 0)
        # all_targets = torch.cat(all_targets, 0)
        # rel_angle_error_over_time = torch.Tensor(rel_angle_error_over_time).mean(0).view(-1)
        #
        # # Relative rotation angle between estimated and target rotation
        # rot_error_logger = self.make_logger('relative_rotation_angles.log')
        # rot_error_logger.clear()
        # rot_error_logger.column('Relative rotation angle between prediction and target', format='{:.4f}')
        # pose_errors = self.relative_rotation_angles2(all_predictions, all_targets)
        # for err in pose_errors:
        #     rot_error_logger.log(err)
        #
        # # The distribution of relative rotation angle
        # thresholds, cdf = self.error_distribution(torch.Tensor(pose_errors))
        # plots.plot_error_distribution(thresholds, cdf, self.make_output_filename('rotation_error_distribution.pdf'))
        # self.test_logger.clear()
        # self.test_logger.print('Cumulative distribution of rotation error:')
        # self.test_logger.print('Threshold: ' + ', '.join([str(t) for t in thresholds]))
        # self.test_logger.print('Fraction:  ' + ', '.join([str(p) for p in cdf]))
        # self.test_logger.print()
        # self.test_logger.print('Average combined loss on testset: {:.4f}'.format(avg_loss))
        # self.test_logger.print('Average rotation loss on testset: {:.4f}'.format(avg_rot_loss))
        # self.test_logger.print('Average translation loss on testset: {:.4f}'.format(avg_trans_loss))
        #
        # plots.plot_sequence_error(list(rel_angle_error_over_time), self.make_output_filename('average_rotation_error_over_time.pdf'))
        # self.test_logger.clear()
        # self.test_logger.print('Average relative rotation error over time:')
        # self.test_logger.print(', '.join([str(i) for i in rel_angle_error_over_time]))
        # self.test_logger.print()

        return avg_loss, avg_rot_loss, avg_trans_loss



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
