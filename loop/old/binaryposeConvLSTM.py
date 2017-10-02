from base import BaseExperiment, AverageMeter, CHECKPOINT_BEST_FILENAME
from dataimport.ImageNet import DiscretePoseGenerator, BinaryPoseSequenceGenerator, FOLDERS
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import plots
import time
import random
from convolution_lstm import ConvLSTM, ConvLSTMShrink
from flownet.models.FlowNetS import flownets
import torch
from torch.autograd import Variable


class PoseConvLSTM(nn.Module):

    def __init__(self, input_size, input_channels, hidden_channels, shrink, kernel_size):
        super(PoseConvLSTM, self).__init__()

        self.input_size = input_size
        self.clstm = ConvLSTMShrink(input_channels, hidden_channels, shrink, kernel_size, bias=True)

        out, _ = self.clstm.forward(Variable(torch.rand(1, 3, input_size[0], input_size[1])))
        print('Outsize:', out.size())
        self.fc = nn.Linear(out.size(1) * out.size(2) * out.size(3), 2)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()

    def forward(self, input):
        # Input format: [sequence_length, channels, h, w]
        hidden = None
        outputs = []
        for i in range(input.size(0)):
            x = input[i, :, :, :].unsqueeze(0)
            output, hidden = self.clstm.forward(x, hidden)
            outputs.append(output.view(1, -1))

        # Apply linear layer to all outputs except first one
        outputs = torch.cat(outputs[1:], 0)
        classifications = self.fc(outputs)

        return classifications, hidden

    def get_parameters(self):
        return list(self.clstm.parameters()) + list(self.fc.parameters())


class BinaryPoseConvLSTM(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--angle', type=float, default=10,
                            help='The maximum range of rotation of the images.')
        parser.add_argument('--zplane', type=float, default=1,
                            help='Location of the image in front of the camera (along Z-axis).')
        parser.add_argument('--max_size', type=int, nargs=3, default=[0, 0, 0],
                            help="""Clips training-, validation-, and testset at the given size. 
                            A zero signalizes that the whole dataset should be used.""")
        parser.add_argument('--sequence', type=int, default=10,
                            help='Length of sequence fed to the LSTM')

    def __init__(self, folder, args):
        super(BinaryPoseConvLSTM, self).__init__(folder, args)

        channels, height, width = 3, 224, 224
        hidden_channels = [128, 64, 64, 32, 32, 16, 16]
        hidden_channels.reverse()

        shrink = [2, None, 2, None, 2, None, 2]

        self.clstm = PoseConvLSTM((height, width), channels, hidden_channels, shrink, 3)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.clstm.get_parameters(), self.lr)

        if self.use_cuda:
            print('Moving LSTM to GPU ...')
            self.clstm.cuda()
            self.criterion.cuda()

        self.sequence_length = args.sequence
        self.print_freq = args.print_freq

        self.training_loss = []
        self.validation_loss = []

    def load_dataset(self, args):
        traindir = '../data/simple'#FOLDERS['training']
        valdir = traindir #FOLDERS['validation']
        testdir = traindir #FOLDERS['test']

        # Image pre-processing
        # For training set
        transform1 = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                transforms.Scale(256),
        ])

        # After homography is applied to image
        transform2 = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # For normalization, see https://github.com/pytorch/vision#models
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 #std=[0.229, 0.224, 0.225])
        ])

        sequence = args.sequence
        step = 20

        train_set = BinaryPoseSequenceGenerator(traindir, sequence_length=sequence, max_angle=args.angle, step_angle=step, z_plane=args.zplane,
                                                transform1=transform1, transform2=transform2, max_size=args.max_size[0])
        val_set   = BinaryPoseSequenceGenerator(valdir, sequence_length=sequence, max_angle=args.angle, step_angle=step, z_plane=args.zplane,
                                                transform1=transform1, transform2=transform2, max_size=args.max_size[1])
        test_set  = BinaryPoseSequenceGenerator(testdir, sequence_length=sequence, max_angle=args.angle, step_angle=step, z_plane=args.zplane,
                                                transform1=transform1, transform2=transform2, max_size=args.max_size[2])

        # Export some examples from the generated dataset
        train_set.visualize = self.out_folder
        inds = random.sample(range(len(train_set)), max(min(10, args.max_size[0]), 1))
        for i in inds:
            tmp = train_set[i]
        train_set.visualize = None

        dataloader_train = DataLoader(train_set, batch_size=1,
                                      shuffle=True, num_workers=args.workers)

        dataloader_val = DataLoader(val_set, batch_size=1,
                                    shuffle=False, num_workers=args.workers)

        dataloader_test = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=args.workers)

        #TODO: undo freeze
        self.frozen = [e for e in train_set]

        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        sample_loss = []
        num_batches = len(self.trainingset)

        epoch = len(self.training_loss) + 1
        #self.adjust_learning_rate(epoch)

        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        # TODO: undo freeze
        for i, (images, poses) in enumerate(self.frozen):

            # Shape: images -> [1, sequence length, channels, h, w]
            #        poses  -> [1, sequence length, 1]

            images.squeeze_(0)
            poses.squeeze_(0)

            inputs = self.to_variable(images)
            targets = self.to_variable(poses)

            self.optimizer.zero_grad()

            outputs, _ = self.clstm.forward(inputs)

            _, ind = torch.max(outputs.data, 1)
            print('Prediction: ', ind.view(1, -1))
            print('Target:     ', targets.view(1, -1))

            #print('outputs', outputs)
            #print('targets', targets)

            loss = self.criterion(outputs, targets)

            #print('loss', loss)

            loss.backward()
            self.optimizer.step()

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Sample [{:d}/{:d}], Loss: {:.4f}'.format(i + 1, num_batches, loss.data[0]))
                sample_loss.append(loss.data[0])
                filename = self.make_output_filename('sample-loss-epoch-{}.pdf'.format(epoch))
                plots.plot_sample_loss(sample_loss, save=filename)

            training_loss.update(loss.data[0])

        training_loss = training_loss.average
        self.training_loss.append(training_loss)

        # Validate after each epoch
        validation_loss, _ = self.test(dataloader=self.validationset)
        self.validation_loss.append(validation_loss)

        # Save extra checkpoint for best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(self.make_checkpoint(), self.make_output_filename(CHECKPOINT_BEST_FILENAME))

    def test(self, dataloader=None):
        if not dataloader:
            dataloader = self.testset

        num_predictions = len(dataloader) * (self.sequence_length - 1)
        accuracy = 0
        avg_loss = AverageMeter()
        for i, (images, poses) in enumerate(dataloader):

            images.squeeze_(0)
            poses.squeeze_(0)

            inputs = self.to_variable(images)
            targets = self.to_variable(poses)

            outputs, _ = self.clstm.forward(inputs)

            # argmax = predicted class
            _, ind = torch.max(outputs.data, 1)

            # print(outputs.data)
            # print(targets.data)
            # print(ind)

            # Correct predictions in the batch
            accuracy += torch.sum(torch.eq(ind, targets.data))

            loss = self.criterion(outputs, targets)
            avg_loss.update(loss.data[0])

        accuracy /= num_predictions
        avg_loss = avg_loss.average

        print('Accuracy: {:.4f}'.format(accuracy))
        return avg_loss, accuracy

    def make_checkpoint(self):
        checkpoint = {
            'epoch': len(self.training_loss),
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'model': self.clstm.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return checkpoint

    def restore_from_checkpoint(self, checkpoint):
        self.training_loss = checkpoint['training_loss']
        self.validation_loss = checkpoint['validation_loss']
        self.clstm.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def plot_performance(self):
        checkpoint = self.load_checkpoint()
        plots.plot_epoch_loss(checkpoint['training_loss'], checkpoint['validation_loss'], save=self.save_loss_plot)
