from base import BaseExperiment, AverageMeter, CHECKPOINT_BEST_FILENAME
from dataimport.ImageNet import DiscretePoseGenerator, FOLDERS
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import plots
import time
import random
from convolution_lstm import ConvLSTM
from flownet.models.FlowNetS import flownets
import torch
from torch.autograd import Variable


class PoseConvLSTM(nn.Module):

    def __init__(self, input_size, input_channels, hidden_channels, kernel_size):
        super(PoseConvLSTM, self).__init__()

        self.input_size = input_size
        self.clstm = ConvLSTM(input_channels, hidden_channels, kernel_size, bias=True)

        # The output size of the last cell defines the input size of the linear layer
        last_hidden_size = input_size[0] * input_size[1] * hidden_channels[-1]
        self.fc = nn.Linear(last_hidden_size, 2)

        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.zero_()

    def forward(self, input):
        # Input format: [sequence_length, channels, h, w]
        hidden = None
        output = None
        for i in range(input.size(0)):
            x = input[i, :, :, :].unsqueeze(0)
            output, hidden = self.clstm.forward(x, hidden)

        # Apply linear layer to last output
        output = self.fc(output.view(1, -1))

        return output, hidden

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

    def __init__(self, folder, args):
        super(BinaryPoseConvLSTM, self).__init__(folder, args)

        # Model for binary classification
        self.pre_cnn = nn.Sequential()

        model = flownets('../data/Pretrained Models/flownets_pytorch.pth')

        model2 = torch.nn.Sequential(
            model.conv1,
            model.conv2,
            model.conv3,
            model.conv3_1,
            model.conv4,
            model.conv4_1,
            model.conv5,
            model.conv5_1,
            model.conv6,
            model.conv6_1,
        )
        self.fc = nn.Linear(1024 * 4 * 4, 2)


        #layers = models.resnet18(pretrained=False)
        #layers.fc = nn.Linear(512, 2)

        #for i in range(17):
        #    self.pre_cnn.add_module('{}'.format(i), layers[i])
        self.pre_cnn = model2
        print(self.pre_cnn)

        if self.use_cuda:
            print('Moving CNN to GPU ...')
            self.pre_cnn.cuda()
            self.fc.cuda()

        #channels, height, width = self.cnn_feature_size(224, 224)
        #print(channels, height, width)

        hidden_channels = [128, 64, 64, 32, 32, 16, 16]
        #hidden_channels.reverse()

        #self.clstm = PoseConvLSTM((height, width), channels, hidden_channels, 3)

        self.criterion = nn.CrossEntropyLoss()

        params = list(self.fc.parameters()) #+ self.clstm.get_parameters()
        self.optimizer = torch.optim.Adam(params, self.lr,
                                         # momentum=args.momentum,
                                         # weight_decay=args.weight_decay)
                                         )

        if self.use_cuda:
            #print('Moving LSTM to GPU ...')
            #self.clstm.cuda()
            self.criterion.cuda()

        self.print_freq = args.print_freq

        self.training_loss = []
        self.validation_loss = []

    def load_dataset(self, args):
        traindir = FOLDERS['training']
        valdir = FOLDERS['validation']
        testdir = FOLDERS['test']

        # Image pre-processing
        # For training set
        transform1 = transforms.Compose([
                #transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])

        # For validation set
        transform2 = transforms.Compose([
            #transforms.Scale(256),
            #transforms.CenterCrop(224),
        ])

        # After homography is applied to image
        transform3 = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # For normalization, see https://github.com/pytorch/vision#models
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 #std=[0.229, 0.224, 0.225])
        ])

        train_set = DiscretePoseGenerator(traindir, max_angle=args.angle, z_plane=args.zplane, transform1=transform1,
                                          transform2=transform3, max_size=args.max_size[0])
        val_set = DiscretePoseGenerator(valdir, max_angle=args.angle, z_plane=args.zplane, transform1=transform2,
                                        transform2=transform3, max_size=args.max_size[1])
        test_set = DiscretePoseGenerator(testdir, max_angle=args.angle, z_plane=args.zplane, transform1=transform2,
                                         transform2=transform3, max_size=args.max_size[2])

        # Export some examples from the generated dataset
        train_set.visualize = self.out_folder
        for x in range(10):
            i = random.randint(0, len(train_set) - 1)
            tmp = train_set[i]
        train_set.visualize = None

        dataloader_train = DataLoader(train_set, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers)

        dataloader_val = DataLoader(val_set, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)

        dataloader_test = DataLoader(test_set, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)

        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        sample_loss = []
        num_batches = len(self.trainingset)

        epoch = len(self.training_loss) + 1
        #self.adjust_learning_rate(epoch)

        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        for i, (images, pose) in enumerate(self.trainingset):

            #images.squeeze_(0)
            #pose.squeeze_(0)


            images = torch.cat((images[:, 0, :, :, :], images[:, 1, :, :, :]), 1)


            input = self.to_variable(images)
            target = self.to_variable(pose)

            self.optimizer.zero_grad()

            output = self.pre_cnn(input)
            output = self.fc(output.view(self.batch_size, -1))
            #output, _ = self.clstm(features)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Sample [{:d}/{:d}], Loss: {:.4f}'.format(i + 1, num_batches, loss.data[0]))
                sample_loss.append(loss.data[0])
                plots.plot_sample_loss(sample_loss, save='sample-loss-epoch-{}.pdf'.format(epoch))

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

        num_samples = len(dataloader) * dataloader.batch_size
        accuracy = 0
        avg_loss = AverageMeter()
        for i, (images, poses) in enumerate(dataloader):
            #images.squeeze_(0)
            #poses.squeeze_(0)

            images = torch.cat((images[:, 0, :, :, :], images[:, 1, :, :, :]), 1)

            input = self.to_variable(images, volatile=True)
            target = self.to_variable(poses, volatile=True)

            output = self.pre_cnn(input)
            output = self.fc(output.view(self.batch_size, -1))
            #output, _ = self.clstm(features)

            # argmax = predicted class
            _, ind = torch.max(output.data, 1)

            # Correct predictions in the batch
            accuracy += torch.sum(torch.eq(ind, target.data))

            loss = self.criterion(output, target)
            avg_loss.update(loss.data[0])

        accuracy /= num_samples
        avg_loss = avg_loss.average

        print('Accuracy: {:.4f}'.format(accuracy))
        return avg_loss, accuracy

    def cnn_feature_size(self, input_height, input_width):
        input = self.to_variable(torch.zeros(1, 3, input_height, input_width), volatile=True)
        output = self.pre_cnn(input)
        # channels, height, width
        return output.size(1), output.size(2), output.size(3)

    def make_checkpoint(self):
        checkpoint = {
            'epoch': len(self.training_loss),
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'model': self.pre_cnn.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return checkpoint

    def restore_from_checkpoint(self, checkpoint):
        self.training_loss = checkpoint['training_loss']
        self.validation_loss = checkpoint['validation_loss']
        self.pre_cnn.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def plot_performance(self):
        checkpoint = self.load_checkpoint()
        plots.plot_epoch_loss(checkpoint['training_loss'], checkpoint['validation_loss'], save=self.save_loss_plot)
