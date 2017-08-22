from base import BaseExperiment, AverageMeter, CHECKPOINT_BEST_FILENAME
from dataimport.ImageNet import DiscretePoseGenerator, BinaryPoseSequenceGenerator, FOLDERS
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import plots
import random
from flownet.models.FlowNetS import flownets
import torch
from torch.autograd import Variable
from convolution_lstm import ConvLSTM


class BinaryFlowNetPose(nn.Module):

    def __init__(self, input_size):
        super(BinaryFlowNetPose, self).__init__()

        flownet = flownets('../data/Pretrained Models/flownets_pytorch.pth')

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
        for param in self.layers.parameters():
            param.requires_grad = False

        temp = self.layers(Variable(torch.zeros(1, 6, input_size[0], input_size[1])))
        self.hidden = 500
        self.nlayers = 3
        self.lstm = nn.LSTM(input_size=temp.size(1) * temp.size(2) * temp.size(3),
                            hidden_size=self.hidden,
                            num_layers=self.nlayers,
                            batch_first=True)

        self.fc = nn.Linear(self.hidden, 1)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-10, 10)
        self.fc.bias.data.zero_()

    def forward(self, input):
        # Input shape: [sequence, channels, h, w]
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

        outputs, _ = self.lstm(pairs.view(1, n - 1, -1), (h0, c0))
        predictions = self.fc(outputs.squeeze(0))

        return predictions

    def get_parameters(self):
        return list(self.lstm.parameters()) + list(self.fc.parameters())


class BinaryPoseRegression(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--angle', type=float, default=10,
                            help='The maximum range of rotation of the images.')
        parser.add_argument('--step', type=float, default=5,
                            help='Increment in rotation angle for generating the sequence.')
        parser.add_argument('--zplane', type=float, default=1,
                            help='Location of the image in front of the camera (along Z-axis).')
        parser.add_argument('--max_size', type=int, nargs=3, default=[0, 0, 0],
                            help="""Clips training-, validation-, and testset at the given size. 
                            A zero signalizes that the whole dataset should be used.""")
        parser.add_argument('--sequence', type=int, default=10,
                            help='Length of sequence fed to the LSTM')

    def __init__(self, folder, args):
        super(BinaryPoseRegression, self).__init__(folder, args)

        # Model for binary classification
        self.model = BinaryFlowNetPose((224, 224))
        print(self.model)

        self.criterion = nn.MSELoss()

        if self.use_cuda:
            print('Moving model to GPU ...')
            self.model.cuda()
            self.criterion.cuda()

        params = self.model.get_parameters()
        self.optimizer = torch.optim.Adam(params, self.lr)

        self.print_freq = args.print_freq
        self.sequence_length = args.sequence
        self.step = args.step
        self.training_loss = []
        self.validation_loss = []

    def load_dataset(self, args):
        traindir = FOLDERS['training']
        valdir = FOLDERS['validation']
        testdir = FOLDERS['test']

        # Image pre-processing
        # For training set
        transform1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.Scale(256),
        ])

        # After homography is applied to image
        transform2 = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # For normalization, see https://github.com/pytorch/vision#models
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225])
        ])

        sequence = args.sequence
        step = args.step

        train_set = BinaryPoseSequenceGenerator(traindir, sequence_length=sequence, max_angle=args.angle,
                                                step_angle=step, z_plane=args.zplane,
                                                transform1=transform1, transform2=transform2, max_size=args.max_size[0])
        val_set = BinaryPoseSequenceGenerator(valdir, sequence_length=sequence, max_angle=args.angle, step_angle=step,
                                              z_plane=args.zplane,
                                              transform1=transform1, transform2=transform2, max_size=args.max_size[1])
        test_set = BinaryPoseSequenceGenerator(testdir, sequence_length=sequence, max_angle=args.angle, step_angle=step,
                                               z_plane=args.zplane,
                                               transform1=transform1, transform2=transform2, max_size=args.max_size[2])

        # Export some examples from the generated dataset
        train_set.visualize = self.out_folder
        inds = random.sample(range(len(train_set)), max(min(10, args.max_size[0]), 1))
        for i in inds:
            _ = train_set[i]
        train_set.visualize = None

        dataloader_train = DataLoader(train_set, batch_size=1, pin_memory=self.use_cuda,
                                      shuffle=True, num_workers=args.workers)

        dataloader_val = DataLoader(val_set, batch_size=1, pin_memory=self.use_cuda,
                                    shuffle=False, num_workers=args.workers)

        dataloader_test = DataLoader(test_set, batch_size=1, pin_memory=self.use_cuda,
                                     shuffle=False, num_workers=args.workers)

        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        sample_loss = []
        num_batches = len(self.trainingset)

        epoch = len(self.training_loss) + 1
        #self.adjust_learning_rate(epoch)

        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        for i, (images, _, angles) in enumerate(self.trainingset):

            images.squeeze_(0)
            angles.squeeze_(0)

            input = self.to_variable(images)
            target = self.to_variable(angles)

            self.optimizer.zero_grad()

            output = self.model(input)

            #_, ind = torch.max(output.data, 1)
            print('Prediction: ', output.view(1, -1))
            print('Target:     ', target.view(1, -1))

            loss = self.criterion(output, target)
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
        validation_loss = self.test(dataloader=self.validationset)
        self.validation_loss.append(validation_loss)

        # Save extra checkpoint for best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(self.make_checkpoint(), self.make_output_filename(CHECKPOINT_BEST_FILENAME))

    def test(self, dataloader=None):
        if not dataloader:
            dataloader = self.testset

        #num_predictions = len(dataloader) * (self.sequence_length - 1)
        avg_loss = AverageMeter()
        for i, (images, _, angles) in enumerate(dataloader):

            images.squeeze_(0)
            angles.squeeze_(0)

            input = self.to_variable(images, volatile=True)
            target = self.to_variable(angles, volatile=True)

            output = self.model(input)

            loss = self.criterion(output, target)
            avg_loss.update(loss.data[0])

        avg_loss = avg_loss.average

        print('Average loss: {:.4f}'.format(avg_loss))
        return avg_loss

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

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def plot_performance(self):
        checkpoint = self.load_checkpoint()
        plots.plot_epoch_loss(checkpoint['training_loss'], checkpoint['validation_loss'], save=self.save_loss_plot)
