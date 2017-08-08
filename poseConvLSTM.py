from convolution_lstm import ConvLSTM
from base import BaseExperiment, AverageMeter, CHECKPOINT_BEST_FILENAME
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import plots
from dataimport import KITTI


class PoseConvLSTM(nn.Module):

    def __init__(self, input_size, input_channels, hidden_channels, kernel_size):
        super(PoseConvLSTM, self).__init__()

        self.input_size = input_size
        self.clstm = ConvLSTM(input_channels, hidden_channels, kernel_size, bias=True)

        # The output size of the last cell defines the input size of the linear layer
        last_hidden_size = input_size[0] * input_size[1] * hidden_channels[-1]
        self.fc = nn.Linear(last_hidden_size, 6)

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

        # Using batchmode of fc layer to transform entire sequence
        outputs = torch.cat(outputs, 0)
        pose_sequence = self.fc(outputs)

        return pose_sequence, hidden

    def get_parameters(self):
        return list(self.clstm.parameters()) + list(self.fc.parameters())


class KITTIPoseConvLSTM(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--cnn_mode', type=int, choices=[0, 1], default=1,
                            help='0: Sequential mode 1: Batch mode')

        # Model parameters
        # parser.add_argument('--hidden_size', type=int, default=1000,
        #                     help='dimension of lstm hidden states')
        # parser.add_argument('--layers', type=int, default=1,
        #                     help='number of layers in LSTM')

        # Data loading parameters
        parser.add_argument('--sequence', type=int, default=10,
                            help='Sequence length')

        parser.add_argument('--image_size', type=int, default=None,
                            help='Input images will be scaled such that the shorter side is equal to the given value.')

        parser.add_argument('--grayscale', action='store_true',
                            help='Convert images to grayscale.')

    def __init__(self, folder, args):
        super(KITTIPoseConvLSTM, self).__init__(folder, args)

        # VGG without classifier, up to a certain amount of layers
        # Input tensor dimensions: [batch, channels, height, width]
        # Output tensor dimensions: [batch, channels2, height2, width2]
        layers = models.vgg19(pretrained=True).features
        self.pre_cnn = nn.Sequential()
        for i in range(20):
            self.pre_cnn.add_module('{}'.format(i), layers[i])

        # Freeze params, no gradient computation required
        for param in self.pre_cnn.parameters():
            param.requires_grad = False

        if self.use_cuda:
            print('Moving CNN to GPU ...')
            self.pre_cnn.cuda()

        # LSTM to predict pose sequence
        # Input size to LSTM is determined by output of pre-CNN
        print('Determine output size of CNN ...')
        channels, height, width = self.cnn_feature_size(self.image_size[1], self.image_size[2])
        self.model = PoseConvLSTM((height, width), channels, [64, 64, 32, 32], 3)

        print('Size of fc layer: {} x {}'.format(self.model.fc.in_features, self.model.fc.out_features))

        # Loss and Optimizer
        # TODO: check if needed
        self.criterion = nn.MSELoss()

        if self.use_cuda:
            print('Moving LSTM to GPU ...')
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch.optim.SGD(params=self.model.get_parameters(), lr=args.lr)

        self.cnn_mode = args.cnn_mode
        self.print_freq = args.print_freq
        self.training_loss = []
        self.validation_loss = []

    def load_dataset(self, args):
        sequence_length = args.sequence

        # Image pre-processing
        transform_list = []
        # Image resize
        if args.image_size:
            transform_list.append(transforms.Scale(args.image_size))
        # Converts image to tensor with values in range [0, 1]
        transform_list.append(transforms.ToTensor())
        # For normalization, see https://github.com/pytorch/vision#models
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225]))

        transform = transforms.Compose(transform_list)

        kitti_train = KITTI.Subsequence(sequence_length, transform, args.grayscale,
                                        sequence_numbers=KITTI.SEQUENCES['training'])

        kitti_val = KITTI.Subsequence(sequence_length, transform, args.grayscale,
                                      sequence_numbers=KITTI.SEQUENCES['validation'])

        kitti_test = KITTI.Subsequence(sequence_length, transform, args.grayscale,
                                       sequence_numbers=KITTI.SEQUENCES['test'])

        self.image_size = kitti_train[0][0].size()[1:4]
        print('Image size:', self.image_size)

        dataloader_train = DataLoader(kitti_train, batch_size=1,
                                      shuffle=True, num_workers=args.workers)

        dataloader_val = DataLoader(kitti_val, batch_size=1,
                                    shuffle=False, num_workers=args.workers)

        dataloader_test = DataLoader(kitti_test, batch_size=1,
                                     shuffle=False, num_workers=args.workers)

        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        num_batches = len(self.trainingset)

        epoch = len(self.training_loss) + 1
        self.adjust_learning_rate(epoch)

        best_validation_loss = float('inf') if not self.validation_loss else min(self.validation_loss)

        for i, (image, pose) in enumerate(self.trainingset):

            # Remove singleton batch dimension from data loader
            image.squeeze_(0)
            pose.squeeze_(0)

            cnn_output = self.apply_cnn_to_sequence(image, batch_mode=self.cnn_mode)
            lstm_input = self.to_variable(cnn_output)
            lstm_target = self.to_variable(pose)

            print('Input sequence to LSTM', lstm_input.size())
            # print('Target sequence to LSTM', lstm_target.size())

            self.model.zero_grad()
            lstm_output, lstm_hidden = self.model(lstm_input)

            # Output dimensions: [sequence_length, 6]
            print('LSTM output', lstm_output.size())

            # TODO continue
            #loss = self.criterion(lstm_output, lstm_target)
            loss = self.loss_function(lstm_output, lstm_target)
            loss.backward()
            self.optimizer.step()

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Sequence [{:d}/{:d}], Loss: {:.4f}'.format(i + 1, num_batches, loss.data[0]))

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

        avg_loss = AverageMeter()
        for i, (images, poses) in enumerate(dataloader):
            images.squeeze_(0)
            poses.squeeze_(0)

            cnn_output = self.apply_cnn_to_sequence(images, batch_mode=self.cnn_mode)
            lstm_input = self.to_variable(cnn_output, volatile=True)
            lstm_target = self.to_variable(poses, volatile=True)

            self.model.zero_grad()
            lstm_output, lstm_hidden = self.model(lstm_input)
            loss = self.loss_function(lstm_output, lstm_target)
            avg_loss.update(loss.data[0])

        return avg_loss

    def loss_function(self, output, target):
        # Dimensions: [sequence_length, 6]
        sequence_length = output.size(1)
        print(output)
        print(target)
        t1 = output[:, 0:3]
        t2 = target[:, 0:3]
        q1 = self.get_quaternion_pose(output)
        q2 = self.get_quaternion_pose(target)

        print(q1)
        print(q2)
        # Loss for rotation, dot product between quaternions
        q1dotq2 = torch.abs((q1 * q2).sum(1))
        loss1 = torch.sum(q1dotq2) / sequence_length

        # Loss for translation
        eps = 0.001

        #self.criterion(t1, t2)
        #t_diff = torch.pow(t1 - t2, 2).sum(2)
        loss2 = torch.log(eps + self.criterion(t1, t2))

        return loss1 + loss2

    def get_quaternion_pose(self, pose):
        # Output vector: [x, y, z, ax, ay, phi]
        # az = sqrt(1 - ax - ay)
        ax = pose[:, 3].contiguous().view(-1, 1)
        ay = pose[:, 4].contiguous().view(-1, 1)
        az = torch.sqrt(torch.abs(1 - ax - ay))

        phi = pose[:, 5].contiguous().view(-1, 1)
        phi_repl = phi.expand(phi.size(0), 3)

        axis = torch.cat((ax, ay, az), 1)

        # Elements of quaternion
        q = torch.cat((torch.cos(phi / 2), torch.sin(phi_repl / 2) * axis), 1)
        return q

    def apply_cnn_to_sequence(self, images, batch_mode=True):
        if batch_mode:
            # print('Forward sequence using CNN in batch mode.')
            # print('Input size:', images.size())
            features = self.pre_cnn(self.to_variable(images, volatile=True))
            return features.data
        else:
            # print('Forward sequence using CNN sequentially.')

            # Transform all images in the sequence to features for the LSTM
            batch_size = images.size(0)
            input_sequence = []
            for i in range(0, batch_size):
                input = self.to_variable(images[i, :].unsqueeze(0), volatile=True)
                print(input.size())
                output = self.pre_cnn(input)
                input_sequence.append(output.data)

            features = torch.cat(tuple(input_sequence), 0)
            # Shape: [sequence length, channels, height, width]
            return features

    # def reshape_cnn_output(self, cnn_output):
    #     sequence_lenth = cnn_output.size(0)
    #     feature_size = cnn_output.size(1) * cnn_output.size(2) * cnn_output.size(3)
    #     return cnn_output.view(1, sequence_lenth, feature_size)

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
