from poseLSTM import PoseLSTM
from dataimport import KITTI, Dummy
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import numpy as np

#root_dir = '../data/KITTI/grayscale/sequences'
root_dir = '../data/KITTI/color/sequences'
#pose_dir = '../data/KITTI odometry/poses_converted/'
pose_dir = '../data/KITTI/poses/'


def main():

    sequence_length = 10
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.Scale(100),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406),
                             #(0.229, 0.224, 0.225))])
        ])

    #kitti_sequence = KITTI.Sequence(root_dir, pose_dir, transform=transform, sequence_number = 2)
    kitti_sequence = KITTI.Subsequence(sequence_length, root_dir,
                                       pose_dir, transform,
                                       sequence_numbers=[0, 1, 2],
                                       eye=2)

    #image_size = kitti_sequence[0][0].size()
    image_size = kitti_sequence[0][0].size()[1:4]
    print('Image size:', image_size)

    #display_torch_image(kitti_sequence[0][0])

    #dummy = Dummy.Random(size=2, width=50, height=50)

    dataloader = DataLoader(kitti_sequence, batch_size = 1,#sequence_length,
                            shuffle = False, num_workers = args.workers)

    # VGG without classifier
    # Input tensor dimensions: [batch, channels, height, width]
    # Output tensor dimensions: [batch, channels2, height2, width2]
    vgg = models.vgg19(pretrained=True).features

    # LSTM to predict pose sequence
    # Input size to LSTM is determined by output of pre-CNN
    input_size = cnn_feature_size(vgg, image_size[1], image_size[2])
    lstm = PoseLSTM(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.layers)

    # Loss and Optimizer
    criterion = nn.MSELoss()

    if use_cuda:
        lstm.cuda()
        criterion.cuda()

    params = list(lstm.lstm.parameters()) + list(lstm.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(dataloader)
    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (images, poses) in enumerate(dataloader):

            # Remove singleton batch dimension from data loader
            images.squeeze_(0)

            # Reshape target pose from [batch, 1, 6] to [1, sequence_length, 6]
            #poses = poses.permute(1, 0, 2)

            cnn_output = apply_cnn_to_sequence(vgg, images, batch_mode=args.cnn_mode)
            lstm_input = Variable(reshape_cnn_output(cnn_output))
            lstm_target = Variable(poses)

            if use_cuda:
                lstm_input = lstm_input.cuda()
                lstm_target = lstm_target.cuda()

            print('Input sequence to LSTM', lstm_input.size())
            print('Target sequence to LSTM', lstm_target.size())

            lstm.zero_grad()
            lstm_output, lstm_hidden = lstm(lstm_input)

            print('LSTM output size:', lstm_output.size())
            print('LSTM hidden size:', lstm_hidden[0].size())
            print('LSTM state size:', lstm_hidden[1].size())

            #print('Output:', lstm_output)
            #print('Target:', lstm_target)

            loss = criterion(lstm_output, lstm_target)
            loss.backward()
            optimizer.step()

            # Print log info
            print('Epoch [{:d}/{:d}], Step [{:d}/{:d}], Loss: {:.4f}'
                  .format(epoch + 1, args.epochs, i + 1, total_step, loss.data[0]))

            epoch_loss += loss.data[0]

        epoch_loss /= len(dataloader)
        with open('loss.txt', 'a') as f:
            f.write('{}\n'.format(epoch_loss))


def apply_cnn_to_sequence(cnn, images, batch_mode=True):
    if use_cuda:
        cnn.cuda()
        images = images.cuda()

    if batch_mode:
        print('Forward sequence using CNN in batch mode.')
        print('Input size:', images.size())
        features = cnn(Variable(images))
        return features.data
    else:
        print('Forward sequence using CNN sequentially.')
        # Transform all images in the sequence to features for the LSTM
        batch_size = images.size(0)
        input_sequence = []
        target_sequence = []
        for i in range(0, batch_size):
            input = Variable(images[i, :].unsqueeze(0))
            print(input.size())
            output = cnn(input)
            input_sequence.append(output.data)

        features = torch.cat(tuple(input_sequence), 0)
        # Shape: [sequence length, channels, height, width]
        return features


def reshape_cnn_output(cnn_output):
    sequence_lenth = cnn_output.size(0)
    feature_size = cnn_output.size(1) * cnn_output.size(2) * cnn_output.size(3)
    return cnn_output.view(1, sequence_lenth, feature_size)


def cnn_feature_size(cnn, input_height, input_width):
    input = Variable(torch.zeros(1, 3, input_height, input_width))
    output = cnn(input)
    return output.size(1) * output.size(2) * output.size(3)


def display_torch_image(img):
    tf = transforms.ToPILImage()
    img = tf(img)
    img.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--cnn_mode', type=int, choices=[0, 1], default=1,
                        help='0: Sequential mode 1: Batch mode')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=1000,
                        help='dimension of lstm hidden states')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of layers in LSTM')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        print('CUDA is available on this machine.')
    else:
        print('CUDA is not available on this machine.')

    use_cuda = torch.cuda.is_available() and args.cuda

    main()