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
import os
import plots
import time
import shutil
import numpy as np

#root_dir = '../data/KITTI/grayscale/sequences'
root_dir = '../data/KITTI/color/sequences'
pose_dir = '../data/KITTI/poses/'
out_folder = 'out'
loss_file = os.path.join(out_folder, 'loss.txt')
save_loss_plot = os.path.join(out_folder, 'loss.pdf')
save_model_name = os.path.join(out_folder, 'checkpoint.pth.tar')


def setup_environment():
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)


def main():
    sequence_length = args.sequence

    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.Scale(100),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406),
                             #(0.229, 0.224, 0.225))])
        ])

    kitti_train = KITTI.Subsequence(sequence_length, root_dir,
                                    pose_dir, transform,
                                    sequence_numbers=[0, 1, 2, 3, 4, 5, 6, 7],
                                    eye=2)

    kitti_test = KITTI.Subsequence(sequence_length, root_dir,
                                   pose_dir, transform,
                                   sequence_numbers=[9, 10],
                                   eye=2)

    image_size = kitti_train[0][0].size()[1:4]
    print('Image size:', image_size)

    #display_torch_image(kitti_sequence[0][0])
    #dummy = Dummy.Random(size=2, width=50, height=50)

    dataloader_train = DataLoader(kitti_train, batch_size=1,
                                  shuffle=True, num_workers=args.workers)

    dataloader_test = DataLoader(kitti_test, batch_size=1,
                                 shuffle=False, num_workers=args.workers)

    # VGG without classifier
    # Input tensor dimensions: [batch, channels, height, width]
    # Output tensor dimensions: [batch, channels2, height2, width2]
    vgg = models.vgg19(pretrained=True).features
    if use_cuda:
        vgg.cuda()

    # LSTM to predict pose sequence
    # Input size to LSTM is determined by output of pre-CNN
    input_size = cnn_feature_size(vgg, image_size[1], image_size[2])
    lstm = PoseLSTM(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.layers)

    # Loss and Optimizer
    criterion = nn.MSELoss()

    if use_cuda:
        lstm.cuda()
        criterion.cuda()

    optimizer = torch.optim.Adam(params=lstm.get_parameters(), lr=args.learning_rate)

    # Train the model
    print('Training...')
    start_time = time.time()
    for epoch in range(args.epochs):

        # Train for one epoch
        epoch_loss = train(vgg, lstm, criterion, optimizer, dataloader_train, epoch)

        with open(loss_file, 'a') as f:
            f.write('{}\n'.format(epoch_loss))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': lstm.state_dict(),
            'optimizer': optimizer.state_dict(),
        })

    elapsed_time = time.time() - start_time
    print('Elapsed time is {:f} hours.'.format(elapsed_time / 3600))

    # Evaluate model on testset
    test_loss = test(vgg, lstm, criterion, dataloader_test)
    print('Loss on testset: {:.4f}'.format(test_loss))

    # Produce plots
    plots.plot_loss_from_file(loss_file, save=save_loss_plot)


def train(pre_cnn, lstm, criterion, optimizer, dataloader, epoch):
    epoch_loss = 0
    num_train_samples = len(dataloader)

    for i, (images, poses) in enumerate(dataloader):

        # Remove singleton batch dimension from data loader
        images.squeeze_(0)

        # Reshape target pose from [batch, 1, 6] to [1, sequence_length, 6]
        # poses = poses.permute(1, 0, 2)

        cnn_output = apply_cnn_to_sequence(pre_cnn, images, batch_mode=args.cnn_mode)
        lstm_input = to_variable(reshape_cnn_output(cnn_output))
        lstm_target = to_variable(poses)

        #print('Input sequence to LSTM', lstm_input.size())
        #print('Target sequence to LSTM', lstm_target.size())

        lstm.zero_grad()
        lstm_output, lstm_hidden = lstm(lstm_input)

        #print('LSTM output size:', lstm_output.size())
        #print('LSTM hidden size:', lstm_hidden[0].size())
        #print('LSTM state size:', lstm_hidden[1].size())

        # print('Output:', lstm_output)
        # print('Target:', lstm_target)

        loss = criterion(lstm_output, lstm_target)
        loss.backward()
        optimizer.step()

        # Print log info
        if (i + 1) % args.print_freq == 0:
            print('Epoch [{:d}/{:d}], Sample [{:d}/{:d}], Loss: {:.4f}'
                  .format(epoch + 1, args.epochs, i + 1, num_train_samples, loss.data[0]))

        epoch_loss += loss.data[0]

    epoch_loss /= num_train_samples
    return epoch_loss


def test(pre_cnn, lstm, criterion, dataloader):
    avg_loss = 0
    for i, (images, poses) in enumerate(dataloader):
        images.squeeze_(0)
        cnn_output = apply_cnn_to_sequence(pre_cnn, images, batch_mode=args.cnn_mode)
        lstm_input = to_variable(reshape_cnn_output(cnn_output), volatile=True)
        lstm_target = to_variable(poses, volatile=True)

        lstm.zero_grad()
        lstm_output, lstm_hidden = lstm(lstm_input)
        loss = criterion(lstm_output, lstm_target)
        avg_loss += loss.data[0]

    avg_loss /= len(dataloader)
    return avg_loss


def apply_cnn_to_sequence(cnn, images, batch_mode=True):

    if batch_mode:
        #print('Forward sequence using CNN in batch mode.')
        #print('Input size:', images.size())
        features = cnn(to_variable(images, volatile=True))
        return features.data
    else:
        #print('Forward sequence using CNN sequentially.')

        # Transform all images in the sequence to features for the LSTM
        batch_size = images.size(0)
        input_sequence = []
        for i in range(0, batch_size):
            input = to_variable(images[i, :].unsqueeze(0), volatile=True)
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
    input = to_variable(torch.zeros(1, 3, input_height, input_width), volatile=True)
    output = cnn(input)
    return output.size(1) * output.size(2) * output.size(3)


def to_variable(data, volatile=False):
    var = Variable(data, volatile=volatile)
    if use_cuda:
        var = var.cuda()
    return var


def display_torch_image(img):
    tf = transforms.ToPILImage()
    img = tf(img)
    img.show()


def save_checkpoint(state, filename=save_model_name):
    torch.save(state, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--cnn_mode', type=int, choices=[0, 1], default=1,
                        help='0: Sequential mode 1: Batch mode')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='Frequency of printed information during training')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=1000,
                        help='dimension of lstm hidden states')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of layers in LSTM')
    parser.add_argument('--sequence', type=int, default=10,
                        help='Sequence length')

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

    setup_environment()
    main()