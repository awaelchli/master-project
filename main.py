from poseLSTM import PoseLSTM
from dataimport import KITTI, Dummy
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import argparse

root_dir = '../data/KITTI odometry/grayscale/sequences'
pose_dir = '../data/KITTI odometry/poses/'


def main():

    #sequence = KITTI.Sequence(root_dir, pose_dir, sequence_number = 2)
    dummy = Dummy.Random(size=2, width=50, height=50)

    #t = sequence.__getitem__(2)

    #print(t)

    dataloader = DataLoader(dummy, batch_size = 1,
                            shuffle = True, num_workers = args.num_workers)

    # LSTM to predict pose sequence
    # Input size to LSTM is determined by output of pre-CNN
    lstm = PoseLSTM(input_size=512, hidden_size=args.hidden_size, num_layers=args.num_layers)

    if use_cuda:
        lstm.cuda()

    # Loss and Optimizer
    criterion = nn.MSELoss()

    params = list(lstm.lstm.parameters()) + list(lstm.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(dataloader)
    for epoch in range(args.num_epochs):
        for i, (images, poses) in enumerate(dataloader):

            # Remove singleton batch dimension from data loader
            images.squeeze_(0)

            cnn_output = apply_cnn_to_sequence(images)
            lstm_input = Variable(reshape_cnn_output(cnn_output))
            lstm_target = Variable(poses)

            if use_cuda:
                lstm_input.cuda()
                lstm_target.cuda()

            print('Input sequence to LSTM', lstm_input.size())
            print('Target sequence to LSTM', lstm_target.size())

            lstm.zero_grad()
            lstm_output, lstm_hidden = lstm(lstm_input)

            print('LSTM output size:', lstm_output.size())
            print('LSTM hidden size:', lstm_hidden[0].size())
            print('LSTM state size:', lstm_hidden[1].size())

            loss = criterion(lstm_output, lstm_target)
            loss.backward()
            optimizer.step()

            # Print log info
            print('Epoch [{:d}/{:d}], Step [{:d}/{d}], Loss: {:.4f}'
                  .format(epoch, args.num_epochs, i, total_step, loss.data[0]))


def apply_cnn_to_sequence(images, batch_mode=True):
    # VGG without classifier
    # Input tensor dimensions: [batch, channels, height, width]
    # Output tensor dimensions: [batch, 512, 11, 38]
    vgg = models.vgg19(pretrained=True).features
    if use_cuda:
        vgg.cuda()

    if batch_mode:
        print('Forward sequence using CNN in batch mode.')
        print('Input size:', images.size())
        features = vgg(Variable(images))
        return features.data
    else:
        print('Forward sequence using CNN sequentially.')
        # Transform all images in the sequence to features for the LSTM
        batch_size = images.size(0)
        input_sequence = []
        target_sequence = []
        for i in range(0, batch_size):
            input = Variable(images[i, :])
            if use_cuda:
                input.cuda()

            output = vgg(input)
            input_sequence.append(output.data)

        features = torch.cat(tuple(input_sequence), 0)
        # Shape: [sequence length, channels, height, width]
        return features


def reshape_cnn_output(cnn_output):
    sequence_lenth = cnn_output.size(0)
    feature_size = cnn_output.size(1) * cnn_output.size(2) * cnn_output.size(3)
    return cnn_output.view(1, sequence_lenth, feature_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=4096,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in LSTM')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    print(args)

    use_cuda = torch.cuda.is_available() and args.cuda

    main()