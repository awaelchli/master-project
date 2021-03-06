from poseLSTM import PoseLSTM
from dataimport import KITTI, Dummy
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
import argparse
import os
import plots
import time
import shutil
import numpy as np


def setup_environment():
    os.makedirs(out_base_folder, exist_ok=True)
    # Wipe all existing data
    if os.path.isdir(out_folder):
        shutil.rmtree(out_folder)

    os.makedirs(out_folder)


def main():
    sequence_length = args.sequence

    # Image pre-processing
    transform = get_transform(normalize=False)

    kitti_train = KITTI.Subsequence(sequence_length, transform, args.grayscale,
                                    sequence_numbers=KITTI.SEQUENCES['training'])

    kitti_val = KITTI.Subsequence(sequence_length, transform, args.grayscale,
                                  sequence_numbers=KITTI.SEQUENCES['validation'])

    kitti_test = KITTI.Subsequence(sequence_length, transform, args.grayscale,
                                   sequence_numbers=KITTI.SEQUENCES['test'])

    image_size = kitti_train[0][0].size()[1:4]
    print('Image size:', image_size)

    dataloader_train = DataLoader(kitti_train, batch_size=1,
                                  shuffle=True, num_workers=args.workers)

    dataloader_val = DataLoader(kitti_val, batch_size=1,
                                shuffle=False, num_workers=args.workers)

    dataloader_test = DataLoader(kitti_test, batch_size=1,
                                 shuffle=False, num_workers=args.workers)

    # VGG without classifier
    # Input tensor dimensions: [batch, channels, height, width]
    # Output tensor dimensions: [batch, channels2, height2, width2]
    vgg = models.vgg19(pretrained=True).features
    # Freeze params, no gradient computation required
    for param in vgg.parameters():
        param.requires_grad = False

    if use_cuda:
        print('Moving CNN to GPU ...')
        vgg.cuda()

    # LSTM to predict pose sequence
    # Input size to LSTM is determined by output of pre-CNN
    print('Determine output size of CNN ...')
    input_size = cnn_feature_size(vgg, image_size[1], image_size[2])
    lstm = PoseLSTM(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.layers)

    # Loss and Optimizer
    criterion = nn.MSELoss()

    if use_cuda:
        print('Moving LSTM to GPU ...')
        lstm.cuda()
        criterion.cuda()

    optimizer = torch.optim.Adam(params=lstm.get_parameters(), lr=args.learning_rate)

    # Train the model
    print('Training ...')

    start_time = time.time()
    train_loss = []
    validation_loss = []

    for epoch in range(args.epochs):

        # Train for one epoch
        epoch_loss = train(vgg, lstm, criterion, optimizer, dataloader_train, epoch)
        train_loss.append(epoch_loss)

        # Validate after each epoch
        epoch_loss = validate(vgg, lstm, criterion, dataloader_val)
        validation_loss.append(epoch_loss)

        # TODO: write validation loss
        with open(loss_file, 'a') as f:
            f.write('{}\n'.format(epoch_loss))

        # TODO: save loses
        save_checkpoint({
            'epoch': epoch + 1,
            'train_loss': None,
            'validation_loss': None,
            'state_dict': lstm.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_model_name)

        plots.plot_epoch_loss(train_loss, validation_loss, save=save_loss_plot)

    elapsed_time = time.time() - start_time
    print('Done. Elapsed time: {:.4f} hours.'.format(elapsed_time / 3600))

    # Evaluate model on testset
    print('Testing ...')
    test_loss = test(vgg, lstm, criterion, dataloader_test)
    print('Done. Loss on testset: {:.4f}'.format(test_loss))


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

        sequence_length = poses.size(1)

        # Output vector: [x, y, z, ax, ay, phi]
        # az = sqrt(1 - ax - ay)
        ax = lstm_output[1, :, 3]
        ay = lstm_output[1, :, 4]
        az = torch.sqrt(1 - ax - ay)
        phi = lstm_output[1, :, 5]
        axis = torch.cat((ax, ay, az), 2)

        # Elements of quaternion
        q = torch.cat((torch.cos(phi / 2), torch.sin(phi / 2) * axis), 2)

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


def validate(pre_cnn, lstm, criterion, dataloader):
    return test(pre_cnn, lstm, criterion, dataloader)


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


def save_checkpoint(state, filename):
    torch.save(state, filename)


def get_transform(normalize=False):
    transform_list = []
    # Image resize
    if args.image_size:
        transform_list.append(transforms.Scale(args.image_size))
    # Converts image to tensor with values in range [0, 1]
    transform_list.append(transforms.ToTensor())
    # For normalization, see https://github.com/pytorch/vision#models
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--cnn_mode', type=int, choices=[0, 1], default=1,
                        help='0: Sequential mode 1: Batch mode')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='Frequency of printed information during training')

    parser.add_argument('--experiment', type=str, default='unnamed',
                        help='Name of the experiment. Output files will be stored in this folder.')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=1000,
                        help='dimension of lstm hidden states')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of layers in LSTM')

    # Data loading parameters
    parser.add_argument('--sequence', type=int, default=10,
                        help='Sequence length')

    parser.add_argument('--image_size', type=int, default=None,
                        help='Input images will be scaled such that the shorter side is equal to the given value.')

    parser.add_argument('--grayscale', action='store_true',
                        help='Convert images to grayscale.')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    print(args)

    out_base_folder = 'out'
    out_folder = os.path.join(out_base_folder, args.experiment)
    loss_file = os.path.join(out_folder, 'loss.txt')
    save_loss_plot = os.path.join(out_folder, 'loss.pdf')
    save_model_name = os.path.join(out_folder, 'checkpoint.pth.tar')

    use_cuda = torch.cuda.is_available() and args.cuda

    if torch.cuda.is_available():
        print('CUDA is available on this machine.')
    else:
        print('CUDA is not available on this machine.')

    setup_environment()
    main()
    