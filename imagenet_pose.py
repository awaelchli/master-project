from poseLSTM import PoseLSTM
from dataimport import KITTI, Dummy, ImageNet
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

    datadir = '../../data/ImageNet/ILSVRC2012'
    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'train')

    angle = 45
    z = 0.7

    # Image pre-processing
    # For training set
    transform1 = None
    if not args.image_size:
        transform1 = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

    # For validation set
    transform2 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
    ])

    # After homography is applied to image
    transform3 = transforms.Compose([
        transforms.ToTensor(),
        # For normalization, see https://github.com/pytorch/vision#models
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    train_set = ImageNet.PoseImageNet(traindir, max_angle=angle, z_plane=z, transform1=transform1, transform2=transform3)
    val_set = ImageNet.PoseImageNet(valdir, max_angle=angle, z_plane=z, transform1=transform2, transform2=transform3)

    dataloader_train = DataLoader(train_set, batch_size=1,
                                  shuffle=True, num_workers=args.workers)

    dataloader_val = DataLoader(val_set, batch_size=1,
                                shuffle=False, num_workers=args.workers)

    # Model for binary classification
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)

    if use_cuda:
        print('Moving CNN to GPU ...')
        model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                #momentum=args.momentum,
                                #weight_decay=args.weight_decay)
                                )

    if use_cuda:
        criterion.cuda()

    # Train the model
    print('Training ...')

    start_time = time.time()
    train_loss = []
    validation_loss = []

    for epoch in range(args.epochs):

        # Train for one epoch
        epoch_loss = train(model, criterion, optimizer, dataloader_train, epoch)
        train_loss.append(epoch_loss)

        # Validate after each epoch
        epoch_loss, _ = validate(model, criterion, dataloader_val)
        validation_loss.append(epoch_loss)

        # # TODO: write validation loss
        # with open(loss_file, 'a') as f:
        #     f.write('{}\n'.format(epoch_loss))
        #
        # # TODO: save loses
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'train_loss': None,
        #     'validation_loss': None,
        #     'state_dict': lstm.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, save_model_name)

        plots.plot_epoch_loss(train_loss, validation_loss, save=save_loss_plot)

    elapsed_time = time.time() - start_time
    print('Done. Elapsed time: {:.4f} hours.'.format(elapsed_time / 3600))

    # Evaluate model on testset
    print('Testing ...')
    test_loss, accuracy = test(model, criterion, dataloader_val)
    print('Done. Accuracy on testset: {:.4f}'.format(accuracy))


def train(model, criterion, optimizer, dataloader, epoch):
    epoch_loss = 0
    num_train_samples = len(dataloader)

    for i, (image, pose) in enumerate(dataloader):

        # Remove singleton batch dimension from data loader
        #images.squeeze_(0)

        # Reshape target pose from [batch, 1, 6] to [1, sequence_length, 6]
        # poses = poses.permute(1, 0, 2)
        #image_size = image.size()[1:4]
        #print('Image size:', image_size)

        input = to_variable(image)
        target = to_variable(pose)

        model.zero_grad()
        output = model(input)

        #print('Output:', output)
        #print('Target:', target)

        loss = criterion(output, target)
        #print(criterion.forward(input, target))
        #print(loss)
        loss.backward()
        optimizer.step()

        # Print log info
        if (i + 1) % args.print_freq == 0:
            print('Epoch [{:d}/{:d}], Sample [{:d}/{:d}], Loss: {:.4f}'
                  .format(epoch + 1, args.epochs, i + 1, num_train_samples, loss.data[0]))

        epoch_loss += loss.data[0]

    epoch_loss /= num_train_samples
    return epoch_loss


def validate(model, criterion, dataloader):
    return test(model, criterion, dataloader)


def test(model, criterion, dataloader):
    accuracy = 0
    avg_loss = 0
    for i, (image, pose) in enumerate(dataloader):

        input = to_variable(image, volatile=True)
        target = to_variable(pose, volatile=True)

        model.zero_grad()
        output = model(input)

        # argmax = predicted class
        _, ind = torch.max(output.data, 1)

        # Correct predictions in the batch
        accuracy += torch.sum(torch.eq(ind, pose))

        loss = criterion(output, target)
        avg_loss += loss.data[0]

    avg_loss /= len(dataloader)
    accuracy /= len(dataloader)
    return avg_loss, accuracy


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
    