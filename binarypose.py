from base import BaseExperiment

from dataimport import ImageNet
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import plots


class BinaryPoseCNN(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--image_size', type=int, default=None,
                            help='Input images will be scaled such that the shorter side is equal to the given value.')

    def __init__(self, args):
        self.angle = 45
        self.z = 0.7

        super(BinaryPoseCNN, self).__init__(args)



        # Model for binary classification
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 2)
        self.criterion = nn.CrossEntropyLoss()

        if self.use_cuda:
            print('Moving CNN to GPU ...')
            self.model.cuda()
            self.criterion.cuda()

        self.print_freq = args.print_freq

    def load_dataset(self, args):
        traindir = ImageNet.FOLDERS['training']
        valdir = ImageNet.FOLDERS['validation']
        testdir = ImageNet.FOLDERS['test']

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

        train_set = ImageNet.PoseGenerator(traindir, max_angle=self.angle, z_plane=self.z, transform1=transform1,
                                           transform2=transform3)
        val_set = ImageNet.PoseGenerator(valdir, max_angle=self.angle, z_plane=self.z, transform1=transform2,
                                         transform2=transform3)
        test_set = ImageNet.PoseGenerator(testdir, max_angle=self.angle, z_plane=self.z, transform1=transform2,
                                          transform2=transform3)

        dataloader_train = DataLoader(train_set, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers)

        dataloader_val = DataLoader(val_set, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)

        dataloader_test = DataLoader(test_set, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)

        return dataloader_train, dataloader_val, dataloader_test

    def train(self, epochs, checkpoint=None):
        start_epoch = 1
        train_loss = []
        validation_loss = []

        optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                    # momentum=args.momentum,
                                    # weight_decay=args.weight_decay)
                                    )

        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        for epoch in range(start_epoch, start_epoch + epochs):
            print('Epoch [{:d}/{:d}]'.format(epoch, start_epoch + epochs - 1))

            # Train for one epoch
            epoch_loss = self.__train_one_epoch(self.model, self.criterion, optimizer, self.trainingset)
            train_loss.append(epoch_loss)

            # TODO: save losses
            checkpoint = {
                'epoch': epoch,
                'train_loss': None,
                'validation_loss': None,
                'model': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            self.save_checkpoint(checkpoint)

            # Validate after each epoch
            epoch_loss, _ = self.test(checkpoint, dataloader=self.validationset)
            validation_loss.append(epoch_loss)

            plots.plot_epoch_loss(train_loss, validation_loss, save=self.save_loss_plot)

    def __train_one_epoch(self, model, criterion, optimizer, dataloader):
        epoch_loss = 0
        num_train_samples = len(dataloader)

        for i, (image, pose) in enumerate(dataloader):

            input = self.to_variable(image)
            target = self.to_variable(pose)

            model.zero_grad()
            output = model(input)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Sample [{:d}/{:d}], Loss: {:.4f}'.format(i + 1, num_train_samples, loss.data[0]))

            epoch_loss += loss.data[0]

        epoch_loss /= num_train_samples
        return epoch_loss

    def test(self, checkpoint, dataloader=None):
        self.model.load_state_dict(checkpoint['model'])

        if not dataloader:
            dataloader = self.testset

        accuracy = 0
        avg_loss = 0
        for i, (image, pose) in enumerate(dataloader):
            input = self.to_variable(image, volatile=True)
            target = self.to_variable(pose, volatile=True)

            self.model.zero_grad()
            output = self.model(input)

            # argmax = predicted class
            _, ind = torch.max(output.data, 1)

            # Correct predictions in the batch
            accuracy += torch.sum(torch.eq(ind, target.data))

            loss = self.criterion(output, target)
            avg_loss += loss.data[0]

        avg_loss /= len(dataloader)
        accuracy /= len(dataloader)
        print('Accuracy on testset: {:.4f}'.format(accuracy))
        return avg_loss, accuracy

