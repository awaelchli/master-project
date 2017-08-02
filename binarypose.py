from base import BaseExperiment, AverageMeter
from dataimport import ImageNet
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import plots


class BinaryPoseCNN(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--angle', type=float, default=10,
                            help='The maximum range of rotation of the images.')
        parser.add_argument('--zplane', type=float, default=1,
                            help='Location of the image in front of the camera (along Z-axis).')
        parser.add_argument('--image_size', type=int, default=None,
                            help='Input images will be scaled such that the shorter side is equal to the given value.')

    def __init__(self, folder, args):
        super(BinaryPoseCNN, self).__init__(folder, args)

        # Model for binary classification
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                    # momentum=args.momentum,
                                    # weight_decay=args.weight_decay)
                                    )

        if self.use_cuda:
            print('Moving CNN to GPU ...')
            self.model.cuda()
            self.criterion.cuda()

        self.print_freq = args.print_freq

        self.training_loss = []
        self.validation_loss = []

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

        train_set = ImageNet.PoseGenerator(traindir, max_angle=args.angle, z_plane=args.zplane, transform1=transform1,
                                           transform2=transform3)
        val_set = ImageNet.PoseGenerator(valdir, max_angle=args.angle, z_plane=args.zplane, transform1=transform2,
                                         transform2=transform3)
        test_set = ImageNet.PoseGenerator(testdir, max_angle=args.angle, z_plane=args.zplane, transform1=transform2,
                                          transform2=transform3)

        dataloader_train = DataLoader(train_set, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers)

        dataloader_val = DataLoader(val_set, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)

        dataloader_test = DataLoader(test_set, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)

        return dataloader_train, dataloader_val, dataloader_test

    def train(self, checkpoint=None):
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        training_loss = AverageMeter()
        num_train_samples = len(self.trainingset)

        for i, (image, pose) in enumerate(self.trainingset):

            input = self.to_variable(image)
            target = self.to_variable(pose)

            self.model.zero_grad()
            output = self.model(input)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Sample [{:d}/{:d}], Loss: {:.4f}'.format(i + 1, num_train_samples, loss.data[0]))

            training_loss.update(loss.data[0])

        training_loss = training_loss.average
        self.training_loss.append(training_loss)

        # Validate after each epoch
        validation_loss, _ = self.test(dataloader=self.validationset)
        self.validation_loss.append(validation_loss)

        # TODO: save losses
        checkpoint = {
            'epoch': len(self.training_loss),
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        self.save_checkpoint(checkpoint)

    def test(self, checkpoint=None, dataloader=None):
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])

        if not dataloader:
            dataloader = self.testset

        accuracy = AverageMeter()
        avg_loss = AverageMeter()
        for i, (image, pose) in enumerate(dataloader):
            input = self.to_variable(image, volatile=True)
            target = self.to_variable(pose, volatile=True)

            self.model.zero_grad()
            output = self.model(input)

            # argmax = predicted class
            _, ind = torch.max(output.data, 1)

            # Correct predictions in the batch
            accuracy.update(torch.sum(torch.eq(ind, target.data)))

            loss = self.criterion(output, target)
            avg_loss.update(loss.data[0])

        avg_loss = avg_loss.average
        accuracy = accuracy.average

        print('Accuracy on testset: {:.4f}'.format(accuracy))
        return avg_loss, accuracy

    def plot_performance(self):
        checkpoint = self.load_checkpoint()
        plots.plot_epoch_loss(checkpoint['training_loss'], checkpoint['validation_loss'], save=self.save_loss_plot)

