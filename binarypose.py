from base import BaseExperiment, AverageMeter
from dataimport.ImageNet import PoseGenerator, FOLDERS
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import plots
import time
import random


class BinaryPoseCNN(BaseExperiment):

    @staticmethod
    def submit_arguments(parser):
        parser.add_argument('--angle', type=float, default=10,
                            help='The maximum range of rotation of the images.')
        parser.add_argument('--zplane', type=float, default=1,
                            help='Location of the image in front of the camera (along Z-axis).')
        parser.add_argument('--image_size', type=int, default=None,
                            help='Input images will be scaled such that the shorter side is equal to the given value.')
        parser.add_argument('--max_size', type=int, default=None,
                            help='Clips the training dataset at the given size.')

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
        traindir = FOLDERS['training']
        valdir = FOLDERS['validation']
        testdir = FOLDERS['test']

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

        train_set = PoseGenerator(traindir, max_angle=args.angle, z_plane=args.zplane, transform1=transform1,
                                  transform2=transform3, max_size=args.max_size)
        val_set = PoseGenerator(valdir, max_angle=args.angle, z_plane=args.zplane, transform1=transform2,
                                transform2=transform3)
        test_set = PoseGenerator(testdir, max_angle=args.angle, z_plane=args.zplane, transform1=transform2,
                                 transform2=transform3)

        dataloader_train = DataLoader(train_set, batch_size=args.batch_size, drop_last=True,
                                      shuffle=True, num_workers=args.workers)

        dataloader_val = DataLoader(val_set, batch_size=args.batch_size, drop_last=True,
                                    shuffle=False, num_workers=args.workers)

        dataloader_test = DataLoader(test_set, batch_size=args.batch_size, drop_last=True,
                                     shuffle=False, num_workers=args.workers)

        return dataloader_train, dataloader_val, dataloader_test

    def train(self):
        training_loss = AverageMeter()
        num_batches = len(self.trainingset)

        for i, (image, pose) in enumerate(self.trainingset):

            self.randomly_save_image(image, 0.05)

            input = self.to_variable(image)
            target = self.to_variable(pose)

            self.model.zero_grad()
            output = self.model(input)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Print log info
            if (i + 1) % self.print_freq == 0:
                print('Batch [{:d}/{:d}], Loss: {:.4f}'.format(i + 1, num_batches, loss.data[0]))

            training_loss.update(loss.data[0])

        training_loss = training_loss.average
        self.training_loss.append(training_loss)

        # Validate after each epoch
        validation_loss, _ = self.test(dataloader=self.validationset)
        self.validation_loss.append(validation_loss)

    def test(self, dataloader=None):
        if not dataloader:
            dataloader = self.testset

        num_samples = len(dataloader) * dataloader.batch_size
        accuracy = 0
        avg_loss = AverageMeter()
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
            avg_loss.update(loss.data[0])

        accuracy /= num_samples
        avg_loss = avg_loss.average

        print('Accuracy: {:.4f}'.format(accuracy))
        return avg_loss, accuracy

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

    def randomly_save_image(self, batch, prob=0.5):
        if random.uniform(0, 1) < prob:
            # Randomly select image from batch
            i = random.randrange(0, batch.size(0))
            image = batch[i].squeeze(0)
            tf = transforms.ToPILImage()
            image = tf(image)
            fname = time.strftime('%Y%m%d-%H%M%S.png')
            image.save(os.path.join(self.out_folder, fname))

