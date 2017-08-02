import os
import shutil
import torch
import plots
from torch.autograd import Variable


CHECKPOINT_FILENAME = 'checkpoint.pth.tar'


class BaseExperiment:

    def __init__(self, folder, args):
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.lr = args.lr
        self.workers = args.workers
        self.batch_size = args.batch_size

        self.out_folder = folder
        self.loss_file = os.path.join(self.out_folder, 'loss.txt')
        self.save_loss_plot = os.path.join(self.out_folder, 'loss.pdf')
        self.save_model_name = os.path.join(self.out_folder, CHECKPOINT_FILENAME)

        self._trainingset, self._validationset, self._testset = self.load_dataset(args)

    @property
    def trainingset(self):
        return self._trainingset

    @property
    def testset(self):
        return self._testset

    @property
    def validationset(self):
        return self._validationset

    def train(self):
        pass

    def test(self):
        pass

    def load_dataset(self, args):
        return None, None, None

    def load_checkpoint(self):
        checkpoint = None
        if os.path.isfile(self.save_model_name):
            checkpoint = torch.load(self.save_model_name)
        else:
            print('No checkpoint found at {}'.format(self.save_model_name))
        return checkpoint

    def save_checkpoint(self, state):
        print('Saving checkpoint ...')
        torch.save(state, self.save_model_name)

    def make_checkpoint(self):
        return None

    def restore_from_checkpoint(self, checkpoint):
        pass

    def plot_performance(self):
        pass

    def to_variable(self, data, volatile=False):
        var = Variable(data, volatile=volatile)
        if self.use_cuda:
            var = var.cuda()
        return var

    @staticmethod
    def submit_arguments(parser):
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value):
        self.sum += value
        self.count += 1
