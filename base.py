import os
import torch
from torch.autograd import Variable


CHECKPOINT_FILENAME = 'checkpoint.pth.tar'
CHECKPOINT_BEST_FILENAME = 'checkpoint-best.pth.tar'


class BaseExperiment:

    def __init__(self, folder, args):
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.lr = args.lr
        self.workers = args.workers
        self.batch_size = args.batch_size

        self.out_folder = folder
        self.loss_file = self.make_output_filename('loss.txt')
        self.save_loss_plot = self.make_output_filename('loss.pdf')
        self.checkpoint_file = self.make_output_filename(args.checkpoint)

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

    @property
    def trainingset_size(self):
        return len(self.trainingset) * self.trainingset.batch_size

    @property
    def testset_size(self):
        return len(self.testset) * self.testset.batch_size

    @property
    def validationset_size(self):
        return len(self.validationset) * self.validationset.batch_size

    def train(self):
        pass

    def test(self):
        pass

    def load_dataset(self, args):
        return None, None, None

    def load_checkpoint(self):
        checkpoint = None
        if os.path.isfile(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
        else:
            print('No checkpoint found at {}'.format(self.checkpoint_file))
        return checkpoint

    def save_checkpoint(self, state):
        print('Saving checkpoint ...')
        torch.save(state, self.checkpoint_file)

    def make_checkpoint(self):
        return None

    def restore_from_checkpoint(self, checkpoint):
        pass

    def adjust_learning_rate(self, epoch):
        pass

    def plot_performance(self):
        pass

    def to_variable(self, data, volatile=False):
        var = Variable(data, volatile=volatile)
        if self.use_cuda:
            var = var.cuda()
        return var

    def make_output_filename(self, filename):
        return os.path.join(self.out_folder, filename)

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
