import os

import torch
from torch.autograd import Variable

CHECKPOINT_FILENAME = 'checkpoint.pth.tar'
CHECKPOINT_BEST_FILENAME = 'checkpoint-best.pth.tar'
INFO_FILENAME = 'info.txt'


class BaseExperiment:

    def __init__(self, in_folder, out_folder, args):
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.lr = args.lr
        self.workers = args.workers
        self.batch_size = args.batch_size

        self.in_folder = in_folder
        self.out_folder = out_folder
        self.save_loss_plot = self.make_output_filename('loss.pdf')
        self.checkpoint_file = self.make_input_filename(args.checkpoint)
        self.train_logger = Logger(self.make_output_filename('training.log'))
        self.test_logger = Logger(self.make_output_filename('test.log'))

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

    def validate(self):
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

    def num_parameters(self):
        pass

    def to_variable(self, data, volatile=False):
        var = Variable(data, volatile=volatile)
        if self.use_cuda:
            var = var.cuda()
        return var

    def make_output_filename(self, filename):
        return os.path.join(self.out_folder, filename)

    def make_input_filename(self, filename):
        return os.path.join(self.in_folder, filename)

    def print_info(self, info, clear=False):
        mode = 'w' if clear else 'a'
        with open(self.make_output_filename(INFO_FILENAME), mode) as f:
            f.write(str(info))
            f.write('\n')

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


class Logger:

    def __init__(self, file):
        self.file = file
        self.column_name = []
        self.format_spec = []
        self.sep = ', '
        self.header_printed = False

    def column(self, name='', format='{}'):
        self.column_name.append(name)
        self.format_spec.append(format)

    def print_header(self):
        if not self.header_printed:
            self.header_printed = True
            with open(self.file, 'w') as f:
                f.write(self.sep.join(self.column_name))
                f.write('\n')

    def log(self, *data):
        assert len(data) == len(self.column_name)
        self.print_header()

        line = [t.format(data) for t, data in zip(self.format_spec, data)]
        line = self.sep.join(line)
        with open(self.file, 'a') as f:
            f.write(line)
            f.write('\n')

    def print(self, line='\n'):
        with open(self.file, 'a') as f:
            f.write(line)
            f.write('\n')

    def clear(self):
        open(self.file, 'w').close()
