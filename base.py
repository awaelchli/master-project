import os
import shutil
import torch
from torch.autograd import Variable

OUT_ROOT_FOLDER = 'out'


class BaseExperiment:

    def __init__(self, args):
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.lr = args.lr
        self.workers = args.workers
        self.batch_size = args.batch_size

        self.out_folder = os.path.join(OUT_ROOT_FOLDER, args.name)
        self.loss_file = os.path.join(self.out_folder, 'loss.txt')
        self.save_loss_plot = os.path.join(self.out_folder, 'loss.pdf')
        self.save_model_name = os.path.join(self.out_folder, 'checkpoint.pth.tar')

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

    def train(self, epochs, checkpoint=None):
        pass

    def test(self, checkpoint):
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

    def setup_environment(self):
        # Wipe all existing data
        if os.path.isdir(self.out_folder):
            shutil.rmtree(self.out_folder)
        os.makedirs(self.out_folder)

    def to_variable(self, data, volatile=False):
        var = Variable(data, volatile=volatile)
        if self.use_cuda:
            var = var.cuda()
        return var

    @staticmethod
    def submit_arguments(parser):
        pass