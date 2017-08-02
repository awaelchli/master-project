import argparse
import torch
import os
import shutil


class BaseExperiment:

    def __init__(self, args):
        self.use_cuda = args.cuda
        self.epochs = args.epochs
        self.lr = args.learning_rate
        self.workers = args.workers
        self.batch_size = args.batch_size

        self.out_folder = os.path.join(OUT_BASE_FOLDER, args.experiment)
        self.loss_file = os.path.join(self.out_folder, 'loss.txt')
        self.save_loss_plot = os.path.join(self.out_folder, 'loss.pdf')
        self.save_model_name = os.path.join(self.out_folder, 'checkpoint.pth.tar')

        self.train_loader, self.test_loader, self.val_loader = self.load_dataset()

    def train(self):
        pass

    def test(self):
        pass

    def load_dataset(self):
        pass

    def load_checkpoint(self):
        if os.path.isfile(args.resume):
            print('Loading checkpoint ...')
            checkpoint = torch.load(self.save_model_name)
            #args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            #model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('No checkpoint found at {}'.format(self.save_model_name))
        return checkpoint

    def save_checkpoint(self, filename):
        #torch.save(state, filename)
        pass

    def setup_environment(self):
        # Wipe all existing data
        if os.path.isdir(self.out_folder):
            shutil.rmtree(self.out_folder)
        os.makedirs(self.out_folder)

    @staticmethod
    def submit_arguments(parse):
        parse.add_argument('--train', action='store_true')
        parse.add_argument('--test', action='store_true')
        parse.add_argument('--resume', action='store_true')

        parse.add_argument('--cuda', action='store_true')

        parse.add_argument('--print_freq', type=int, default=100,
                           help='Frequency of printed information during training')

        parse.add_argument('--name', type=str, default='unnamed',
                           help='Name of the experiment. Output files will be stored in this folder.')

        parse.add_argument('--batch_size', type=int, default=1)
        parse.add_argument('--workers', type=int, default=2)

        # Training parameters
        parse.add_argument('--epochs', type=int, default=1)
        parse.add_argument('--learning_rate', type=float, default=0.001)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)

    BaseExperiment.submit_arguments(parser)

    args = parser.parse_args()
    print(args)

    OUT_BASE_FOLDER = 'out'
    os.makedirs(OUT_BASE_FOLDER, exist_ok=True)
    USE_CUDA = torch.cuda.is_available() and args.cuda

    ARCHS = {
        'poselstm': BaseExperiment
    }

    e = BaseExperiment(args)
    print(e)
    e.setup_environment()

    if torch.cuda.is_available():
        print('CUDA is available on this machine.')
    else:
        print('CUDA is not available on this machine.')


