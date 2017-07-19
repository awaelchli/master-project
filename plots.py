import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_loss_from_file(file, save=None):
    losses = read_loss_from_file(file)
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    if save:
        plt.savefig(save, bbox_inches='tight')


def read_loss_from_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    losses = [float(l) for l in lines]
    return losses


plot_loss_from_file('out/loss.txt', save='out/loss.pdf')
