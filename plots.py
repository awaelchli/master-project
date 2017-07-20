import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg') # For machines without display (e.g. cluster)


def plot_loss_from_file(file, save=None):
    losses = read_loss_from_file(file)
    epochs = list(range(1, len(losses) + 1))
    plt.clf()
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


def plot_epoch_loss(train_loss, validation_loss=None, save='loss.pdf'):
    epochs = list(range(1, len(train_loss) + 1))
    plt.clf()
    plt.plot(epochs, train_loss, 'b', label='Training')
    if validation_loss is not None:
        plt.plot(epochs, validation_loss, 'g', label='Validation')

    plt.legend()

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    if save:
        plt.savefig(save, bbox_inches='tight')


# def save_train_data(file, train_loss, validation_loss):
#     epochs = train_loss.numel()
#     epoch_index = torch.range(1, epochs).unsqueeze(1)
#
#     data = torch.cat((epoch_index, train_loss.view(epochs, 1), validation_loss.view(epochs, 1)), 1)
#
#     pd.DataFrame(np.random.randn(6, 4), index=None, columns=list('ABCD'))


#plot_loss_from_file('out/loss.txt', save='out/loss.pdf')
#plot_train_data(torch.Tensor([0.1, 0.2, 0.3]), torch.Tensor([0.3, 0.5, 0.8]), save='t.pdf')
#plot_epoch_loss([0.1, 0.2, 0.3], [0.3, 0.5, 0.8], save='t.pdf')