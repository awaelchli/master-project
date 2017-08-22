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


def plot_epoch_accuracy(accuracy, save='accuracy.pdf'):
    epochs = list(range(1, len(accuracy) + 1))
    plt.clf()
    plt.plot(epochs, accuracy, 'b')

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_sample_loss(train_loss, save='sample_loss.pdf'):
    samples = list(range(1, len(train_loss) + 1))
    plt.clf()
    plt.plot(samples, train_loss, 'b')

    plt.ylabel('Loss')
    plt.xlabel('Samples')

    if save:
        plt.savefig(save, bbox_inches='tight')
