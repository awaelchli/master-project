import matplotlib.pyplot as plt


def plot_xyz_error(predictions, targets, output_file):
    positions1 = predictions[:, :3]
    positions2 = targets[:, :3]

    marker_freq = int(0.1 * len(predictions))
    ms = 5

    x1, y1, z1 = positions1[:, 0], positions1[:, 1], positions1[:, 2]
    x2, y2, z2 = positions2[:, 0], positions2[:, 1], positions2[:, 2]

    plt.clf()

    plt.subplot(311)
    plt.plot(x2, 'ro-', label='Ground Truth', markevery=marker_freq, markersize=ms)
    plt.plot(x1, 'bo-', label='Prediction', markevery=marker_freq, markersize=ms)

    #plt.legend()
    plt.ylabel('x')
    plt.xlabel('Time')

    plt.subplot(312)
    plt.plot(y2, 'ro-', label='Ground Truth', markevery=marker_freq, markersize=ms)
    plt.plot(y1, 'bo-', label='Prediction', markevery=marker_freq, markersize=ms)

    #plt.legend()
    plt.ylabel('y')
    plt.xlabel('Time')

    plt.subplot(313)
    plt.plot(z2, 'ro-', label='Ground Truth', markevery=marker_freq, markersize=ms)
    plt.plot(z1, 'bo-', label='Prediction', markevery=marker_freq, markersize=ms)

    plt.legend(loc=3)
    plt.ylabel('z')
    plt.xlabel('Time')

    plt.savefig(output_file, bbox_inches='tight')


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


def plot_sequence_error(errors, save='sequence_error.pdf'):
    sequence = list(range(1, len(errors) + 1))
    plt.clf()
    plt.plot(sequence, errors, 'b')

    plt.ylabel('Average Absolute Error in Degrees')
    plt.xlabel('Sequence')

    plt.xlim(0, len(sequence) + 1)

    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_error_distribution(t, cdf, save='error_distribution.pdf'):
    plt.clf()
    plt.plot(t, cdf, 'b')

    plt.ylabel('Distribution')
    plt.xlabel('Average Absolute Error (degrees)')

    plt.xlim(min(t), max(t))

    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_translation_error_per_meter(x, errors, save=None):
    plt.clf()
    plt.plot(x, errors, 'b')

    plt.ylabel('Average Translation Error [m]')
    plt.xlabel('Distance [m]')

    plt.xlim(min(x), max(x))

    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_rotation_error_per_meter(x, errors, save=None):
    plt.clf()
    plt.plot(x, errors, 'b')

    plt.ylabel('Average Relative Rotation Error [deg]')
    plt.xlabel('Distance [m]')

    plt.xlim(min(x), max(x))

    if save:
        plt.savefig(save, bbox_inches='tight')