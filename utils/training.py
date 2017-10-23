class AverageMeter(object):
    """ Computes and stores the average and current value """
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
