from torch.utils.data.sampler import SequentialSampler


class StridedSampler(SequentialSampler):
    """ Samples elements sequentially, always in the same order, but with a given stride (skipping elements).

    Arguments:
        data_source (Dataset): dataset to sample from
        stride (int): stride to apply when sampling
        max_size (int): clips the dataset such that there are at most max_size samples
    """

    def __init__(self, data_source, stride=1, max_size=None):
        super(StridedSampler, self).__init__(data_source)
        self.stride = stride
        self.max_size = max_size
        self.indices = self.get_indices()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def get_indices(self):
        indices = range(0, len(self.data_source), self.stride)
        if self.max_size:
            indices = indices[:self.max_size]
        return indices