import torch
from torch.utils.data import Dataset


class Random(Dataset):
    """Dummy Dataset"""

    def __init__(self, size=10, sequence_length=10, height=50, width=50, channels=3):
        self.size = size
        self.height = height
        self.width = width
        self.sequence_length = sequence_length
        self.channels = channels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        images = torch.rand(self.sequence_length, self.channels, self.height, self.width)
        poses = torch.rand(self.sequence_length, 6)

        sequence = (images, poses)
        return sequence
