import os
import unittest
from dataimport import KITTI
from torchvision import transforms
from torch.utils.data import DataLoader


class TestUtils(unittest.TestCase):

    def test_kitti_import(self):
        sequence_length = 20

        os.makedirs('out', exist_ok=True)

        transform = transforms.Compose([
            transforms.Scale(100),
            transforms.ToTensor()
        ])

        kitti_train = KITTI.Subsequence(sequence_length, transform, grayscale=True,
                                        sequence_numbers=[2])

        dataloader_train = DataLoader(kitti_train, batch_size=1,
                                      shuffle=True, num_workers=1)

        i, sample = next(enumerate(dataloader_train))
        for i, image in enumerate(sample[0].squeeze(0)):
            self.assertTrue(image.max() <= 1)
            self.assertTrue(image.min() >= 0)

            tf = transforms.ToPILImage()
            img = tf(image).convert('RGB')
            img.save('out/test_img_{:d}.png'.format(i))


if __name__ == '__main__':
    unittest.main()