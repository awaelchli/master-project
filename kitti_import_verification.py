import os
from dataimport import KITTI
from torchvision import transforms
from torch.utils.data import DataLoader


def test_kitti_import():
    sequence_length = 20

    os.makedirs('out', exist_ok=True)

    transform = transforms.Compose([
        transforms.Scale(100),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ])

    kitti_train = KITTI.Subsequence(sequence_length, transform, grayscale=True,
                                    sequence_numbers=[2])

    image_size = kitti_train[0][0].size()[1:4]
    print('Image size:', image_size)

    dataloader_train = DataLoader(kitti_train, batch_size=1,
                                  shuffle=True, num_workers=1)

    i, sample = next(enumerate(dataloader_train))
    print(i, sample)
    for i, image in enumerate(sample[0].squeeze(0)):
        tf = transforms.ToPILImage()
        img = tf(image).convert('RGB')
        img.save('out/test_img_{:d}.png'.format(i))


if __name__ == '__main__':
    test_kitti_import()