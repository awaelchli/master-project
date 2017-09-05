import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from GTAV import Subsequence, FOLDERS
from flownet.models.FlowNetS import flownets

import matplotlib.pyplot as plt
import matplotlib.colors as cl
plt.switch_backend('agg')
import numpy as np

from torchvision.utils import save_image


def to_variable(data, volatile=False):
    var = Variable(data, volatile=volatile)
    if use_cuda:
        var = var.cuda()
    return var


def forward(input):
    # Input shape: [sequence, channels, h, w]
    n = input.size(0)
    first = input[:n-1]
    second = input[1:]

    # New shape: [sequence - 1, 2 * channels, h, w]
    pairs = torch.cat((first, second), 1)

    assert pairs.size(0) == n - 1

    flows = []
    for pair in pairs:
        flows.append(flownet(pair.unsqueeze(0)))

    return flows


def load_dataset():
    traindir = FOLDERS['standing']['training']

    # Image pre-processing
    transform = transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop((320, 448)),
        transforms.ToTensor(),
    ])

    dataset = Subsequence(
        data_folder=traindir['data'],
        pose_folder=traindir['pose'],
        sequence_length=sequence_length,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=use_cuda,
        shuffle=False,
        num_workers=workers
    )

    return dataloader


def compute_flow():

    dataloader = load_dataset()

    for i, (images, _) in enumerate(dataloader):

        images.squeeze_(0)

        for j, image in enumerate(images):
            name = 'im-{}-{}.png'.format(i, j)
            save_image(image, os.path.join(output_folder, name))

        input = to_variable(images)
        flows = forward(input)

        for j, flow in enumerate(flows):
            flow.squeeze_(0)
            flow = flow.permute(1, 2, 0)

            if use_cuda:
                flow = flow.cpu()

            valid = torch.ones(flow.size(0), flow.size(1), 1)
            flow = torch.cat((flow, valid), 2)

            flow = flow.data.numpy()

            save = os.path.join(output_folder, 'fl-{:04d}-{:02d}.png'.format(i, j))
            visualize_flow(flow, save)


# Source: https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py
def visualize_flow(flow, save):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """

    (h, w) = flow.shape[0:2]
    du = flow[:, :, 0]
    dv = flow[:, :, 1]
    valid = flow[:, :, 2]
    max_flow = max(np.max(du), np.max(dv))
    img = np.zeros((h, w, 3), dtype=np.float64)
    # angle layer
    img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
    # magnitude layer, normalized to 1
    img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
    # phase layer
    img[:, :, 2] = 8 - img[:, :, 1]
    # clip to [0,1]
    small_idx = img[:, :, 0:3] < 0
    large_idx = img[:, :, 0:3] > 1
    img[small_idx] = 0
    img[large_idx] = 1
    # convert to rgb
    img = cl.hsv_to_rgb(img)
    # remove invalid point
    img[:, :, 0] = img[:, :, 0] * valid
    img[:, :, 1] = img[:, :, 1] * valid
    img[:, :, 2] = img[:, :, 2] * valid
    # show
    plt.imshow(img)
    plt.title('Max: {}'.format(max_flow))
    plt.savefig(save)


if __name__ == '__main__':
    sequence_length = 20
    use_cuda = True
    workers = 8
    output_folder = 'out/flowmaps/'

    os.makedirs(output_folder, exist_ok=True)

    flownet = flownets('../data/Pretrained Models/flownets_pytorch.pth')
    flownet.train(False)

    if use_cuda:
        flownet.cuda()

    compute_flow()