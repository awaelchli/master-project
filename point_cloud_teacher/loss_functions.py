import torch


class InnerL2:
    """ Inner product of quaterions + L2 loss for translation """

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, output, target):
        # Dimensions: [sequence_length, 7]
        sequence_length = output.size(0)

        t1 = output[:, :3]
        t2 = target[:, :3]
        q1 = output[:, 3:]
        q2 = target[:, 3:]

        assert q1.size(1) == q2.size(1) == 4

        # Loss for rotation
        loss1 = 1 - (q1 * q2).sum(1) ** 2
        loss1 = loss1.sum() / sequence_length

        # Loss for translation
        loss2 = torch.norm(t1 - t2, 2, dim=1)
        loss2 = loss2.sum() / sequence_length

        return loss1 + self.beta * loss2, loss1, loss2


class InnerL1:
    """ Inner product of quaterions + L1 loss for translation """

    def __init__(self, beta):
        self.beta = beta
        self.L1 = torch.nn.L1Loss(size_average=False)

    def __call__(self, output, target):
        # Dimensions: [sequence_length, 7]
        sequence_length = output.size(0)

        t1 = output[:, :3]
        t2 = target[:, :3]
        q1 = output[:, 3:]
        q2 = target[:, 3:]

        assert q1.size(1) == q2.size(1) == 4

        # Loss for rotation
        loss1 = 1 - (q1 * q2).sum(1) ** 2
        loss1 = loss1.sum() / sequence_length

        # Loss for translation
        loss2 = self.L1(t1, t2)
        loss2 /= sequence_length

        return loss1 + self.beta * loss2, loss1, loss2


class InnerLogL2:
    """ Inner product of quaterions + log of L2 loss for translation """

    def __init__(self, beta):
        self.beta = beta
        self.eps = 0.001

    def __call__(self, output, target):
        # Dimensions: [sequence_length, 7]
        sequence_length = output.size(0)

        t1 = output[:, :3]
        t2 = target[:, :3]
        q1 = output[:, 3:]
        q2 = target[:, 3:]

        assert q1.size(1) == q2.size(1) == 4

        # Loss for rotation: dot product between quaternions
        loss1 = 1 - (q1 * q2).sum(1) ** 2
        loss1 = loss1.sum() / sequence_length

        # Loss for translation
        loss2 = torch.norm(t1 - t2, 2, dim=1)
        loss2 = torch.log(self.eps + loss2)
        loss2 = loss2.sum() / sequence_length

        return loss1 + self.beta * loss2, loss1, loss2


class L1L1:
    """ Balanced L1 loss for rotation and translation """

    def __init__(self, beta):
        self.beta = beta
        self.L1 = torch.nn.L1Loss(size_average=False)

    def __call__(self, output, target):
        # Dimensions: [sequence_length, 7]
        sequence_length = output.size(0)

        t1 = output[:, :3]
        t2 = target[:, :3]
        q1 = output[:, 3:]
        q2 = target[:, 3:]

        assert q1.size(1) == q2.size(1) == 4

        # Loss for rotation
        loss1 = self.L1(q1, q2)
        loss1 /= sequence_length

        # Loss for translation
        loss2 = self.L1(t1, t2)
        loss2 /= sequence_length

        return loss1 + self.beta * loss2, loss1, loss2


class TranslationL2:
    """ L2 loss on translation only. Rotation component is ignored. """

    def __call__(self, output, target):
        # Dimensions: [sequence_length, 7]
        sequence_length = output.size(0)

        t1 = output[:, :3]
        t2 = target[:, :3]
        q1 = output[:, 3:]
        q2 = target[:, 3:]

        assert q1.size(1) == q2.size(1) == 4

        # Loss for translation
        loss2 = torch.norm(t1 - t2, 2, dim=1)
        loss2 = loss2.sum() / sequence_length

        return loss2, loss2, loss2