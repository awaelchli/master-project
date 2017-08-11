import torch
from math import radians


def get_quaternion_pose(pose):
    # Output vector: [x, y, z, ax, ay, az]
    # phi = norm(a)

    # ax = pose[:, 3].contiguous().view(-1, 1)
    # ay = pose[:, 4].contiguous().view(-1, 1)
    # az = torch.sqrt(torch.abs(1 - (ax ** 2) - (ay ** 2)))

    # Norm of axis = rotation angle + 1
    axis = pose[:, 3:6].contiguous().view(-1, 3)
    norm = torch.sqrt((axis ** 2).sum(1))
    norm_repl = norm.expand(norm.size(0), 3)
    axis = axis / norm_repl

    phi = norm - 1
    phi_repl = norm_repl - 1

    # phi = pose[:, 5].contiguous().view(-1, 1)


    # axis = torch.cat((ax, ay, az), 1)

    # Elements of quaternion
    q = torch.cat((torch.cos(phi / 2), torch.sin(phi_repl / 2) * axis), 1)
    return q


def loss_function(output, target):
    # Dimensions: [sequence_length, 6]
    sequence_length = output.size(1)
    # print(output)
    # print(target)
    t1 = output[:, 0:3]
    t2 = target[:, 0:3]
    q1 = get_quaternion_pose(output)
    q2 = get_quaternion_pose(target)

    # print(q1)
    # print(q2)
    # Loss for rotation, dot product between quaternions
    loss1 = 1 - torch.abs((q1 * q2).sum(1))
    loss1 = loss1.sum() / sequence_length

    # Loss for translation
    eps = 0.001

    # self.criterion(t1, t2)
    t_diff = torch.pow(t1 - t2, 2).sum(1)
    loss2 = torch.log(eps + t_diff)
    loss2 = loss2.sum() / sequence_length

    return loss1, loss2


def make_pose(t, axis, theta):
    t = torch.Tensor([t])
    axis = torch.Tensor([axis]) * (1 + radians(theta))
    return torch.cat((t, axis), 1)


pose1 = make_pose([0, 0, 0], [1, 0, 0], 0)
for x in range(0, 20):
    pose2 = make_pose([x, 0, 0], [1, 0, 0], 0)
    print(loss_function(pose1, pose2))

