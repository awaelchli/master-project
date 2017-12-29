import torch
from math import acos


for i in range(10):

    q1 = torch.rand(4)
    q2 = torch.rand(4)

    q1 = q1 / torch.norm(q1)
    q2 = q2 / torch.norm(q2)

    theta1 = 2 * acos(abs((q1 * q2).sum()))
    theta2 = acos(2 * (q1 * q2).sum() ** 2 - 1)

    print(theta1)
    print(theta2)