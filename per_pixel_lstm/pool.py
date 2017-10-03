import torch
import torch.nn as nn
import torch.nn.functional as fn


x1 = torch.Tensor(
    [[1, 2, 4, 5],
     [4, 1, 2, 1],
     [0, 5, 0, 0],
     [1, 1, 1, 1]]
)

x2 = torch.Tensor(
    [[2, 2, 2, 2],
     [4, 0, 1, 4],
     [5, 2, 2, 1],
     [0, 1, 4, 1],
    ]
)

# Batch size: 1
# Channels: 2
input = torch.stack((x1, x2), 0).unsqueeze(0)

pool = nn.MaxPool2d(2, 2, return_indices=True)

o, i = pool(input)

print('output, indices')
print(o)
print(i)

print('linear')
print(o.view(-1))
print(i.view(-1))

print('index select')
tmp = torch.index_select(input.view(-1), 0, i.data.view(-1))
print(tmp)

print(tmp.view(1, 2, 2, 2))

