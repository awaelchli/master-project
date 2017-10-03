import torch.nn.functional


def corr(features1, features2):
    assert features1.size() == features2.size()
    num_batches = features1.size(0)
    num_planes = features1.size(1)
    h, w = features1.size(2), features1.size(3)

    output = torch.zeros(num_batches, h * w, h, w)

    l = 0
    for i in range(h):
        for j in range(w):
            replicas = features1[:, :, i, j].contiguous().view(num_batches, num_planes, 1, 1).expand_as(features1)
            dot_prod = torch.sum(replicas * features2, 1, keepdim=True)
            output[:, l, :, :] = dot_prod
            l += 1

    # Output shape: [batches, h * w, h, w]
    return output



# ## Test
#
# # First pair
# i11 = torch.Tensor([[0, 1], [2, 3]]).unsqueeze(0)
# i12 = torch.Tensor([[4, 1], [3, 3]]).unsqueeze(0)
# i21 = torch.Tensor([[3, 2], [2, 2]]).unsqueeze(0)
# i22 = torch.Tensor([[2, 1], [4, 4]]).unsqueeze(0)
# i1 = torch.cat((i11, i12), 0).unsqueeze(0)
# i2 = torch.cat((i21, i22), 0).unsqueeze(0)
#
# # Second pair
# i31 = torch.Tensor([[1, 3], [3, 2]]).unsqueeze(0)
# i32 = torch.Tensor([[1, 1], [3, 4]]).unsqueeze(0)
# i41 = torch.Tensor([[1, 4], [0, 0]]).unsqueeze(0)
# i42 = torch.Tensor([[1, 1], [0, 4]]).unsqueeze(0)
# i3 = torch.cat((i31, i32), 0).unsqueeze(0)
# i4 = torch.cat((i41, i42), 0).unsqueeze(0)
#
# a = torch.cat((i1, i3), 0)
# b = torch.cat((i2, i4), 0)
#
# out = corr(a, b)
#
# print(out.size())
#
# print(out[0, 0, :, :])
# print(out[0, 1, :, :])
#
# print(out[1, 0, :, :])
# print(out[1, 1, :, :])