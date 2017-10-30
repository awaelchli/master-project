import torch


def measure_distance_along_path(positions):
    """
    Measures the distance along a 3D path.
    :param positions: A torch tensor of dimensions n x 3, where n is the number of points in the path
    :return: A torch tensor of dimension n which contains the distance along the path up to vertex i.
    """

    p0 = torch.cat((torch.zeros(1, 3), positions), 0)
    p0 = p0[:-1, :]

    norms = torch.norm(positions - p0, p=2, dim=1)
    distances = torch.cumsum(norms, 0)

    return distances

