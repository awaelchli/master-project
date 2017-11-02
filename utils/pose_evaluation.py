from pose_transforms import euler_to_quaternion
import torch
from math import degrees


def relative_quaternion_rotation_error(predictions, targets):
    """ The input quaternions are assumed to be normalized. """
    assert predictions.size() == targets.size()
    assert predictions.size(1) == targets.size(1) == 4

    rel_angles = torch.acos(2 * (predictions * targets).sum(1) ** 2 - 1)
    return [degrees(a) for a in rel_angles.view(-1)]


def relative_euler_rotation_error(predictions, targets):
    q1 = euler_to_quaternion(predictions)
    q2 = euler_to_quaternion(targets)
    return relative_quaternion_rotation_error(q1, q2)


def error_distribution(errors, start=0.0, stop=180.0, step=1.0):
    thresholds = list(torch.arange(start, stop, step))
    n = torch.numel(errors)
    distribution = [torch.sum(errors <= t) / n for t in thresholds]
    return thresholds, distribution


def measure_distance_along_path(positions):
    """
    Measures the distance along a 3D path.
    :param positions: A torch tensor of dimensions m x n x 3, where n is the number of points in the path and m is
    the batch size.
    :return: A torch tensor of dimension m x n which contains the distance along the path up to vertex i for each batch.
    """
    m = positions.size(0)
    n = positions.size(1)
    assert positions.size(2) == 3

    p0 = torch.cat((torch.zeros(m, 1, 3), positions), 1)
    p0 = p0[:, :-1, :]

    norms = torch.norm(positions - p0, p=2, dim=2)
    distances = torch.cumsum(norms, 1)

    assert distances.size(0) == m
    assert distances.size(1) == n

    return distances


def translation_error_along_path(positions, targets):
    """
    :param positions: m x n x 3 torch tensor
    :param targets: m x n x 3 torch tensor
    :return: m x n torch tensor
    """
    assert positions.size(2) == 3

    error = torch.norm(positions - targets, 2, dim=2)
    return error


def rel_rotation_angle_along_path_from_euler(predictions, targets):
    """
    :param predictions: m x n x 3 torch tensor
    :param targets: m x n x 3 torch tensor
    :return: m x n torch tensor
    """
    assert predictions.size(2) == 3

    angles = []
    for p, t in zip(predictions, targets):
        angles.append(torch.Tensor(relative_euler_rotation_error(p, t)))

    rel_angles = torch.stack(angles)
    return rel_angles


def translation_error_per_meters(positions, targets, start=0.0, stop=100.0, step=1.0):
    """
    :param positions: m x n x 3 torch tensor, where m is the batch size and n is the sequence length
    :param targets: same shape as positions
    :param start:
    :param stop:
    :param step:
    :return: Returns two matrices eval_points and err_per_meter. The first is a 1D tensor of increasing evaluation points
            (meters) and the second is also a tensor of the same size containing the average relative rotation error along the path

    """
    m = positions.size(0)
    n = positions.size(1)

    distance = measure_distance_along_path(targets)
    errors = translation_error_along_path(positions, targets)

    eval_points = torch.arange(start, stop, step)
    err_per_meter = torch.zeros(m, len(eval_points))
    valid = torch.zeros(m, len(eval_points))

    # Loop over batch dimension
    for i in range(m):

        # Loop over meter increments
        for j, d in enumerate(eval_points):

            # Loop over path
            for k in range(n):
                if distance[i, k] >= d:
                    err_per_meter[i, j] = errors[i, k]
                    valid[i, j] = 1
                    break

    counts = torch.sum(valid.float(), 0)
    counts[counts == 0] = 1
    err_per_meter = err_per_meter.sum(0) / counts
    assert eval_points.size() == err_per_meter.size()

    return eval_points, err_per_meter


def relative_rotation_error_per_meters_from_euler_pose(predictions, targets, start=0.0, stop=100.0, step=1.0):
    """
    :param positions: m x n x 6 torch tensor, where m is the batch size and n is the sequence length
    :param targets: same shape as positions
    :param start:
    :param stop:
    :param step:
    :return: Returns two matrices eval_points and err_per_meter. The first is a 1D tensor of increasing evaluation points
            (meters) and the second is also a tensor of the same size containing the average relative rotation error along the path

    """
    m = predictions.size(0)
    n = predictions.size(1)

    distance = measure_distance_along_path(targets[:, :, :3])
    errors = rel_rotation_angle_along_path_from_euler(predictions[:, :, 3:], targets[:, :, 3:])

    eval_points = torch.arange(start, stop, step)
    err_per_meter = torch.zeros(m, len(eval_points))
    valid = torch.zeros(m, len(eval_points))

    # Loop over batch dimension
    for i in range(m):

        # Loop over meter increments
        for j, d in enumerate(eval_points):

            # Loop over path
            for k in range(n):
                if distance[i, k] >= d:
                    err_per_meter[i, j] = errors[i, k]
                    valid[i, j] = 1
                    break

    counts = torch.sum(valid.float(), 0)
    counts[counts == 0] = 1
    err_per_meter = err_per_meter.sum(0) / counts
    assert eval_points.size() == err_per_meter.size()

    return eval_points, err_per_meter

