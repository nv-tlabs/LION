from torch.autograd import Function

from third_party.pvcnn.functional.backend import _backend

__all__ = ['ball_query']


def ball_query(centers_coords, points_coords, radius, num_neighbors):
    """
        :param centers_coords: coordinates of centers, FloatTensor[B, 3, M]
        :param points_coords: coordinates of points, FloatTensor[B, 3, N]
        :param radius: float, radius of ball query
        :param num_neighbors: int, maximum number of neighbors
        :return:
            neighbor_indices: indices of neighbors, IntTensor[B, M, U]
        """
    centers_coords = centers_coords[:,:3].contiguous()
    points_coords = points_coords[:,:3].contiguous()
    return _backend.ball_query(centers_coords, points_coords, radius,
                               num_neighbors)
