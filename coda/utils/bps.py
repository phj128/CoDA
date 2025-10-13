import torch
import numpy as np


def bps_gen_ball_inside(n_bps=1000, random_seed=100, scale=1.0):
    np.random.seed(random_seed)
    x = np.random.normal(size=[n_bps, 3])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms  # points on the unit ball surface
    x_unit = x_unit * scale
    r = np.random.uniform(size=[n_bps, 1])
    u = np.power(r, 1.0 / 3)
    basis_set = 1 * x_unit * u  # basic set coordinates, [n_bps, 3]
    return torch.tensor(basis_set, dtype=torch.float32)


def calculate_bps(basis_point, target_verts, return_xyz=False):
    # F, N, 3 and F, M, 3
    dist = torch.cdist(basis_point, target_verts)  # (F, N, M)
    min_dist, min_ind = torch.min(dist, dim=-1)  # (F, M)
    if return_xyz:
        delta_pos = []
        for i in range(min_dist.shape[0]):
            b_i = 0 if basis_point.shape[0] == 1 else i
            delta_pos_ = target_verts[i, min_ind[i]] - basis_point[b_i]
            delta_pos.append(delta_pos_)
        delta_pos = torch.stack(delta_pos, dim=0)
        return delta_pos, min_ind
    else:
        return min_dist, min_ind


def get_pc_center(pc):
    # N, 3
    min_coords = pc.min(dim=-2)[0]
    max_coords = pc.max(dim=-2)[0]
    center = (min_coords + max_coords) / 2
    return center


def get_pc_scale(pc, multiply=1.0):
    # N, 3
    scale = pc.max(dim=0)[0] - pc.min(dim=0)[0]
    scale = torch.max(scale)
    return scale * multiply
