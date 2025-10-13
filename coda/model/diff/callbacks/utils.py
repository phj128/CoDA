import torch
import torch.nn.functional as F


def compute_mpjpe(pred, target, mask=None):
    pred = pred.reshape(-1, 3)
    target = target.reshape(-1, 3)
    if mask is not None:
        mask = mask.reshape(-1, 3)
        mpjpe = ((pred - target) * mask).norm(dim=-1)
        N = mask.sum() // 3 + 1e-6
        return mpjpe.sum() / N
    else:
        mpjpe = (pred - target).norm(dim=-1)
        return mpjpe.mean()


def resample_motion_fps(motion, target_length):
    N, J, C = motion.shape  # N: 帧数，J: 关节点数，C: 坐标数（通常为3）

    motion = motion.permute(1, 2, 0)

    upsampled_motion = F.interpolate(motion, size=target_length, mode="linear", align_corners=True)

    upsampled_motion = upsampled_motion.permute(2, 0, 1)
    return upsampled_motion


def compute_distance_error(pred, target, mask=None):
    pred = pred.reshape(-1, 1)
    target = target.reshape(-1, 1)
    if mask is not None:
        mask = mask.reshape(-1, 1)
        mpjpe = ((pred - target) * mask).norm(dim=-1)
        N = mask.sum() + 1e-6
        return mpjpe.sum() / N
    else:
        mpjpe = (pred - target).norm(dim=-1)
        return mpjpe.mean()


def calculate_foot_sliding(motion):
    # motion: (B, L, J, 3)
    foot_ind = [10, 11]
    foot_global_pos = motion[..., foot_ind, :]
    foot_global_h = foot_global_pos[..., :-1, :, 1]  # (B, L, j)
    foot_global_vel = foot_global_pos[..., 1:, :, [0, 2]] - foot_global_pos[..., :-1, :, [0, 2]]
    # assume 30 fps
    foot_global_vel = 30 * torch.norm(foot_global_vel, dim=-1)  # (B, L-1, j)
    left_foot_mask = (foot_global_h[..., 0] < foot_global_h[..., 1]).float()
    foot_sliding = foot_global_vel[..., 0] * left_foot_mask + foot_global_vel[..., 1] * (1 - left_foot_mask)
    foot_sliding = foot_sliding.mean(dim=-1)  # (B, )
    return foot_sliding
