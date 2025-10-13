import torch
import torch.nn.functional as F
import math
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines
import coda.utils.matrix as matrix

# import chamfer_distance as chd
from einops import einsum


class WholeBodyPartCondKeyLocationsLoss:
    def __init__(
        self,
        target=None,
        target_mask=None,
        motion_length=None,
        body_inv_transform=None,
        lefthand_inv_transform=None,
        righthand_inv_transform=None,
        use_mse_loss=False,
        only_body_steps=None,
        no_body_steps=None,
        no_pene_steps=None,
        body_w=1.0,
        hand_w=1.0,
        detach_hand_w=0.0,
        delta_w=1.0,
        sdf_w=1.0,
        root_height_reg_w=1.0,
        foot_sliding_reg_w=1.0,
        foot_floating_reg_w=1.0,
        head_w=0.2,
        is_vis=False,
        is_moving_obj=False,
        moving_obj_kwargs=None,
        save_res=False,
    ):
        self.target = target
        self.target_mask = target_mask
        self.motion_length = motion_length
        self.body_inv_transform = body_inv_transform
        self.lefthand_inv_transform = lefthand_inv_transform
        self.righthand_inv_transform = righthand_inv_transform
        self.use_mse_loss = use_mse_loss
        self.only_body_steps = only_body_steps
        self.no_body_steps = no_body_steps
        self.no_pene_steps = no_pene_steps
        self.body_w = body_w
        self.hand_w = hand_w
        self.detach_hand_w = detach_hand_w
        self.delta_w = delta_w
        self.sdf_w = sdf_w
        self.root_height_reg_w = root_height_reg_w
        self.foot_sliding_reg_w = foot_sliding_reg_w
        self.foot_floating_reg_w = foot_floating_reg_w
        self.head_w = head_w
        self.is_vis = is_vis
        self.is_moving_obj = is_moving_obj
        self.moving_obj_kwargs = moving_obj_kwargs
        # save_res = True
        self.save_res = save_res
        # NOTE: DNO says optimizing to the whole trajectory is not good.

    def __call__(
        self,
        body_xstart_in,
        lefthand_xstart_in,
        righthand_xstart_in,
        left_hand_ind,
        right_hand_ind,
        obj_verts,
        obj_normals,
        smplx_model,
        inputs=None,
        iter=None,
        vis_i=None,
    ):
        """
        Args:
            xstart_in: [bs, L, C]
            target: [bs, L, J, 3]
            motion_length: [bs]
            target_mask: [bs, L, J, 3]
        """
        # TODO: try relative joint distance loss
        target = self.target.clone()
        target_mask = self.target_mask.clone()
        detach_target_mask = self.target_mask.clone()
        delta_target_mask = self.target_mask.clone()
        motion_length = self.motion_length

        # motion_mask shape [bs, 120, 22, 3]
        if motion_length is None:
            motion_mask = torch.ones_like(target_mask)
        else:
            # the mask is only for the first motion_length frames
            motion_mask = torch.zeros_like(target_mask)
            for i, m_len in enumerate(motion_length):
                motion_mask[i, :m_len, :, :] = 1.0

        with torch.enable_grad():
            idle_body_global_out = self.body_inv_transform(torch.zeros_like(body_xstart_in), inputs=inputs)
            idle_global_pos = idle_body_global_out["global_pos"]  # (B, L, J, 3)
            idle_root_height = idle_global_pos[:, 0, 0, 1] - idle_global_pos[:, 0, :, 1].min(dim=-1)[0]  # (B, )

            body_global_out = self.body_inv_transform(body_xstart_in, inputs=inputs)
            body_global_mat = body_global_out["global_mat"]
            body_global_pos = body_global_out["global_pos"]

            left_hand_inputs = {}
            for k, v in inputs.items():
                if k in [
                    "left_base_rotmat",
                    "left_base_pos",
                    "left_handpose",
                    "left_hand_localpos",
                    "left_handtip_localpos",
                ]:
                    left_hand_inputs[k.replace("left_", "")] = v
                    continue
                left_hand_inputs[k] = v
            left_hand_inputs["base_rotmat"] = matrix.get_rotation(body_global_mat[..., 20, :, :])
            left_hand_inputs["base_pos"] = body_global_pos[..., 20, :]
            lefthand_global_out = self.lefthand_inv_transform(lefthand_xstart_in, inputs=left_hand_inputs)
            lefthand_global_mat = lefthand_global_out["tip_global_mat"]
            lefthand_global_pos = lefthand_global_out["tip_global_pos"]

            detach_lefthand_xstart_in = lefthand_xstart_in.clone().detach()
            detach_lefthand_global_out = self.lefthand_inv_transform(detach_lefthand_xstart_in, inputs=left_hand_inputs)
            detach_lefthand_global_mat = detach_lefthand_global_out["tip_global_mat"]
            detach_lefthand_global_pos = detach_lefthand_global_out["tip_global_pos"]

            right_hand_inputs = {}
            for k, v in inputs.items():
                if k in [
                    "right_base_rotmat",
                    "right_base_pos",
                    "right_handpose",
                    "right_hand_localpos",
                    "right_handtip_localpos",
                ]:
                    right_hand_inputs[k.replace("right_", "")] = v
                    continue
                right_hand_inputs[k] = v
            right_hand_inputs["base_rotmat"] = matrix.get_rotation(body_global_mat[..., 21, :, :])
            right_hand_inputs["base_pos"] = body_global_pos[..., 21, :]
            righthand_global_out = self.righthand_inv_transform(righthand_xstart_in, inputs=right_hand_inputs)
            righthand_global_mat = righthand_global_out["tip_global_mat"]
            righthand_global_pos = righthand_global_out["tip_global_pos"]

            detach_righthand_xstart_in = righthand_xstart_in.clone().detach()
            detach_righthand_global_out = self.righthand_inv_transform(
                detach_righthand_xstart_in, inputs=right_hand_inputs
            )
            detach_righthand_global_mat = detach_righthand_global_out["tip_global_mat"]
            detach_righthand_global_pos = detach_righthand_global_out["tip_global_pos"]

            global_pos = torch.cat(
                [body_global_pos, lefthand_global_pos[..., 1:, :], righthand_global_pos[..., 1:, :]], dim=-2
            )

            global_pos_detachhand = torch.cat(
                [body_global_pos, detach_lefthand_global_pos[..., 1:, :], detach_righthand_global_pos[..., 1:, :]],
                dim=-2,
            )

            if ((iter + 1) % 100 == 0 or iter == 0) and self.is_vis:
                wis3d = make_wis3d(name=f"debug_wholebody_{vis_i:03d}")
                add_motion_as_lines(body_global_pos[0].clone(), wis3d, name=f"{iter:03d}_optim-body")
                add_motion_as_lines(
                    lefthand_global_pos[0].clone(),
                    wis3d,
                    name=f"{iter:03d}_optim-left_hand",
                    skeleton_type="handtip",
                    radius=0.005,
                )
                add_motion_as_lines(
                    righthand_global_pos[0].clone(),
                    wis3d,
                    name=f"{iter:03d}_optim-right_hand",
                    skeleton_type="handtip",
                    radius=0.005,
                )
                for j in range(body_global_pos.shape[1]):
                    wis3d.set_scene_id(j)
                    wis3d.add_point_cloud(target[0, j, [20, 21]], name=f"target_wrist")
                    wis3d.add_point_cloud(target[0, j, left_hand_ind], name=f"target_lefthand")
                    wis3d.add_point_cloud(target[0, j, right_hand_ind], name=f"target_righthand")

                if self.save_res:
                    pred_data = {
                        "transl": body_global_out["transl"],
                        "global_orient": body_global_out["global_orient"],
                        "body_pose": body_global_out["body_pose"],
                        "left_hand_pose": lefthand_global_out["hand_pose"],
                        "right_hand_pose": righthand_global_out["hand_pose"],
                    }
                    torch.save(pred_data, f"wholebody_iter{iter:03d}.pt")

            if iter < self.only_body_steps:
                target_mask[..., 22:, :] = 0.0

            loss_dict = {}

            obj_pos = obj_verts.mean(dim=-2)[:, 0]  # (B, 3)
            obj_dist = torch.norm(obj_pos[..., [2]], dim=-1)  # (B, )
            # if vis_i == 3 or vis_i == 8 or vis_i == 12:
            #     print(obj_dist)
            #     import ipdb

            #     ipdb.set_trace()
            DIST_THRESH = 0.5  # GRAB
            # DIST_THRESH = 0.3 # ARCTIC modify
            # PRE_N = 60 # ARCTIC modify
            PRE_N = 40  # GRAB
            is_obj_far = obj_dist > DIST_THRESH
            if is_obj_far[0]:

                for i in range(is_obj_far.shape[0]):
                    if is_obj_far[i]:
                        PRE_N = min(PRE_N, target_mask.shape[1])
                        target_mask[i, 1 : PRE_N + 1] = 0.0  # (L, J, 3)
                        delta_target_mask[i, 1 : PRE_N + 1] = 0.0
                        detach_target_mask[i, 1 : PRE_N + 1] = 0.0
                        for j in range(1, PRE_N + 1, 10):
                            target_mask[i, j, 0, 0] = 1.0
                            target_mask[i, j, 0, 2] = 1.0
                            target[i, j, 0, 0] = 0.0 + j * obj_pos[i, 0] / PRE_N
                            target[i, j, 0, 2] = 0.0 + j * (obj_pos[i, 2] - DIST_THRESH) / PRE_N

                    else:
                        # already very close, do not move root
                        target_mask[i, :, 0, :] = 1.0

            IS_MOVING_OBJ = self.is_moving_obj
            if IS_MOVING_OBJ:
                if iter < 100:
                    # only optim root during first 100 steps
                    # target_mask[...] = 0.0
                    detach_target_mask[...] = 0.0
                START_T = 2 * 30
                MOVE_T = 6 * 30
                MOVE_DIST = self.moving_obj_kwargs["MOVE_DIST"]
                axis = self.moving_obj_kwargs["AXIS"]
                target_mask[:, :START_T, 0, axis] = 1.0
                for i in range(0, MOVE_T + 1, 15):
                    target[:, START_T + i, 0, axis] = i * MOVE_DIST / MOVE_T
                    target_mask[:, START_T + i, 0, axis] = 1.0
                    # target_mask[:, START_T + i, 0, 0] = 1.0
                for i in range(START_T + MOVE_T, target_mask.shape[1], 15):
                    target[:, i, 0, axis] = MOVE_DIST
                    target_mask[:, i, 0, axis] = 1.0
                    # target_mask[:, i, 0, 0] = 1.0

            #### body and hand tracking loss ####
            loss_fn = F.mse_loss if self.use_mse_loss else F.l1_loss
            body_loss_sum = (
                loss_fn(global_pos[..., :22, :], target[..., :22, :], reduction="none")
                * target_mask[..., :22, :]
                * motion_mask[..., :22, :]
            )
            hand_loss_sum = (
                loss_fn(global_pos[..., 22:, :], target[..., 22:, :], reduction="none")
                * target_mask[..., 22:, :]
                * motion_mask[..., 22:, :]
            )
            detach_hand_loss_sum = (
                loss_fn(global_pos_detachhand[..., 22:, :], target[..., 22:, :], reduction="none")
                * detach_target_mask[..., 22:, :]
                * motion_mask[..., 22:, :]
            )  # for wrist rotation
            if iter >= self.no_body_steps:
                body_loss_sum = body_loss_sum * 0.0
                detach_hand_loss_sum = detach_hand_loss_sum * 0.0
            loss_sum = (
                self.body_w * body_loss_sum.sum(dim=[1, 2, 3])
                + self.hand_w * hand_loss_sum.sum(dim=[1, 2, 3])
                + self.detach_hand_w * detach_hand_loss_sum.sum(dim=[1, 2, 3])
            )
            # average the loss over the number of valid joints
            loss_sum = loss_sum / (target_mask * motion_mask).sum(dim=[1, 2, 3])
            loss_dict["target_pos_loss"] = loss_sum
            #########################################################

            ########### head loss ###########
            head_rotmat = matrix.get_rotation(body_global_mat[..., 15, :, :])
            # z vec is the head direction
            head_dir = head_rotmat[..., :, 2]  # (B, L, 3)
            head_pos = body_global_pos[..., 15, :]  # (B, L, 3)
            target_pos = obj_verts.mean(dim=-2)  # (B, L, 3)
            target_head_dir = target_pos - head_pos  # (B, L, 3)
            target_head_dir = target_head_dir / (torch.norm(target_head_dir, dim=-1, keepdim=True) + 1e-6)
            head_loss = 1 - (head_dir * target_head_dir).sum(dim=-1)  # (B, L)
            head_loss = self.head_w * head_loss.mean()
            loss_dict["head_loss"] = head_loss
            loss_sum = loss_sum + head_loss
            #########################################################

            #### hand relative wrist loss ####
            left_finger_pos = global_pos[..., left_hand_ind, :]
            right_finger_pos = global_pos[..., right_hand_ind, :]
            left_wrist_pos = global_pos[..., [20], :]
            right_wrist_pos = global_pos[..., [21], :]
            left_delta_pos = left_finger_pos - left_wrist_pos
            right_delta_pos = right_finger_pos - right_wrist_pos

            target_left_finger_pos = target[..., left_hand_ind, :]
            target_right_finger_pos = target[..., right_hand_ind, :]
            target_left_wrist_pos = target[..., [20], :]
            target_right_wrist_pos = target[..., [21], :]
            target_left_delta_pos = target_left_finger_pos - target_left_wrist_pos
            target_right_delta_pos = target_right_finger_pos - target_right_wrist_pos

            left_delta_loss = (
                loss_fn(left_delta_pos, target_left_delta_pos, reduction="none")
                * delta_target_mask[..., left_hand_ind, :]
                * motion_mask[..., left_hand_ind, :]
            )
            right_delta_loss = (
                loss_fn(right_delta_pos, target_right_delta_pos, reduction="none")
                * delta_target_mask[..., right_hand_ind, :]
                * motion_mask[..., right_hand_ind, :]
            )
            delta_loss_sum = left_delta_loss.sum(dim=[1, 2, 3]) + right_delta_loss.sum(dim=[1, 2, 3])
            delta_loss_sum = delta_loss_sum / (delta_target_mask * motion_mask).sum(dim=[1, 2, 3])
            if iter >= self.no_body_steps:
                delta_loss_sum = delta_loss_sum * 0.0
            if iter <= self.only_body_steps:
                delta_loss_sum = delta_loss_sum * 0.0
            loss_dict["delta_loss"] = delta_loss_sum * self.delta_w
            loss_sum = loss_sum + delta_loss_sum * self.delta_w
            #########################################################

            #### root height reg loss ####
            root_global_h = body_global_pos[..., 0, 1]  # (B, L)
            root_height_reg_loss = (root_global_h - idle_root_height).abs()
            loss_dict["root_height_reg_loss"] = root_height_reg_loss * self.root_height_reg_w
            loss_sum = loss_sum + root_height_reg_loss * self.root_height_reg_w
            #########################################################

            #### foot sliding reg loss ####
            FOOT_THRESH = 0.05
            foot_ind = [7, 8, 10, 11]
            foot_global_pos = body_global_pos[..., foot_ind, :]
            foot_global_h = foot_global_pos[..., 1]  # (B, L, j)
            foot_global_vel = foot_global_pos[:, 1:, :, [0, 2]] - foot_global_pos[:, :-1, :, [0, 2]]
            foot_global_vel = torch.abs(30 * foot_global_vel).mean(dim=-1)  # (B, L-1, j)
            foot_global_h_min = foot_global_h.min(dim=-1)[0]  # at least one foot is on the ground
            foot_floating_loss_1 = torch.maximum(foot_global_h_min - 0.02, torch.zeros_like(foot_global_h_min)).mean()
            foot_floating_loss_2 = torch.maximum(-0.02 - foot_global_h_min, torch.zeros_like(foot_global_h_min)).mean()
            foot_floating_loss = foot_floating_loss_1 + foot_floating_loss_2
            foot_global_vel = foot_global_vel * (foot_global_h[:, :-1] < FOOT_THRESH).float()  # (B, L-1, j)
            foot_sliding_reg_loss = foot_global_vel.mean()
            if iter <= self.only_body_steps:
                foot_sliding_reg_loss = foot_sliding_reg_loss * 0.0
            loss_sum = (
                loss_sum
                + foot_sliding_reg_loss * self.foot_sliding_reg_w
                + foot_floating_loss * self.foot_floating_reg_w
            )
            loss_dict["foot_sliding_reg_loss"] = foot_sliding_reg_loss * self.foot_sliding_reg_w
            loss_dict["foot_floating_reg_loss"] = foot_floating_loss * self.foot_floating_reg_w
            #########################################################

            #### pene loss ####
            if iter >= self.no_pene_steps:
                # obj_verts: [B, L, M, 3]
                radius = 0.005  # 5mm
                min_dist = 0.005  # 5mm
                positive_threshold = 0.02  # 5mm
                negative_threshold = 0.02  # 5mm
                finger_ind = [
                    1,
                    2,
                    3,
                    4,  # index
                    5,
                    6,
                    7,
                    8,  # middle
                    9,
                    10,
                    11,
                    12,  # pinky
                    13,
                    14,
                    15,
                    16,  # ring
                    17,
                    18,
                    19,
                    20,  # thumb
                ]
                hand_global_mat = torch.cat(
                    [lefthand_global_mat[..., finger_ind, :, :], righthand_global_mat[..., finger_ind, :, :]],
                    dim=-3,
                )  # (B, L, J, 4, 4)
                # add finger root with wrist rot to avoid strange rot (close to obj but wierd rot to avoid y penetration)
                finger_root_ind = [1, 5, 9, 13, 17]
                left_finger_root_pos = matrix.get_position(lefthand_global_mat[..., finger_root_ind, :, :])
                right_finger_root_pos = matrix.get_position(righthand_global_mat[..., finger_root_ind, :, :])
                left_wrist_rot = matrix.get_rotation(lefthand_global_mat[..., [0], :, :])
                right_wrist_rot = matrix.get_rotation(righthand_global_mat[..., [0], :, :])
                left_wrist_rot = left_wrist_rot.expand(-1, -1, 5, -1, -1)
                right_wrist_rot = right_wrist_rot.expand(-1, -1, 5, -1, -1)
                left_finger_root_globalmat = matrix.get_TRS(left_wrist_rot, left_finger_root_pos)
                right_finger_root_globalmat = matrix.get_TRS(right_wrist_rot, right_finger_root_pos)
                hand_global_mat = torch.cat(
                    [left_finger_root_globalmat, right_finger_root_globalmat], dim=-3
                )  # (B, L, J, 4, 4)
                #########################################################

                obj_verts_in_hand = matrix.get_relative_position_to(
                    obj_verts[..., None, :, :], hand_global_mat
                )  # (B, L, J, M, 3)
                obj_verts_in_hand_y = -obj_verts_in_hand[..., 1]  # ray is along -y axis
                obj_verts_in_hand_xz = obj_verts_in_hand[..., [0, 2]]
                obj_verts_in_hand_xzdist = torch.norm(obj_verts_in_hand_xz, dim=-1)
                # not along ray assign very big number to filter out
                obj_verts_in_hand_y = obj_verts_in_hand_y + 1000000.0 * (obj_verts_in_hand_xzdist > radius).float()
                obj_verts_in_hand_signdist = obj_verts_in_hand_y.min(dim=-1)[0]  # (B, L, J)
                # filter out the verts that are too far away
                sdf_out_mask = torch.logical_or(
                    obj_verts_in_hand_signdist > positive_threshold,
                    obj_verts_in_hand_signdist < -negative_threshold,
                )
                sdf_mask = ~sdf_out_mask
                # obj verts should always in front of finger, negative sd is behind finger, should be penalized
                sdf_loss = F.relu(-obj_verts_in_hand_signdist + min_dist) * sdf_mask.float()
                sdf_loss = sdf_loss.mean()

                loss_dict["sdf_loss"] = sdf_loss * self.sdf_w
                loss_sum = loss_sum + sdf_loss * self.sdf_w
            #########################################################

        return loss_sum, loss_dict


def warmup_scheduler(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1


def cosine_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using cosine decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return (math.cos((step) / decay_steps * math.pi) + 1) / 2
    else:
        if step < total_steps - decay_steps:
            return 1
        return (math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1) / 2


def noise_regularize_1d(noise, stop_at=2, dim=3):
    """
    Args:
        noise (torch.Tensor): (N, C, 1, size)
        stop_at (int): stop decorrelating when size is less than or equal to stop_at
        dim (int): the dimension to decorrelate
    """
    all_dims = set(range(len(noise.shape)))
    loss = 0
    size = noise.shape[dim]

    # pad noise in the size dimention so that it is the power of 2
    if size != 2 ** int(math.log2(size)):
        new_size = 2 ** int(math.log2(size) + 1)
        pad = new_size - size
        pad_shape = list(noise.shape)
        pad_shape[dim] = pad
        pad_noise = torch.randn(*pad_shape).to(noise.device)

        noise = torch.cat([noise, pad_noise], dim=dim)
        size = noise.shape[dim]

    while True:
        # this loss penalizes spatially correlated noise
        # the noise is rolled in the size direction and the dot product is taken
        # (bs, )
        loss = loss + (noise * torch.roll(noise, shifts=1, dims=dim)).mean(
            # average over all dimensions except 0 (batch)
            dim=list(all_dims - {0})
        ).pow(2)

        # stop when size is 8
        if size <= stop_at:
            break

        # (N, C, 1, size) -> (N, C, 1, size // 2, 2)
        noise_shape = list(noise.shape)
        noise_shape[dim] = size // 2
        noise_shape.insert(dim + 1, 2)
        noise = noise.reshape(noise_shape)
        # average pool over (2,) window
        noise = noise.mean([dim + 1])
        size //= 2

    return loss
