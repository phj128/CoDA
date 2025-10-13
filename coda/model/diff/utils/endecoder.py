import torch
import torch.nn as nn
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from coda.configs import MainStore, builds
import coda.utils.matrix as matrix
from coda.utils.pylogger import Log
from coda.dataset.arctic.utils import *
from coda.utils.smplx_utils import make_smplx, detect_foot_contact
from . import stats_compose


class EnDecoder(nn.Module):
    def __init__(self, stats_name="DEFAULT_01"):
        super().__init__()
        # Load mean, std
        self.vae = None
        stats = getattr(stats_compose, stats_name)
        Log.info(f"[EnDecoder] Use {stats_name} for statistics!")
        self.register_buffer("mean", torch.tensor(stats["mean"]).float(), False)
        self.register_buffer("std", torch.tensor(stats["std"]).float(), False)
        self.stats = stats

    def _encode(self, inputs):
        raise NotImplementedError

    def _decode(self, x_norm, inputs=None):
        raise NotImplementedError

    def encode(self, inputs):
        x_norm = self._encode(inputs)
        if self.vae is not None:
            if len(x_norm.shape) == 2:
                x_norm = x_norm[None]
                length = torch.tensor(inputs["length"], device=x_norm.device)[None]  # (1)
                flag = True
            else:
                length = inputs["length"]
                flag = False
            sampled_z, dist, mask = self.vae.encode(x_norm, length)
            latent = dist.loc
            latent_norm = (latent - self.latent_mean) / self.latent_std
            if flag:
                latent_norm = latent_norm[:, 0]  # (K, B, C) -> (K, C), B = 1
            else:
                latent_norm = latent_norm.transpose(0, 1)  # (K, B, C) -> (B, K, C)
            return latent_norm
        else:
            return x_norm

    def encode_condition(self, inputs):
        raise NotImplementedError

    def decode(self, x_norm, inputs=None):
        if self.vae is not None:
            latent_norm = x_norm
            latent = (latent_norm * self.latent_std) + self.latent_mean
            if len(latent_norm.shape) == 2:
                latent = latent[:, None]  # (K, C) -> (K, B, C)
                length = torch.tensor(inputs["length"], device=latent.device)[None]  # (1)
                flag = True
            else:
                latent = latent.transpose(0, 1)  # (B, K, C) -> (K, B, C)
                length = inputs["length"]
                flag = False

            x_norm = self.vae.decode(latent, length, max_len=300)

            if flag:
                x_norm = x_norm[0]
        decode_out = self._decode(x_norm, inputs=inputs)
        return decode_out

    def set_vae(self, vae):
        self.vae = vae
        if vae is not None:
            if "latent_mean" in self.stats:
                self.register_buffer("latent_mean", torch.tensor(self.stats["latent_mean"]).float(), False)
                self.register_buffer("latent_std", torch.tensor(self.stats["latent_std"]).float(), False)
            else:
                self.register_buffer("latent_mean", torch.tensor([0.0]).float(), False)
                self.register_buffer("latent_std", torch.tensor([1.0]).float(), False)


class BodyPoseEnDecoder(EnDecoder):
    def __init__(self, stats_name="DEFAULT_01", is_angleay=False):
        super().__init__(stats_name)
        self.is_angleay = is_angleay
        stats = getattr(stats_compose, stats_name)
        if "beta_mean" in stats:
            self.register_buffer("beta_mean", torch.tensor(stats["beta_mean"]).float(), False)
            self.register_buffer("beta_std", torch.tensor(stats["beta_std"]).float(), False)
        else:
            self.register_buffer("beta_mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("beta_std", torch.tensor(stats["std"]).float(), False)
        if "skeleton_mean" in stats:
            self.register_buffer("skeleton_mean", torch.tensor(stats["skeleton_mean"]).float(), False)
            self.register_buffer("skeleton_std", torch.tensor(stats["skeleton_std"]).float(), False)
        else:
            self.register_buffer("skeleton_mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("skeleton_std", torch.tensor(stats["std"]).float(), False)
        self.parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        self.is_to_ayfz = True

    def encode_condition(self, inputs):
        """
        definition: {
            }
        """
        beta = inputs["beta"]
        skeleton = inputs["skeleton"][..., 1:22, :].clone()  # (B, 1, 21, 3)
        skeleton_vec = skeleton.flatten(-2)  # (B, 1, 63)

        beta_norm = (beta - self.beta_mean) / self.beta_std
        skeleton_norm = (skeleton_vec - self.skeleton_mean) / self.skeleton_std

        condition_dict = {
            "beta": beta_norm,
            "skeleton": skeleton_norm,
        }

        return condition_dict

    def _encode(self, inputs):
        """
        definition: {
            }
        """
        transl = inputs["transl"]  # (B, N, 3)
        global_orient = inputs["global_orient"]  # (B, N, 3)
        body_pose = inputs["body_pose"]  # (B, N, 63)
        skeleton = inputs["skeleton"][..., :22, :]  # (B, 1, 22, 3)

        # if len(transl.shape) == 2:
        #     L = transl.shape[0]
        #     transl = torch.cat([transl, transl[-1:, :].repeat(300 - L, 1)], dim=-2)
        #     global_orient = torch.cat([global_orient, global_orient[-1:, :].repeat(300 - L, 1)], dim=-2)
        #     body_pose = torch.cat([body_pose, body_pose[-1:, :].repeat(300 - L, 1)], dim=-2)

        is_to_ayfz = self.is_to_ayfz
        if is_to_ayfz:
            ###### convert to ayfz ######
            idle_root_transl = skeleton[..., 0, :]  # (B, 1, 3)
            all_pose = torch.cat([global_orient, body_pose], dim=-1)
            rotmat = matrix.axis_angle_to_matrix(all_pose.reshape(all_pose.shape[:-1] + (-1, 3)))
            transl_with_zero = transl + idle_root_transl  # (B, N, 3)
            local_pos = torch.cat(
                [transl_with_zero[..., None, :], skeleton[..., 1:, :] + torch.zeros_like(transl)[..., None, :]], dim=-2
            )
            mat = matrix.get_TRS(rotmat, local_pos)
            global_mat = matrix.forward_kinematics(mat, self.parent)
            global_orient_quat = matrix.quat_wxyz2xyzw(matrix.axis_angle_to_quaternion(global_orient))
            ay_quat = matrix.calc_heading_quat(global_orient_quat, head_ind=2, gravity_axis="y")  # (B, N, 4)
            ay_rotmat = matrix.quaternion_to_matrix(matrix.quat_xyzw2wxyz(ay_quat))  # (B, N, 3, 3)
            ay_rootmat = matrix.get_TRS(ay_rotmat, matrix.get_position(global_mat)[..., 0, :])  # (B, N, 4, 4)
            ayfz_global_mat = matrix.get_mat_BtoA(ay_rootmat[..., :1, None, :, :], global_mat)  # (B, N, J, 4, 4)
            ## floor = 0
            if len(transl.shape) == 2:
                floor_height = matrix.get_position(ayfz_global_mat)[..., 1].min()
                new_global_pos = matrix.get_position(ayfz_global_mat).clone()
                new_global_pos[..., 1] = new_global_pos[..., 1] - floor_height
                ayfz_global_mat = matrix.set_position(ayfz_global_mat, new_global_pos)
            else:
                B = transl.shape[0]
                motion_y = matrix.get_position(ayfz_global_mat)[..., 1]  # (B, L, J)
                motion_y = motion_y.reshape(B, -1)  # (B, L * J)
                floor_height = motion_y.min(dim=-1)[0]  # (B, )
                new_global_pos = matrix.get_position(ayfz_global_mat).clone()  # (B, L, J, 3)
                new_global_pos[..., 1] = new_global_pos[..., 1] - floor_height[..., None, None]  # (B, L, J)
                ayfz_global_mat = matrix.set_position(ayfz_global_mat, new_global_pos)
            ###########

            ayfz_root_mat = ayfz_global_mat[..., 0, :, :]
            ayfz_transl = matrix.get_position(ayfz_root_mat) - idle_root_transl
            ayfz_global_orient = matrix.matrix_to_axis_angle(matrix.get_rotation(ayfz_root_mat))
            transl = ayfz_transl
            global_orient = ayfz_global_orient
            #############################

        root_transl = transl + skeleton[..., 0, :]  # (B, N, 3)

        rotmat = matrix.axis_angle_to_matrix(
            torch.cat([global_orient, body_pose], dim=-1).reshape(body_pose.shape[:-1] + (-1, 3))
        )
        local_pos = torch.cat(
            [root_transl[..., None, :], skeleton[..., 1:, :] + torch.zeros_like(root_transl)[..., None, :]], dim=-2
        )
        local_mat = matrix.get_TRS(rotmat, local_pos)
        global_mat = matrix.forward_kinematics(local_mat, self.parent)
        global_pos = matrix.get_position(global_mat)
        contact_l, contact_r = detect_foot_contact(global_pos)
        contact = torch.cat([contact_l, contact_r], dim=-1)

        fps = 30.0
        root_h = root_transl[..., 1:2]  # (B, N, 1)
        global_orient_rotmat = matrix.axis_angle_to_matrix(global_orient)  # (B, N, 3, 3)
        global_orient_quat = matrix.quat_wxyz2xyzw(matrix.axis_angle_to_quaternion(global_orient))  # (B, N, 4)
        ay_root_quat = matrix.calc_heading_quat(global_orient_quat, head_ind=2, gravity_axis="y")  # (B, N, 4)
        ay_root_rotmat = matrix.quaternion_to_matrix(matrix.quat_xyzw2wxyz(ay_root_quat))  # (B, N, 3, 3)
        ay_root_mat = matrix.get_TRS(ay_root_rotmat, root_transl)  # (B, N, 4, 4)
        ay_root_mat_plus1 = torch.cat([ay_root_mat, ay_root_mat[..., -1:, :, :]], dim=-3)  # (B, N+1, 4, 4)
        ay_root_vel = matrix.get_mat_BtoA(ay_root_mat, ay_root_mat_plus1[..., 1:, :, :])  # (B, N, 4, 4)
        ay_root_xzvel = matrix.get_position(ay_root_vel)[..., [0, 2]] * fps  # (B, N, 2)

        ay_root_rotmat_plus1 = torch.cat([ay_root_rotmat, ay_root_rotmat[..., -1:, :, :]], dim=-3)  # (B, N+1, 3, 3)
        if self.is_angleay:
            ### discontinuous calcualtion, do not use ###
            # ay_root_quat_plus1 = torch.cat([ay_root_quat, ay_root_quat[..., -1:, :]], dim=-2) # (B, N+1, 4)
            # ay_angle_plus1 = torch.asin(ay_root_quat_plus1[..., 1:2]) # this has discontinuous
            # ay_angle_vel = (ay_angle_plus1[..., 1:, :] - ay_angle_plus1[..., :-1, :])
            # ay_rot_vec = ay_angle_vel
            #############################################

            ay_root_rotmat_rel = matrix.get_mat_BtoA(
                ay_root_rotmat, ay_root_rotmat_plus1[..., 1:, :, :]
            )  # (B, N, 3, 3)
            ay_root_quat_rel = matrix.matrix_to_quaternion(ay_root_rotmat_rel)  # (B, N, 4)
            ay_root_angle_vel = torch.asin(ay_root_quat_rel[..., 2:3])
            ay_rot_vec = ay_root_angle_vel
        else:
            ay_root_rotmat_rel = matrix.get_mat_BtoA(
                ay_root_rotmat, ay_root_rotmat_plus1[..., 1:, :, :]
            )  # (B, N, 3, 3)
            ay_root_xzdir_rel = matrix.rotmat2xzdir(ay_root_rotmat_rel)  # (B, N, 4)
            ay_rot_vec = ay_root_xzdir_rel

        global_orient_aftery_rotmat = matrix.get_mat_BtoA(ay_root_rotmat, global_orient_rotmat)  # (B, N, 3, 3)
        global_orient_aftery = matrix.matrix_to_axis_angle(global_orient_aftery_rotmat)  # (B, N, 3)
        pose = torch.cat([global_orient_aftery, body_pose], dim=-1)  # (B, N, 66)
        pose_6d = matrix.axis_angle_to_rotation_6d(pose.reshape(pose.shape[:-1] + (-1, 3)))  # (B, N, 22, 6)
        pose_6d = pose_6d.flatten(-2)  # (B, N, 132)

        x = torch.cat([ay_root_xzvel, root_h, ay_rot_vec, pose_6d, contact], dim=-1)  # (B, N, 143)
        std_ = self.std.clone()
        std_[std_ < 1e-4] = 1.0
        x_norm = (x - self.mean) / std_
        return x_norm

    def _decode(self, x_norm, inputs=None):
        fps = 30.0
        x = (x_norm * self.std) + self.mean
        ay_root_xzvel = x[..., :2] / fps
        root_h = x[..., 2]
        if self.is_angleay:
            accum_dim = 4
            ay_angle_vel = x[..., 3:4]
        else:
            accum_dim = 7
            ay_root_xzdir_rel = x[..., 3:7]
        pose_6d = x[..., accum_dim : accum_dim + 132]
        cotnact = x[..., accum_dim + 132 : accum_dim + 132 + 4]
        pose_6d = pose_6d.reshape(pose_6d.shape[:-1] + (-1, 6))
        pose = matrix.rotation_6d_to_axis_angle(pose_6d)  # (B, N, 22, 3)
        pose = pose.flatten(-2)  # (B, N, 66)

        body_pose = pose[..., 3:]

        global_orient_aftery = pose[..., :3]
        global_orient_aftery_rotmat = matrix.axis_angle_to_matrix(global_orient_aftery)  # (B, N, 3, 3)

        if self.is_angleay:
            ay_angle = torch.zeros_like(ay_angle_vel)  # (B, N, 1)
            ay_angle = torch.cat([ay_angle[..., :1, :], ay_angle_vel[..., :-1, :]], dim=-2)  # (B, N, 1)
            ay_angle = torch.cumsum(ay_angle, dim=-2)  # (B, N, 1)

            quat_root_y = torch.zeros(x_norm.shape[:-1] + (4,), device=x_norm.device)
            quat_root_y = torch.cat(
                [torch.cos(ay_angle), torch.zeros_like(ay_angle), torch.sin(ay_angle), torch.zeros_like(ay_angle)],
                dim=-1,
            )

            ay_root_rotmat = matrix.quaternion_to_matrix(quat_root_y)  # (B, N, 3, 3)

        else:
            ay_root_rotmat_rel = matrix.xzdir2rotmat(ay_root_xzdir_rel)  # (B, N, 3, 3)
            ay_root_rotmat = matrix.identity_mat(ay_root_rotmat_rel, device=x.device)[..., :3, :3]
            for i in range(1, ay_root_rotmat_rel.shape[-3]):
                ay_root_rotmat_ = matrix.get_mat_BfromA(
                    ay_root_rotmat[..., i - 1, :, :], ay_root_rotmat_rel[..., i - 1, :, :]
                )
                ay_root_rotmat = torch.cat(
                    [
                        ay_root_rotmat[..., :i, :, :],
                        ay_root_rotmat_[..., None, :, :],
                        ay_root_rotmat[..., i + 1 :, :, :],
                    ],
                    dim=-3,
                )

        global_orient_rotmat = matrix.get_mat_BfromA(ay_root_rotmat, global_orient_aftery_rotmat)
        global_orient = matrix.matrix_to_axis_angle(global_orient_rotmat)

        ay_root_vel = torch.cat(
            [ay_root_xzvel[..., :1], torch.zeros_like(ay_root_xzvel[..., 1:2]), ay_root_xzvel[..., 1:2]], dim=-1
        )
        ay_root_vel_inayfz = matrix.get_position_from_rotmat(ay_root_vel, ay_root_rotmat)
        ay_root_pos = torch.cumsum(ay_root_vel_inayfz, dim=-2)  # (B, N, 3)
        # first frame xz is zero
        ay_root_pos = torch.cat(
            [torch.zeros_like(ay_root_pos[..., :1, :]), ay_root_pos[..., :-1, :]], dim=-2
        )  # (B, N, 3)
        ay_root_pos = torch.cat([ay_root_pos[..., :1], root_h[..., None], ay_root_pos[..., 2:]], dim=-1)  # (B, N, 3)

        output = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "root_pos": ay_root_pos,
        }

        if inputs is not None:
            zero_transl = inputs["skeleton"][..., 0, :]  # (B, N, 3)
            transl = ay_root_pos - zero_transl
            output["transl"] = transl

            skeleton = inputs["skeleton"][..., :22, :]
            rotmat = matrix.axis_angle_to_matrix(
                torch.cat([global_orient, body_pose], dim=-1).reshape(body_pose.shape[:-1] + (-1, 3))
            )
            local_pos = torch.cat(
                [ay_root_pos[..., None, :], skeleton[..., 1:, :] + torch.zeros_like(ay_root_pos)[..., None, :]], dim=-2
            )
            local_mat = matrix.get_TRS(rotmat, local_pos)
            global_mat = matrix.forward_kinematics(local_mat, self.parent)
            output["global_mat"] = global_mat
            global_pos = matrix.get_position(global_mat)
            output["global_pos"] = global_pos

        return output


group_name = "endecoder/diffusion"
cfg_base = builds(BodyPoseEnDecoder, populate_full_signature=True)
MainStore.store(
    name="v1_bodypose",
    node=cfg_base(stats_name="BODYPOSE_HML3D"),
    group=group_name,
)

MainStore.store(
    name="v1_bodypose_hml3darcticgrab",
    node=cfg_base(stats_name="BODYPOSE_HML3D_ARCTIC_GRAB"),
    group=group_name,
)
MainStore.store(
    name="v1_bodypose_hml3darcticgrab_angley",
    node=cfg_base(stats_name="BODYPOSE_HML3D_ARCTIC_GRAB_ANGLEY", is_angleay=True),
    group=group_name,
)


class HandPoseEnDecoder(EnDecoder):
    def __init__(self, stats_name="DEFAULT_01", is_nobeta=False):
        super().__init__(stats_name)
        self.is_nobeta = is_nobeta
        stats = getattr(stats_compose, stats_name)
        if "beta_mean" in stats:
            self.register_buffer("beta_mean", torch.tensor(stats["beta_mean"]).float(), False)
            self.register_buffer("beta_std", torch.tensor(stats["beta_std"]).float(), False)
        else:
            self.register_buffer("beta_mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("beta_std", torch.tensor(stats["std"]).float(), False)
        self.parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        self.tip_parent = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    def encode_condition(self, inputs, obj_outputs=None):
        """
        definition: {
            }
        """
        beta = inputs["beta"]

        beta_norm = (beta - self.beta_mean) / self.beta_std

        if self.is_nobeta:
            beta_norm = torch.zeros_like(beta_norm)

        obj_mat = inputs["obj"].clone()  # (B, N, 4, 4)
        if len(obj_mat.shape) == 5:
            obj_mat = obj_mat[..., 0, :, :]
        if obj_outputs is not None:
            obj_mat = obj_outputs

        obj_mat_plus1 = torch.cat([obj_mat, obj_mat[..., -1:, :, :]], dim=-3)  # (B, N+1, 4, 4)
        obj_relative_mat = matrix.get_mat_BtoA(obj_mat, obj_mat_plus1[:, 1:])
        obj_relative_transl = matrix.get_position(obj_relative_mat) * 30.0
        obj_relative_rot = matrix.get_rotation(obj_relative_mat)
        obj_relative_rot6d = matrix.matrix_to_rotation_6d(obj_relative_rot)
        objtraj = torch.cat([obj_relative_transl, obj_relative_rot6d], dim=-1)

        condition_dict = {
            "objtraj": objtraj,
            "beta": beta_norm,
        }

        return condition_dict

    def _encode(self, inputs):
        """
        definition: {
            }
        """
        hand_localmat = inputs["handpose"]
        hand_rotmat = matrix.get_rotation(hand_localmat)
        hand_rot6d = matrix_to_rotation_6d(hand_rotmat)
        hand_rot6d = hand_rot6d.flatten(-2)
        x = hand_rot6d
        std_ = self.std.clone()
        std_[std_ < 1e-4] = 1.0
        x_norm = (x - self.mean) / std_
        return x_norm

    def _decode(self, x_norm, inputs=None):
        """x_norm: (B, L, C)"""
        x = (x_norm * self.std) + self.mean
        flag = False
        if len(x.shape) == 2:
            x = x[None]
            flag = True
        B, L, _ = x.shape
        hand_rot6d = x.reshape(B, L, -1, 6)
        hand_rotmat = rotation_6d_to_matrix(hand_rot6d)
        hand_pose = matrix.matrix_to_axis_angle(hand_rotmat)
        if flag:
            x = x[0]
            hand_rotmat = hand_rotmat[0]
            hand_pose = hand_pose[0]
        output = {
            "hand_rotmat": hand_rotmat,
            "hand_pose": hand_pose,
        }
        if inputs is not None:
            hand_localpos = inputs["hand_localpos"].clone()
            hand_localmat = matrix.get_TRS(hand_rotmat, hand_localpos)
            identity_mat = matrix.identity_mat(hand_localmat, device=hand_localmat.device)
            # attach wrist mat
            hand_localmat = torch.cat([identity_mat[..., :1, :, :], hand_localmat], dim=-3)
            output["hand_localmat"] = hand_localmat
            global_mat = matrix.forward_kinematics(hand_localmat, self.parent)

            base_rotmat = inputs["base_rotmat"]
            base_pos = inputs["base_pos"]
            base_mat = matrix.get_TRS(base_rotmat, base_pos)
            global_mat = matrix.get_mat_BfromA(base_mat[..., None, :, :], global_mat)
            output["hand_globalmat"] = global_mat
            output["global_pos"] = matrix.get_position(global_mat)

            if "handtip_localpos" in inputs.keys():
                handtip_localpos = inputs["handtip_localpos"].clone()
                handtip_localmat = rotation_6d_to_matrix(hand_rot6d)
                handtip_localmat_shape = handtip_localmat.shape
                handtip_localmat = handtip_localmat.reshape(handtip_localmat_shape[:-3] + (5, -1, 3, 3))
                identity_rotmat = matrix.get_rotation(
                    matrix.identity_mat(handtip_localmat, device=handtip_localmat.device)
                )
                handtip_localmat = torch.cat([handtip_localmat, identity_rotmat[..., :1, :, :]], dim=-3)
                handtip_localmat = handtip_localmat.reshape(handtip_localmat_shape[:-3] + (-1, 3, 3))
                handtip_localmat = matrix.get_TRS(handtip_localmat, handtip_localpos)
                handtip_localmat = torch.cat([identity_mat[..., :1, :, :], handtip_localmat], dim=-3)
                output["handtip_localmat"] = handtip_localmat
                handtip_global_mat = matrix.forward_kinematics(handtip_localmat, self.tip_parent)
                tip_global_mat = matrix.get_mat_BfromA(base_mat[..., None, :, :], handtip_global_mat)
                output["tip_global_mat"] = tip_global_mat
                output["tip_global_pos"] = matrix.get_position(tip_global_mat)

        return output


group_name = "endecoder/diffusion"
cfg_base = builds(HandPoseEnDecoder, populate_full_signature=True)
MainStore.store(
    name="v1_left_handpose",
    node=cfg_base(stats_name="LEFTHANDPOSE_ARCTIC"),
    group=group_name,
)
MainStore.store(
    name="v1_right_handpose",
    node=cfg_base(stats_name="RIGHTHANDPOSE_ARCTIC"),
    group=group_name,
)
MainStore.store(
    name="v1_left_handpose_arcticgrab",
    node=cfg_base(stats_name="LEFTHANDPOSE_ARCTIC_GRAB"),
    group=group_name,
)
MainStore.store(
    name="v1_right_handpose_arcticgrab",
    node=cfg_base(stats_name="RIGHTHANDPOSE_ARCTIC_GRAB"),
    group=group_name,
)
MainStore.store(
    name="v1_left_handpose_arcticgrabinterhand",
    node=cfg_base(stats_name="LEFTHANDPOSE_ARCTIC_GRAB_INTERHAND"),
    group=group_name,
)
MainStore.store(
    name="v1_right_handpose_arcticgrabinterhand",
    node=cfg_base(stats_name="RIGHTHANDPOSE_ARCTIC_GRAB_INTERHAND"),
    group=group_name,
)
MainStore.store(
    name="v1_left_handpose_arcticgrabinterhandhot3d",
    node=cfg_base(stats_name="LEFTHANDPOSE_ARCTIC_GRAB_INTERHAND_HOT3D"),
    group=group_name,
)
MainStore.store(
    name="v1_right_handpose_arcticgrabinterhandhot3d",
    node=cfg_base(stats_name="RIGHTHANDPOSE_ARCTIC_GRAB_INTERHAND_HOT3D"),
    group=group_name,
)


group_name = "endecoder1/diffusion"
cfg_base = builds(HandPoseEnDecoder, populate_full_signature=True)
MainStore.store(
    name="v1_left_handpose",
    node=cfg_base(stats_name="LEFTHANDPOSE_ARCTIC"),
    group=group_name,
)
MainStore.store(
    name="v1_left_handpose_arcticgrab",
    node=cfg_base(stats_name="LEFTHANDPOSE_ARCTIC_GRAB"),
    group=group_name,
)
MainStore.store(
    name="v1_left_handpose_arcticgrabinterhandhot3d",
    node=cfg_base(stats_name="LEFTHANDPOSE_ARCTIC_GRAB_INTERHAND_HOT3D"),
    group=group_name,
)


group_name = "endecoder2/diffusion"
cfg_base = builds(HandPoseEnDecoder, populate_full_signature=True)
MainStore.store(
    name="v1_right_handpose",
    node=cfg_base(stats_name="RIGHTHANDPOSE_ARCTIC"),
    group=group_name,
)
MainStore.store(
    name="v1_right_handpose_arcticgrab",
    node=cfg_base(stats_name="RIGHTHANDPOSE_ARCTIC_GRAB"),
    group=group_name,
)
MainStore.store(
    name="v1_right_handpose_arcticgrabinterhandhot3d",
    node=cfg_base(stats_name="RIGHTHANDPOSE_ARCTIC_GRAB_INTERHAND_HOT3D"),
    group=group_name,
)


class BPSTrajEnDecoder(EnDecoder):
    def __init__(
        self,
        stats_name="DEFAULT_01",
        is_norot=False,
        is_zerorot=False,
        is_nobeta=False,
        is_trajrope=False,
        is_in_world=False,
    ):
        super().__init__(stats_name)
        self.is_norot = is_norot
        self.is_zerorot = is_zerorot
        self.is_nobeta = is_nobeta
        self.is_trajrope = is_trajrope
        self.is_in_world = is_in_world
        stats = getattr(stats_compose, stats_name)
        if "objtraj_mean" in stats:
            self.register_buffer("objtraj_mean", torch.tensor(stats["objtraj_mean"]).float(), False)
            self.register_buffer("objtraj_std", torch.tensor(stats["objtraj_std"]).float(), False)
        else:
            self.register_buffer("objtraj_mean", torch.tensor([0.0]).float(), False)
            self.register_buffer("objtraj_std", torch.tensor([1.0]).float(), False)
        if "beta_mean" in stats:
            self.register_buffer("beta_mean", torch.tensor(stats["beta_mean"]).float(), False)
            self.register_buffer("beta_std", torch.tensor(stats["beta_std"]).float(), False)
        else:
            self.register_buffer("beta_mean", torch.tensor([0.0]).float(), False)
            self.register_buffer("beta_std", torch.tensor([1.0]).float(), False)
        if "bps_mean" in stats:
            self.register_buffer("bps_mean", torch.tensor(stats["bps_mean"]).float(), False)
            self.register_buffer("bps_std", torch.tensor(stats["bps_std"]).float(), False)
        else:
            self.register_buffer("bps_mean", torch.tensor([0.0]).float(), False)
            self.register_buffer("bps_std", torch.tensor([1.0]).float(), False)

    def encode_condition(self, inputs, obj_outputs=None):
        """
        definition: {
            }
        """
        objframe0 = inputs["obj_frame0"]  # (B, 2, 4, 4)
        objframe0_pos = matrix.get_position(objframe0[..., 0, :, :])  # (B, 3)
        obj_mat = inputs["obj"]  # (B, N, 2, 4, 4)
        beta = inputs["beta"]
        scale = inputs["scale"]
        angles = inputs["angles"]
        bps = inputs["bps"]
        fps = 30.0
        contact = inputs["contact"]
        contact_mask = torch.any(contact, dim=-1, keepdim=True).float()  # (B, N, 1)
        if obj_outputs is not None:
            contact_mask = (obj_outputs["contact_mask"] > 0.95).float()

        if self.is_norot:
            obj_mat_plus1 = torch.cat([obj_mat, obj_mat[..., -1:, :, :, :]], dim=-4)  # (B, N+1, 2, 4, 4)
            obj_pos = matrix.get_position(obj_mat_plus1[..., 0, :, :])  # (B, N+1, 3)
            obj_vel = obj_pos[..., 1:, :] - obj_pos[..., :-1, :]  # (B, N, 3)
            scale = torch.ones_like(obj_vel[..., :1]) * scale[..., None]
            objtraj = torch.cat([obj_vel * fps, scale], dim=-1)  # (B, N, 4)
        else:
            obj_mat_plus1 = torch.cat([obj_mat, obj_mat[..., -1:, :, :, :]], dim=-4)  # (B, N+1, 2, 4, 4)
            obj_mat_tp1_to_t = matrix.get_mat_BtoA(obj_mat, obj_mat_plus1[..., 1:, :, :, :])  # (B, N, 2, 4, 4)
            obj_mat_tp1_to_t = obj_mat_tp1_to_t[..., 0, :, :]
            obj_vel = matrix.get_position(obj_mat_tp1_to_t)  # (B, N, 3)
            obj_rotvel = matrix.get_rotation(obj_mat_tp1_to_t)  # (B, N, 3, 3)
            obj_rotvel = matrix_to_rotation_6d(obj_rotvel)  # (B, N, 6)
            obj_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, N, 3, 3)
            obj_rot = matrix_to_rotation_6d(obj_rot)  # (B, N, 6)
            if self.is_zerorot:
                obj_rot = torch.zeros_like(obj_rot)
            scale = torch.ones_like(obj_vel[..., :1]) * scale[..., None]
            obj_global_pos = matrix.get_position(obj_mat[..., 0, :, :])  # (B, N, 3)
            obj_relpos_to0 = obj_global_pos - objframe0_pos[..., None, :]  # (B, N, 3)
            obj_relpos_to0[..., 1] = obj_global_pos[..., 1]  # use absolute height
            if self.is_in_world:
                obj_relpos_to0[..., 0] = obj_global_pos[..., 0]
                obj_relpos_to0[..., 2] = obj_global_pos[..., 2]
            if self.is_trajrope:
                # objtraj = torch.cat([obj_rot, scale, angles], dim=-1)  # (B, N, 8)
                objtraj = torch.cat([obj_rot, scale, angles, obj_relpos_to0], dim=-1)  # (B, N, 11)
            else:
                objtraj = torch.cat(
                    [obj_vel * fps, obj_rotvel, obj_rot, scale, angles, obj_relpos_to0], dim=-1
                )  # (B, N, 20)

        objtrajmat = obj_mat[..., 0, :, :]

        objtraj_norm = (objtraj - self.objtraj_mean) / self.objtraj_std
        beta_norm = (beta - self.beta_mean) / self.beta_std
        bps_norm = (bps - self.bps_mean) / self.bps_std
        if self.is_nobeta:
            beta_norm = torch.zeros_like(beta_norm)

        condition_dict = {
            "objtraj": objtraj_norm,
            "beta": beta_norm,
            "bps": bps_norm,
            "trajmat": objtrajmat,
            "contact_mask": contact_mask,
        }

        return condition_dict

    def _encode(self, inputs):
        """
        definition: {
            }
        """
        contact = inputs["contact"]  # (N, M, 10)
        finger_dist = inputs["finger_dist"]  # (N, M, 12)
        finger_dist = finger_dist.flatten(-2)  # (N, M*12)
        vert_dist = inputs["finger_vert_dist"]  # (N, 10)
        x = torch.cat([finger_dist, contact.float(), vert_dist], dim=-1)  # (N, M*12+10+10)

        std_ = self.std.clone()
        std_[std_ < 1e-4] = 1.0
        x_norm = (x - self.mean) / std_
        return x_norm

    def _decode(self, x_norm, inputs=None):
        """x_norm: (B, L, C)"""
        x = (x_norm * self.std) + self.mean
        finger_dist = x[..., :-20]
        contact = x[..., -20:-10]
        vert_dist = x[..., -10:]

        output = {
            "finger_dist": finger_dist,
            "contact": contact,
            "finger_vert_dist": vert_dist,
        }

        return output


group_name = "endecoder/diffusion"
cfg_base = builds(BPSTrajEnDecoder, populate_full_signature=True)
MainStore.store(
    name="v1_bps",
    node=cfg_base(stats_name="BPSTRAJ_ARCTIC"),
    group=group_name,
)
cfg_base = builds(BPSTrajEnDecoder, populate_full_signature=True)
MainStore.store(
    name="v1_bps_tip",
    node=cfg_base(stats_name="BPSTIPTRAJROT_ARCTIC"),
    group=group_name,
)
MainStore.store(
    name="v1_bps_norot",
    node=cfg_base(stats_name="BPSTIPTRAJNOROT_ARCTIC", is_norot=True),
    group=group_name,
)
group_name = "endecoder3/diffusion"
cfg_base = builds(BPSTrajEnDecoder, populate_full_signature=True)
MainStore.store(
    name="v1_bps",
    node=cfg_base(stats_name="BPSTRAJ_ARCTIC"),
    group=group_name,
)
cfg_base = builds(BPSTrajEnDecoder, populate_full_signature=True)
MainStore.store(
    name="v1_bps_tip",
    node=cfg_base(stats_name="BPSTIPTRAJROT_ARCTIC"),
    group=group_name,
)


class GRABBPSTrajEnDecoder(BPSTrajEnDecoder):
    def encode_condition(self, inputs, obj_outputs=None):
        """
        definition: {
            }
        """
        objframe0 = inputs["obj_frame0"]  # (B, 2, 4, 4)
        objframe0_pos = matrix.get_position(objframe0[..., 0, :, :])  # (B, 3)
        obj_mat = inputs["obj"]  # (B, N, 2, 4, 4)
        beta = inputs["beta"]
        scale = inputs["scale"]
        bps = inputs["bps"]
        fps = 30.0

        contact = inputs["contact"]
        contact_mask = torch.any(contact, dim=-1, keepdim=True).float()  # (B, N, 1)
        if obj_outputs is not None:
            contact_mask = (obj_outputs["contact_mask"] > 0.95).float()

        if self.is_norot:
            obj_mat_plus1 = torch.cat([obj_mat, obj_mat[..., -1:, :, :, :]], dim=-4)  # (B, N+1, 2, 4, 4)
            obj_pos = matrix.get_position(obj_mat_plus1[..., 0, :, :])  # (B, N+1, 3)
            obj_vel = obj_pos[..., 1:, :] - obj_pos[..., :-1, :]  # (B, N, 3)
            scale = torch.ones_like(obj_vel[..., :1]) * scale[..., None]
            objtraj = torch.cat([obj_vel * fps, scale], dim=-1)  # (B, N, 4)
        else:
            obj_mat_plus1 = torch.cat([obj_mat, obj_mat[..., -1:, :, :, :]], dim=-4)  # (B, N+1, 2, 4, 4)
            obj_mat_tp1_to_t = matrix.get_mat_BtoA(obj_mat, obj_mat_plus1[..., 1:, :, :, :])  # (B, N, 2, 4, 4)
            obj_mat_tp1_to_t = obj_mat_tp1_to_t[..., 0, :, :]
            obj_vel = matrix.get_position(obj_mat_tp1_to_t)  # (B, N, 3)
            obj_rotvel = matrix.get_rotation(obj_mat_tp1_to_t)  # (B, N, 3, 3)
            obj_rotvel = matrix_to_rotation_6d(obj_rotvel)  # (B, N, 6)
            obj_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, N, 3, 3)
            obj_rot = matrix_to_rotation_6d(obj_rot)  # (B, N, 6)
            if self.is_zerorot:
                obj_rot = torch.zeros_like(obj_rot)
            scale = torch.ones_like(obj_vel[..., :1]) * scale[..., None]
            obj_global_pos = matrix.get_position(obj_mat[..., 0, :, :])  # (B, N, 3)
            obj_relpos_to0 = obj_global_pos - objframe0_pos[..., None, :]  # (B, N, 3)
            obj_relpos_to0[..., 1] = obj_global_pos[..., 1]  # use absolute height
            if self.is_in_world:
                obj_relpos_to0[..., 0] = obj_global_pos[..., 0]
                obj_relpos_to0[..., 2] = obj_global_pos[..., 2]
            if self.is_trajrope:
                objtraj = torch.cat([obj_rot, scale, obj_relpos_to0], dim=-1)  # (B, N, 10)
            else:
                objtraj = torch.cat([obj_vel * fps, obj_rotvel, obj_rot, scale, obj_relpos_to0], dim=-1)  # (B, N, 19)

        objtrajmat = obj_mat[..., 0, :, :]

        objtraj_norm = (objtraj - self.objtraj_mean) / self.objtraj_std
        beta_norm = (beta - self.beta_mean) / self.beta_std
        bps_norm = (bps - self.bps_mean) / self.bps_std
        if self.is_nobeta:
            beta_norm = torch.zeros_like(beta_norm)

        condition_dict = {
            "objtraj": objtraj_norm,
            "beta": beta_norm,
            "bps": bps_norm,
            "trajmat": objtrajmat,
            "contact_mask": contact_mask,
        }

        return condition_dict


class ARCTICTrajEnDecoder(EnDecoder):
    def __init__(self, stats_name="DEFAULT_01", is_in_world=False, is_rely=False, is_relative_rot=False):
        super().__init__(stats_name)
        self.is_in_world = is_in_world
        self.is_rely = is_rely
        self.is_relative_rot = is_relative_rot
        stats = getattr(stats_compose, stats_name)
        if "bps_mean" in stats:
            self.register_buffer("bps_mean", torch.tensor(stats["bps_mean"]).float(), False)
            self.register_buffer("bps_std", torch.tensor(stats["bps_std"]).float(), False)
        else:
            self.register_buffer("bps_mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("bps_std", torch.tensor(stats["std"]).float(), False)

    def encode_condition(self, inputs):
        """
        definition: {
            }
        """
        scale = inputs["scale"]
        bps = inputs["idle_bps"]
        angles = inputs["angles"]
        obj_mat = inputs["obj"]  # (B, N, 2, 4, 4)
        obj_frame0 = inputs["obj_frame0"]  # (B, 2, 4, 4)
        contact = inputs["contact"]

        contact_mask = torch.any(contact, dim=-1, keepdim=True).float()  # (B, L, 1)
        frame0_pos = matrix.get_position(obj_frame0[..., 0, :, :])  # (B, 3)

        root_pos = matrix.get_position(obj_mat[..., 0, :, :])  # (B, L, 3)
        # if not self.is_in_world:
        #     root_pos[..., 0] -= frame0_pos[..., None, 0]
        #     root_pos[..., 2] -= frame0_pos[..., None, 2]
        root_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
        root_rot = matrix.matrix_to_rotation_6d(root_rot)  # (B, L, 6)
        x = torch.cat([root_pos, root_rot, angles, contact_mask], dim=-1)  # (B, L, 11)
        first_x = x[..., :1, :]

        obj_norm = scale[..., None, :]
        bps_norm = (bps - self.bps_mean) / self.bps_std

        condition_dict = {
            "obj": obj_norm,
            "bps": bps_norm,
            "first_x": first_x,
        }

        return condition_dict

    def _encode(self, inputs):
        """
        definition: {
            }
        """
        # center = inputs["center"] # (B, 3)
        angles = inputs["angles"]
        obj_mat = inputs["obj"].clone()  # (B, N, 2, 4, 4)
        obj_frame0 = inputs["obj_frame0"]  # (B, 2, 4, 4)
        frame0_pos = matrix.get_position(obj_frame0[..., 0, :, :])  # (B, 3)
        contact = inputs["contact"]
        contact_mask = torch.any(contact, dim=-1, keepdim=True).float()  # (B, L, 1)

        # global_obj_center_pos = matrix.get_position_from(center[:, None], obj_mat[..., 0, :, :]) # (B, L, 3)
        root_pos = matrix.get_position(obj_mat[..., 0, :, :])  # (B, L, 3)
        # root_pos[..., 0] -= frame0_pos[..., None, 0]
        # root_pos[..., 2] -= frame0_pos[..., None, 2]
        if not self.is_in_world:
            first_pos = root_pos[..., :1, :].clone()
            root_pos[..., 0] -= first_pos[..., 0]
            root_pos[..., 2] -= first_pos[..., 2]
            if self.is_rely:
                root_pos[..., 1] -= first_pos[..., 1]

        root_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
        if self.is_relative_rot:
            root_rot = matrix.get_mat_BtoA(root_rot[..., :1, :, :], root_rot)
        root_rot = matrix.matrix_to_rotation_6d(root_rot)  # (B, L, 6)
        x = torch.cat([root_pos, root_rot, angles, contact_mask], dim=-1)  # (B, L, 11)

        std_ = self.std.clone()
        std_[std_ < 1e-4] = 1.0
        x_norm = (x - self.mean) / std_
        return x_norm

    def _decode(self, x_norm, inputs=None):
        """x_norm: (B, L, C)"""
        x = (x_norm * self.std) + self.mean
        root_pos = x[..., :3]
        root_rot = x[..., 3:9]
        angles = x[..., 9:10]
        contact_mask = x[..., 10:]
        root_rotmat = matrix.rotation_6d_to_matrix(root_rot)  # (B, L, 3, 3)

        if inputs is not None:
            if not self.is_in_world:
                frame0_pos = matrix.get_position(inputs["obj_frame0"][..., 0, :, :])
                first_pos = matrix.get_position(inputs["obj"][..., 0, 0, :, :]).clone()
                root_pos[..., 0] += first_pos[..., None, 0]
                root_pos[..., 2] += first_pos[..., None, 2]
                if self.is_rely:
                    root_pos[..., 1] += first_pos[..., None, 1]
            if self.is_relative_rot:
                first_rotmat = matrix.get_rotation(inputs["obj"][..., :1, 0, :, :])
                root_rotmat = matrix.get_mat_BfromA(first_rotmat, root_rotmat)

        obj_mat = matrix.get_TRS(root_rotmat, root_pos)  # (B, L, 4, 4)

        global_orient = matrix.matrix_to_axis_angle(root_rotmat)
        transl = root_pos[..., :3]

        output = {
            "obj": obj_mat,
            "angles": angles,
            "global_orient": global_orient,
            "transl": transl,
            "contact_mask": contact_mask,
        }

        return output


class GRABTrajEnDecoder(EnDecoder):
    def __init__(self, stats_name="DEFAULT_01"):
        super().__init__(stats_name)
        stats = getattr(stats_compose, stats_name)
        if "bps_mean" in stats:
            self.register_buffer("bps_mean", torch.tensor(stats["bps_mean"]).float(), False)
            self.register_buffer("bps_std", torch.tensor(stats["bps_std"]).float(), False)
        else:
            self.register_buffer("bps_mean", torch.tensor(stats["mean"]).float(), False)
            self.register_buffer("bps_std", torch.tensor(stats["std"]).float(), False)

    def encode_condition(self, inputs):
        """
        definition: {
            }
        """
        scale = inputs["scale"]
        bps = inputs["idle_bps"]
        obj_mat = inputs["obj"]  # (B, N, 2, 4, 4)
        obj_frame0 = inputs["obj_frame0"]  # (B, 2, 4, 4)
        contact = inputs["contact"]

        contact_mask = torch.any(contact, dim=-1, keepdim=True).float()  # (B, L, 1)
        frame0_pos = matrix.get_position(obj_frame0[..., 0, :, :])  # (B, 3)

        root_pos = matrix.get_position(obj_mat[..., 0, :, :])  # (B, L, 3)
        root_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
        root_rot = matrix.matrix_to_rotation_6d(root_rot)  # (B, L, 6)
        x = torch.cat([root_pos, root_rot, contact_mask], dim=-1)  # (B, L, 10)
        first_x = x[..., :1, :]

        obj_norm = scale[..., None, :]
        bps_norm = (bps - self.bps_mean) / self.bps_std

        condition_dict = {
            "obj": obj_norm,
            "bps": bps_norm,
            "first_x": first_x,
        }

        return condition_dict

    def _encode(self, inputs):
        """
        definition: {
            }
        """
        # center = inputs["center"] # (B, 3)
        obj_mat = inputs["obj"].clone()  # (B, N, 2, 4, 4)
        obj_frame0 = inputs["obj_frame0"]  # (B, 2, 4, 4)
        frame0_pos = matrix.get_position(obj_frame0[..., 0, :, :])  # (B, 3)
        contact = inputs["contact"]
        contact_mask = torch.any(contact, dim=-1, keepdim=True).float()  # (B, L, 1)

        root_pos = matrix.get_position(obj_mat[..., 0, :, :])  # (B, L, 3)
        first_pos = root_pos[..., :1, :].clone()
        root_pos[..., 0] -= first_pos[..., 0]
        root_pos[..., 2] -= first_pos[..., 2]
        root_pos[..., 1] -= first_pos[..., 1]

        root_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
        root_rot = matrix.get_mat_BtoA(root_rot[..., :1, :, :], root_rot)
        root_rot = matrix.matrix_to_rotation_6d(root_rot)  # (B, L, 6)
        x = torch.cat([root_pos, root_rot, contact_mask], dim=-1)  # (B, L, 10)

        std_ = self.std.clone()
        std_[std_ < 1e-4] = 1.0
        x_norm = (x - self.mean) / std_
        return x_norm

    def _decode(self, x_norm, inputs=None):
        """x_norm: (B, L, C)"""
        x = (x_norm * self.std) + self.mean
        root_pos = x[..., :3]
        root_rot = x[..., 3:9]
        contact_mask = x[..., 9:]
        root_rotmat = matrix.rotation_6d_to_matrix(root_rot)  # (B, L, 3, 3)

        if inputs is not None:
            frame0_pos = matrix.get_position(inputs["obj_frame0"][..., 0, :, :])
            first_pos = matrix.get_position(inputs["obj"][..., 0, 0, :, :]).clone()
            root_pos[..., 0] += first_pos[..., None, 0]
            root_pos[..., 2] += first_pos[..., None, 2]
            root_pos[..., 1] += first_pos[..., None, 1]
            first_rotmat = matrix.get_rotation(inputs["obj"][..., :1, 0, :, :])
            root_rotmat = matrix.get_mat_BfromA(first_rotmat, root_rotmat)

        obj_mat = matrix.get_TRS(root_rotmat, root_pos)  # (B, L, 4, 4)

        global_orient = matrix.matrix_to_axis_angle(root_rotmat)
        transl = root_pos[..., :3]

        output = {
            "obj": obj_mat,
            "global_orient": global_orient,
            "transl": transl,
            "contact_mask": contact_mask,
        }

        return output
