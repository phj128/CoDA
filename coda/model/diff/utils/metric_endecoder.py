import pytorch3d.structures
import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from coda.configs import MainStore, builds
from coda.utils.text2hoi.utils import get_NN, get_interior
from coda.utils.geo.augment_noisy_pose import gaussian_augment
import coda.utils.matrix as matrix
from coda.utils.pylogger import Log
from coda.dataset.arctic.utils import *
from coda.utils.geo.hmr_global import (
    get_transl_to0,
    get_transl_from_translto0,
    get_local_transl_vel,
    rollout_local_transl_vel,
)
from coda.utils.smplx_utils import make_smplx
from coda.utils.hml3d_utils.utils import detect_foot_contact
from coda.model.diff.utils.endecoder import EnDecoder
from . import stats_compose


class WholeBodyEnDecoder(EnDecoder):
    def __init__(self, stats_name="DEFAULT_01", with_obj=False):
        super().__init__(stats_name)
        self.with_obj = with_obj
        stats = getattr(stats_compose, stats_name)
        if "bps_mean" in stats:
            self.register_buffer("bps_mean", torch.tensor(stats["bps_mean"]).float(), False)
            self.register_buffer("bps_std", torch.tensor(stats["bps_std"]).float(), False)
        else:
            self.register_buffer("bps_mean", torch.tensor([0.0]).float(), False)
            self.register_buffer("bps_std", torch.tensor([1.0]).float(), False)
        if "beta_mean" in stats:
            self.register_buffer("beta_mean", torch.tensor(stats["beta_mean"]).float(), False)
            self.register_buffer("beta_std", torch.tensor(stats["beta_std"]).float(), False)
        else:
            self.register_buffer("beta_mean", torch.tensor([0.0]).float(), False)
            self.register_buffer("beta_std", torch.tensor([1.0]).float(), False)
        self.parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        self.tip_parent = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    def encode_condition(self, inputs):
        """
        definition: {
            }
        """
        obj_mat = inputs["obj"]  # (B, N, 2, 4, 4)
        beta = inputs["beta"]
        scale = inputs["scale"]  # (B,)
        angles = inputs["angles"]
        center = inputs["center"]  # (B, 3)
        objframe0 = inputs["obj_frame0"]  # (B, 2, 4, 4)
        bps = inputs["bps"]
        idle_bps = inputs["idle_bps"]  # (B, 1, 256)
        fps = 30.0

        obj_root_mat = obj_mat[..., 0, :, :]  # (B, N, 4, 4)
        obj_root_pos = matrix.get_position_from(center[..., None, :], obj_root_mat)  # (B, N, 3)
        bps_norm = (bps - self.bps_mean) / self.bps_std
        idle_bps_norm = (idle_bps - self.bps_mean) / self.bps_std

        beta_norm = (beta - self.beta_mean) / self.beta_std

        transl = inputs["transl"]  # (B, N, 3)
        global_orient = inputs["global_orient"]  # (B, N, 3)
        body_pose = inputs["body_pose"]  # (B, N, 63)
        lefthand_pose = inputs["left_hand_pose"]  # (B, N, 15)
        righthand_pose = inputs["right_hand_pose"]  # (B, N, 15)
        skeleton = inputs["skeleton"][..., :22, :]  # (B, 1, 22, 3)

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

        fps = 30.0
        global_orient_rotmat = matrix.axis_angle_to_matrix(global_orient)  # (B, N, 3, 3)
        global_orinet_rot6d = matrix.matrix_to_rotation_6d(global_orient_rotmat)  # (B, N, 6)

        pose = body_pose  # (B, N, 63)
        pose_6d = matrix.axis_angle_to_rotation_6d(pose.reshape(pose.shape[:-1] + (-1, 3)))  # (B, N, 21, 6)
        pose_6d = pose_6d.flatten(-2)  # (B, N, 132)
        lefthand_pose6d = lefthand_pose.reshape(lefthand_pose.shape[:-1] + (-1, 3))  # (B, N, 15, 3)
        lefthand_pose6d = matrix.axis_angle_to_rotation_6d(lefthand_pose6d)  # (B, N, 15, 6)
        lefthand_pose6d = lefthand_pose6d.flatten(-2)  # (B, N, 90)
        righthand_pose6d = righthand_pose.reshape(righthand_pose.shape[:-1] + (-1, 3))  # (B, N, 15, 3)
        righthand_pose6d = matrix.axis_angle_to_rotation_6d(righthand_pose6d)  # (B, N, 15, 6)
        righthand_pose6d = righthand_pose6d.flatten(-2)  # (B, N, 90)

        human_x = torch.cat(
            [
                root_transl,  # (3, )
                global_orinet_rot6d,  # (6, )
                pose_6d,  # (126, )
                lefthand_pose6d,  # (90, )
                righthand_pose6d,  # (90, )
            ],
            dim=-1,
        )  # (B, N, 315)

        obj_root_pos = matrix.get_position(obj_mat[..., 0, :, :])  # (B, L, 3)

        obj_root_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
        obj_root_rot6d = matrix.matrix_to_rotation_6d(obj_root_rot)  # (B, L, 6)
        obj_x = torch.cat(
            [
                obj_root_pos,
                obj_root_rot6d,
                angles,
            ],
            dim=-1,
        )  # (B, N, 10)

        first_x = torch.cat([human_x[..., :1, :], obj_x[..., :1, :]], dim=-1)  # (B, 1, 325)

        if self.with_obj:
            condition_dict = {
                "obj_pos": scale[..., None, :],
                "bps": idle_bps_norm,
                "beta": beta_norm,
                "first_x": first_x,
            }
        else:
            condition_dict = {
                "obj_pos": obj_root_pos,
                "bps": bps_norm,
                "beta": beta_norm,
            }

        return condition_dict

    def _encode(self, inputs):
        """
        definition: {
            }
        """
        obj_mat = inputs["obj"].clone()  # (B, N, 2, 4, 4)
        angles = inputs["angles"]
        human_global_mat = inputs["humanoid"]  # (B, N, 4, 4)
        center = inputs["center"]  # (B, 3)

        global_obj_center = matrix.get_position_from(center[..., None, :], obj_mat[..., 0, :, :])  # (B, L, 3)
        obj_root_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
        obj_arti_rot = matrix.get_rotation(obj_mat[..., 1, :, :])  # (B, L, 3, 3)
        global_obj_mat = matrix.get_TRS(obj_root_rot, global_obj_center)  # (B, L, 4, 4)
        global_obj_arti_mat = matrix.get_TRS(obj_arti_rot, global_obj_center)  # (B, L, 4, 4)

        human_pos = matrix.get_position(human_global_mat)[..., HUMANOID2SMPLX, :]  # (B, N, J, 3)
        invhuman_pos1 = matrix.get_relative_position_to(human_pos, global_obj_mat)
        invhuman_pos2 = matrix.get_relative_position_to(human_pos, global_obj_arti_mat)
        human_vec = torch.cat(
            [human_pos.flatten(-2), invhuman_pos1.flatten(-2), invhuman_pos2.flatten(-2)], dim=-1
        )  # (B, N, J*9)
        obj_vec = torch.cat([global_obj_center, angles], dim=-1)  # (B, N, 4)

        x = torch.cat([human_vec, obj_vec], dim=-1)  # (B, N, J*9+4)

        return x

    def _decode(self, x_norm, inputs=None):
        return {}


class GRABWholeBodyEnDecoder(EnDecoder):
    def __init__(self, stats_name="DEFAULT_01", with_obj=False):
        super().__init__(stats_name)
        self.with_obj = with_obj
        stats = getattr(stats_compose, stats_name)
        if "bps_mean" in stats:
            self.register_buffer("bps_mean", torch.tensor(stats["bps_mean"]).float(), False)
            self.register_buffer("bps_std", torch.tensor(stats["bps_std"]).float(), False)
        else:
            self.register_buffer("bps_mean", torch.tensor([0.0]).float(), False)
            self.register_buffer("bps_std", torch.tensor([1.0]).float(), False)
        if "beta_mean" in stats:
            self.register_buffer("beta_mean", torch.tensor(stats["beta_mean"]).float(), False)
            self.register_buffer("beta_std", torch.tensor(stats["beta_std"]).float(), False)
        else:
            self.register_buffer("beta_mean", torch.tensor([0.0]).float(), False)
            self.register_buffer("beta_std", torch.tensor([1.0]).float(), False)
        self.parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        self.tip_parent = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    def encode_condition(self, inputs):
        """
        definition: {
            }
        """
        obj_mat = inputs["obj"]  # (B, N, 2, 4, 4)
        beta = inputs["beta"]
        scale = inputs["scale"]  # (B,)
        center = inputs["center"]  # (B, 3)
        objframe0 = inputs["obj_frame0"]  # (B, 2, 4, 4)
        bps = inputs["bps"]
        idle_bps = inputs["idle_bps"]  # (B, 1, 256)
        fps = 30.0

        obj_root_mat = obj_mat[..., 0, :, :]  # (B, N, 4, 4)
        obj_root_pos = matrix.get_position_from(center[..., None, :], obj_root_mat)  # (B, N, 3)
        bps_norm = (bps - self.bps_mean) / self.bps_std
        idle_bps_norm = (idle_bps - self.bps_mean) / self.bps_std

        beta_norm = (beta - self.beta_mean) / self.beta_std

        transl = inputs["transl"]  # (B, N, 3)
        global_orient = inputs["global_orient"]  # (B, N, 3)
        body_pose = inputs["body_pose"]  # (B, N, 63)
        lefthand_pose = inputs["left_hand_pose"]  # (B, N, 15)
        righthand_pose = inputs["right_hand_pose"]  # (B, N, 15)
        skeleton = inputs["skeleton"][..., :22, :]  # (B, 1, 22, 3)

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

        fps = 30.0
        global_orient_rotmat = matrix.axis_angle_to_matrix(global_orient)  # (B, N, 3, 3)
        global_orinet_rot6d = matrix.matrix_to_rotation_6d(global_orient_rotmat)  # (B, N, 6)

        pose = body_pose  # (B, N, 63)
        pose_6d = matrix.axis_angle_to_rotation_6d(pose.reshape(pose.shape[:-1] + (-1, 3)))  # (B, N, 21, 6)
        pose_6d = pose_6d.flatten(-2)  # (B, N, 132)
        lefthand_pose6d = lefthand_pose.reshape(lefthand_pose.shape[:-1] + (-1, 3))  # (B, N, 15, 3)
        lefthand_pose6d = matrix.axis_angle_to_rotation_6d(lefthand_pose6d)  # (B, N, 15, 6)
        lefthand_pose6d = lefthand_pose6d.flatten(-2)  # (B, N, 90)
        righthand_pose6d = righthand_pose.reshape(righthand_pose.shape[:-1] + (-1, 3))  # (B, N, 15, 3)
        righthand_pose6d = matrix.axis_angle_to_rotation_6d(righthand_pose6d)  # (B, N, 15, 6)
        righthand_pose6d = righthand_pose6d.flatten(-2)  # (B, N, 90)

        human_x = torch.cat(
            [
                root_transl,  # (3, )
                global_orinet_rot6d,  # (6, )
                pose_6d,  # (126, )
                lefthand_pose6d,  # (90, )
                righthand_pose6d,  # (90, )
            ],
            dim=-1,
        )  # (B, N, 315)

        obj_root_pos = matrix.get_position(obj_mat[..., 0, :, :])  # (B, L, 3)

        obj_root_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
        obj_root_rot6d = matrix.matrix_to_rotation_6d(obj_root_rot)  # (B, L, 6)
        obj_x = torch.cat(
            [
                obj_root_pos,
                obj_root_rot6d,
            ],
            dim=-1,
        )  # (B, N, 10)

        first_x = torch.cat([human_x[..., :1, :], obj_x[..., :1, :]], dim=-1)  # (B, 1, 325)

        if self.with_obj:
            condition_dict = {
                "obj_pos": scale[..., None, :],
                "bps": idle_bps_norm,
                "beta": beta_norm,
                "first_x": first_x,
            }
        else:
            condition_dict = {
                "obj_pos": obj_root_pos,
                "bps": bps_norm,
                "beta": beta_norm,
            }

        return condition_dict

    def _encode(self, inputs):
        """
        definition: {
            }
        """
        obj_mat = inputs["obj"].clone()  # (B, N, 2, 4, 4)
        human_global_mat = inputs["humanoid"]  # (B, N, 4, 4)
        center = inputs["center"]  # (B, 3)

        global_obj_center = matrix.get_position_from(center[..., None, :], obj_mat[..., 0, :, :])  # (B, L, 3)
        obj_root_rot = matrix.get_rotation(obj_mat[..., 0, :, :])  # (B, L, 3, 3)
        obj_identity_mat = matrix.identity_mat(obj_mat[..., 0, :, :])  # (B, L, 4, 4)
        obj_root_rot = matrix.get_rotation(obj_identity_mat)  # (B, L, 3, 3)
        global_obj_mat = matrix.get_TRS(obj_root_rot, global_obj_center)  # (B, L, 4, 4)

        human_pos = matrix.get_position(human_global_mat)[..., HUMANOID2SMPLX, :]  # (B, N, J, 3)
        invhuman_pos1 = matrix.get_relative_position_to(human_pos, global_obj_mat)
        human_vec = torch.cat([human_pos.flatten(-2), invhuman_pos1.flatten(-2)], dim=-1)  # (B, N, J*6)
        obj_vec = torch.cat([global_obj_center], dim=-1)  # (B, N, 4)

        x = torch.cat([human_vec, obj_vec], dim=-1)  # (B, N, J*6+3)

        return x

    def _decode(self, x_norm, inputs=None):
        return {}
