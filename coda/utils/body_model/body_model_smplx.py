import torch
import torch.nn as nn
import smplx

kwargs_disable_member_var = {
    "create_body_pose": False,
    "create_betas": False,
    "create_global_orient": False,
    "create_transl": False,
    "create_left_hand_pose": False,
    "create_right_hand_pose": False,
    "create_expression": False,
    "create_jaw_pose": False,
    "create_leye_pose": False,
    "create_reye_pose": False,
}


SMPLXTIP_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    # "jaw",
    # "left_eye_smplhf",
    # "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]


# SMPLX in HUMANOID order
SMPLXHUMANOID_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_index",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_middle",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_pinky",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_ring",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_index",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_middle",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_pinky",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_ring",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb",
]

SMPLXTIP2HUMANOID = [SMPLXTIP_JOINT_NAMES.index(name) for name in SMPLXHUMANOID_JOINT_NAMES]


class BodyModelSMPLX(nn.Module):
    """Support Batch inference"""

    def __init__(self, model_path, **kwargs):
        super().__init__()
        # enable flexible batchsize, handle missing variable at forward()
        kwargs.update(kwargs_disable_member_var)
        self.bm = smplx.create(model_path=model_path, **kwargs)
        self.faces = self.bm.faces
        self.hand_pose_dim = self.bm.num_pca_comps if self.bm.use_pca else 3 * self.bm.NUM_HAND_JOINTS

        # For fast computing of skeleton under beta
        shapedirs = self.bm.shapedirs  # (V, 3, 10)
        J_regressor = self.bm.J_regressor[:22, :]  # (22, V)
        v_template = self.bm.v_template  # (V, 3)
        J_template = J_regressor @ v_template  # (22, 3)
        J_shapedirs = torch.einsum("jv, vcd -> jcd", J_regressor, shapedirs)  # (22, 3, 10)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)

        # for smplx
        shapedirs = self.bm.shapedirs  # (V, 3, 10)
        J_regressor = self.bm.J_regressor[:55, :]  # (22, V)
        v_template = self.bm.v_template  # (V, 3)
        J_template = J_regressor @ v_template  # (22, 3)
        J_shapedirs = torch.einsum("jv, vcd -> jcd", J_regressor, shapedirs)  # (22, 3, 10)
        self.register_buffer("smplx_J_template", J_template, False)
        self.register_buffer("smplx_J_shapedirs", J_shapedirs, False)

    def forward(
        self,
        betas=None,
        global_orient=None,
        transl=None,
        body_pose=None,
        left_hand_pose=None,
        right_hand_pose=None,
        expression=None,
        jaw_pose=None,
        leye_pose=None,
        reye_pose=None,
        **kwargs
    ):

        device, dtype = self.bm.shapedirs.device, self.bm.shapedirs.dtype

        model_vars = [
            betas,
            global_orient,
            body_pose,
            transl,
            expression,
            left_hand_pose,
            right_hand_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
        ]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if body_pose is None:
            body_pose = (
                torch.zeros(3 * self.bm.NUM_BODY_JOINTS, device=device, dtype=dtype)[None]
                .expand(batch_size, -1)
                .contiguous()
            )
        if left_hand_pose is None:
            left_hand_pose = (
                torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[None].expand(batch_size, -1).contiguous()
            )
        if right_hand_pose is None:
            right_hand_pose = (
                torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[None].expand(batch_size, -1).contiguous()
            )
        if jaw_pose is None:
            jaw_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if leye_pose is None:
            leye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if reye_pose is None:
            reye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.bm.num_expression_coeffs], dtype=dtype, device=device)
        if betas is None:
            betas = torch.zeros([batch_size, self.bm.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        bm_out = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            **kwargs
        )

        return bm_out

    def get_skeleton(self, betas):
        """betas: (*, 10) -> skeleton_beta: (*, 22, 3)"""
        skeleton_beta = self.J_template + torch.einsum("...d, jcd -> ...jc", betas, self.J_shapedirs)  # (22, 3)
        return skeleton_beta

    def get_local_skeleton(self, betas):
        """betas: (*, 10) -> skeleton_beta: (*, 22, 3)"""
        skeleton = self.get_skeleton(betas)
        skeleton_parent = skeleton[:, self.bm.parents[:22]]
        skeleton_local = skeleton - skeleton_parent
        skeleton_local[:, 0] = skeleton[:, 0]
        return skeleton_local

    def get_skeleton_with_finger(self, betas):
        """betas: (*, 16) -> skeleton_beta: (*, 52, 3)"""
        skeleton = self.smplx_J_template + torch.einsum("...d, jcd -> ...jc", betas, self.smplx_J_shapedirs)  # (52, 3)
        return skeleton

    def get_local_skeleton_with_finger(self, betas):
        """betas: (*, 16) -> skeleton_beta: (*, 52, 3)"""
        skeleton = self.get_skeleton_with_finger(betas)
        skeleton_parent = skeleton[:, self.bm.parents[:55]]
        skeleton_local = skeleton - skeleton_parent
        skeleton_local[:, 0] = skeleton[:, 0]
        skeleton_local = torch.cat([skeleton_local[:, :22], skeleton_local[:, 25:55]], dim=1)
        return skeleton_local

    def get_skeleton_with_fingertip(self, betas, is_physics=False):
        """betas: (*, 16) -> skeleton_beta: (*, 62, 3)"""
        skeleton = self.get_skeleton_with_finger(betas)
        smplx_out = self(betas=betas, return_verts=False)
        # left to right
        # thumb, index, middle, ring, pinky
        fingertip_ind = [66, 67, 68, 69, 70, 71, 72, 73, 74, 75]
        fingertip_pos = smplx_out.joints[:, fingertip_ind, :]
        skeleton = torch.cat([skeleton, fingertip_pos], dim=-2)
        if is_physics:
            skeleton = torch.cat([skeleton[:, :22], skeleton[:, 25:]], dim=1)
            skeleton = skeleton[:, SMPLXTIP2HUMANOID, :]
        return skeleton

    def get_local_skeleton_with_fingertip(self, betas, is_physics=False):
        skeleton = self.get_skeleton_with_fingertip(betas, is_physics=False)
        parent = torch.cat(
            [self.bm.parents, torch.tensor([39, 27, 30, 36, 33, 54, 42, 45, 51, 48], dtype=torch.int64)],
            dim=-1,
        )
        skeleton_parent = skeleton[:, parent, :]
        skeleton_local = skeleton - skeleton_parent
        skeleton_local[:, 0] = skeleton[:, 0]
        if is_physics:
            skeleton_local = torch.cat([skeleton_local[:, :22], skeleton_local[:, 25:]], dim=1)
            skeleton_local = skeleton_local[:, SMPLXTIP2HUMANOID, :]
        return skeleton_local

    def forward_bfc(self, **kwargs):
        """Wrap (B, F, C) to (B*F, C) and unwrap (B*F, C) to (B, F, C)"""
        for k in kwargs:
            assert len(kwargs[k].shape) == 3
        B, F = kwargs["body_pose"].shape[:2]
        smplx_out = self.forward(**{k: v.reshape(B * F, -1) for k, v in kwargs.items()})
        smplx_out.vertices = smplx_out.vertices.reshape(B, F, -1, 3)
        smplx_out.joints = smplx_out.joints.reshape(B, F, -1, 3)
        return smplx_out
