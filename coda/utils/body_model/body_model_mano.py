import torch
import torch.nn as nn
import smplx


class BodyModelMANO(nn.Module):
    """Support Batch inference"""

    def __init__(self, model_path, with_fingertip=False, **kwargs):
        super().__init__()
        self.with_fingertip = with_fingertip
        # enable flexible batchsize, handle missing variable at forward()
        self.bm = smplx.create(model_path=model_path, **kwargs)
        self.faces = self.bm.faces
        self.hand_pose_dim = self.bm.num_pca_comps if self.bm.use_pca else 3 * self.bm.NUM_HAND_JOINTS
        self.fingertip_index = self.bm.vertex_joint_selector.extra_joints_idxs

        shapedirs = self.bm.shapedirs  # (V, 3, 10)
        J_regressor = self.bm.J_regressor[:16, :]  # (16, V)
        v_template = self.bm.v_template  # (V, 3)
        J_template = J_regressor @ v_template  # (16, 3)
        J_shapedirs = torch.einsum("jv, vcd -> jcd", J_regressor, shapedirs)  # (16, 3, 10)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)

        self.parents_tip = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9]

    def forward(self, betas=None, global_orient=None, hand_pose=None, transl=None, **kwargs):

        device, dtype = self.bm.shapedirs.device, self.bm.shapedirs.dtype

        model_vars = [betas, global_orient, hand_pose, transl]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if hand_pose is None:
            hand_pose = (
                torch.zeros(3 * self.bm.NUM_HAND_JOINTS, device=device, dtype=dtype)[None]
                .expand(batch_size, -1)
                .contiguous()
            )
        if betas is None:
            betas = torch.zeros([batch_size, self.bm.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        bm_out = self.bm(betas=betas, global_orient=global_orient, hand_pose=hand_pose, transl=transl, **kwargs)
        if self.with_fingertip:
            fingertips = bm_out.vertices[:, self.fingertip_index]
            bm_out.fingertips = fingertips
            # parents = np.concatenate([parents, np.array([15, 3, 6, 12, 9])])
            # ** original MANO joint order (right hand)
            #                16-15-14-13-\  thumb
            #                             \
            #          17 --3 --2 --1------0  index
            #        18 --6 --5 --4-------/  middle
            #        19 -12 -11 --10-----/  ring
            #          20 --9 --8 --7---/  pinky
            # **
        return bm_out

    def get_skeleton(self, betas):
        """betas: (*, 10) -> skeleton_beta: (*, 16, 3)"""
        skeleton_beta = self.J_template + torch.einsum("...d, jcd -> ...jc", betas, self.J_shapedirs)  # (16, 3)
        return skeleton_beta

    def get_local_skeleton(self, betas):
        """betas: (*, 10) -> skeleton_beta: (*, 16, 3)"""
        skeleton = self.get_skeleton(betas)
        skeleton_parent = skeleton[:, self.bm.parents[:16]]
        skeleton_local = skeleton - skeleton_parent
        skeleton_local[:, 0] = skeleton[:, 0]
        return skeleton_local

    def get_skeleton_with_fingertip(self, betas):
        bm_out = self.bm(betas=betas)
        fingertips = bm_out.vertices[:, self.fingertip_index]
        skeleton = self.get_skeleton(betas)
        tip_skeleton = torch.cat([skeleton, fingertips], dim=1)
        return tip_skeleton

    def get_local_skeleton_with_fingertip(self, betas):
        tip_skeleton = self.get_skeleton_with_fingertip(betas)
        tip_skeleton_parent = tip_skeleton[:, self.parents_tip]
        tip_skeleton_local = tip_skeleton - tip_skeleton_parent
        tip_skeleton_local[:, 0] = tip_skeleton[:, 0]
        return tip_skeleton_local
