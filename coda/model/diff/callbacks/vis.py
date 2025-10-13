import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from pytorch_lightning.utilities import rank_zero_only
from coda.configs import MainStore, builds
from hydra.utils import instantiate

from coda.utils.comm.gather import all_gather
from coda.utils.pylogger import Log

from coda.utils.eval.eval_utils import (
    compute_camcoord_metrics,
    compute_global_metrics,
    compute_camcoord_perjoint_metrics,
    rearrange_by_mask,
    as_np_array,
)
import coda.utils.matrix as matrix
from coda.utils.eval.metric_utils import ListAggregator
from coda.model.diff.utils.motion3d_endecoder import HmlvecArcticEnDecoder
from coda.utils.smplx_utils import make_smplx
from einops import einsum, rearrange
from coda.utils.arctic.object_tensors import ObjectTensors
from coda.utils.grab.object_model import ObjectModel
from coda.utils.grab.mesh import Mesh
import coda.network.evaluator.t2m_motionenc as t2m_motionenc
import coda.network.evaluator.t2m_textenc as t2m_textenc
from coda.network.evaluator.word_vectorizer import POS_enumerator
from coda.utils.hml3d.metric import (
    euclidean_distance_matrix,
    calculate_top_k,
    calculate_diversity_np,
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_multimodality_np,
)


from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from coda.model.diff.callbacks.utils import compute_mpjpe, compute_distance_error
import imageio
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2


class VisWholeBodyPose(pl.Callback):
    def __init__(self, prefix=""):
        """
        Args:
            emdb_split: 1 to evaluate incam, 2 to evaluate global
        """
        super().__init__()
        # vid->result
        self.target_dataset_id = "ARCTIC_WHOLEBODYPOSE"
        self.metric_aggregator = {}
        self.prefix = prefix

        self.object_tensor = ObjectTensors().cuda()

        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        # if dataset_id != self.target_dataset_id:
        #     return

        vid = batch["meta"][0]["vid"]
        seq_length = batch["length"][0].item()
        text = batch["caption"][0]

        # Groundtruth
        body_pos = outputs["gt_output"]["global_pos"][0]
        lefthand_pos = outputs["gt_output"]["lefthand_global_pos"][0]
        righthand_pos = outputs["gt_output"]["righthand_global_pos"][0]

        # pred
        pred_body_pos = outputs["global_pos"][0]
        pred_lefthand_pos = outputs["lefthand_global_pos"][0]
        pred_righthand_pos = outputs["righthand_global_pos"][0]

        prefix = self.prefix
        pred_data = {
            "transl": outputs["transl"],
            "global_orient": outputs["global_orient"],
            "body_pose": outputs["body_pose"],
            "left_hand_pose": outputs["left_hand_pose"],
            "right_hand_pose": outputs["right_hand_pose"],
        }
        gt_data = {
            "transl": outputs["gt_output"]["transl"],
            "global_orient": outputs["gt_output"]["global_orient"],
            "body_pose": outputs["gt_output"]["body_pose"],
            "left_hand_pose": outputs["gt_output"]["left_hand_pose"],
            "right_hand_pose": outputs["gt_output"]["right_hand_pose"],
        }

        save_data = {
            "pred_data": pred_data,
            "gt_data": gt_data,
            "beta": batch["beta"],
            "obj_name": batch["meta"][0]["obj_name"],
            "obj_mat": batch["obj"],
            "scale": batch["scale"],
            "center": batch["center"],
            "obj_transl": batch["obj_transl"],
            "obj_global_orient": batch["obj_global_orient"],
        }
        if "obj_angles" in batch.keys():
            save_data["obj_angles"] = batch["obj_angles"]

        if "obj" in outputs.keys():
            save_data["pred_obj_transl"] = outputs["obj_transl"]
            save_data["pred_obj_global_orient"] = outputs["obj_global_orient"]
            if "obj_angles" in outputs.keys():
                save_data["pred_obj_angles"] = outputs["obj_angles"]

        if "moving_obj_verts" in outputs:
            save_data["moving_obj_verts"] = outputs["moving_obj_verts"]

        # save_dir = f"./save_dirs/{prefix}"
        # os.makedirs(save_dir, exist_ok=True)
        # torch.save(save_data, os.path.join(save_dir, f"{self.wis_i:03d}.pth"))

        obj_name = batch["meta"][0]["obj_name"]

        if "obj" in outputs.keys():
            if "obj_angles" in outputs.keys():
                arctic_obj_data = {}
                arctic_obj_data["angles"] = outputs["obj_angles"][0].cpu()
                arctic_obj_data["transl"] = outputs["obj_transl"][0].cpu()
                arctic_obj_data["global_orient"] = outputs["obj_global_orient"][0].cpu()
                with torch.no_grad():
                    obj_ns = [obj_name for _ in range(arctic_obj_data["angles"].shape[0])]
                    obj_out = self.object_tensor(**arctic_obj_data, query_names=obj_ns)
                pred_obj_verts = obj_out["v"]
                pred_obj_faces = obj_out["f"][0]
            else:
                grab_path = "./inputs/grab_extracted/tools/object_meshes/contact_meshes"
                obj_mesh = os.path.join(grab_path, f"{obj_name}.ply")
                obj_mesh = Mesh(filename=obj_mesh)
                obj_vtemp = np.array(obj_mesh.vertices)
                grab_object_tensor = ObjectModel(v_template=obj_vtemp)
                grab_object_tensor = grab_object_tensor.to(outputs["obj"].device)

                grab_obj_data = {}
                grab_obj_data["transl"] = outputs["obj_transl"][0]
                grab_obj_data["global_orient"] = outputs["obj_global_orient"][0]
                with torch.no_grad():
                    obj_out = grab_object_tensor(**grab_obj_data)
                pred_obj_verts = obj_out.vertices
                pred_obj_faces = obj_mesh.faces
        else:
            pred_obj_verts = None
            pred_obj_faces = None

        if "obj_angles" in batch.keys():
            arctic_obj_data = {}
            arctic_obj_data["angles"] = batch["obj_angles"][0].cpu()
            arctic_obj_data["transl"] = batch["obj_transl"][0].cpu()
            arctic_obj_data["global_orient"] = batch["obj_global_orient"][0].cpu()
            with torch.no_grad():
                obj_ns = [obj_name for _ in range(arctic_obj_data["angles"].shape[0])]
                obj_out = self.object_tensor(**arctic_obj_data, query_names=obj_ns)
            obj_verts = obj_out["v"]
            obj_faces = obj_out["f"][0]
            dataset_name = "arctic"
        else:
            grab_path = "./inputs/grab_extracted/tools/object_meshes/contact_meshes"
            obj_mesh = os.path.join(grab_path, f"{obj_name}.ply")
            obj_mesh = Mesh(filename=obj_mesh)
            obj_vtemp = np.array(obj_mesh.vertices)
            grab_object_tensor = ObjectModel(v_template=obj_vtemp)
            grab_object_tensor = grab_object_tensor.to(batch["obj_transl"].device)

            grab_obj_data = {}
            grab_obj_data["transl"] = batch["obj_transl"][0]
            grab_obj_data["global_orient"] = batch["obj_global_orient"][0]
            with torch.no_grad():
                obj_out = grab_object_tensor(**grab_obj_data)
            obj_verts = obj_out.vertices
            obj_faces = obj_mesh.faces
            dataset_name = "grab"

        wis3d = make_wis3d(name=f"debug_{dataset_name}_wholebodypose_{self.wis_i:03d}")
        add_motion_as_lines(body_pos, wis3d, name="gt-body", radius=0.005)
        add_motion_as_lines(pred_body_pos, wis3d, name=f"pred-{prefix}-body", radius=0.005)
        add_motion_as_lines(lefthand_pos, wis3d, name="gt-lefthand", skeleton_type="handtip", radius=0.005)
        add_motion_as_lines(
            pred_lefthand_pos, wis3d, name=f"pred-{prefix}-lefthand", skeleton_type="handtip", radius=0.005
        )
        add_motion_as_lines(righthand_pos, wis3d, name="gt-righthand", skeleton_type="handtip", radius=0.005)
        add_motion_as_lines(
            pred_righthand_pos, wis3d, name=f"pred-{prefix}-righthand", skeleton_type="handtip", radius=0.005
        )
        for i in range(obj_verts.shape[0]):
            wis3d.set_scene_id(i)
            wis3d.add_mesh(obj_verts[i], obj_faces, name=f"obj_{text}")
            if pred_obj_verts is not None:
                wis3d.add_mesh(pred_obj_verts[i], pred_obj_faces, name=f"pred-{prefix}-obj")
        if "target_finger_traj" in outputs:
            # prefix = "rope"
            for i in range(outputs["target_finger_traj"][0].shape[0]):
                wis3d.set_scene_id(i)
                wis3d.add_point_cloud(outputs["target_finger_traj"][0][i], name=prefix + "target-finger-traj")
                wis3d.add_point_cloud(outputs["target_wrist_traj"][0][i], name=prefix + "target-wrist-traj")
                if "finger_traj_global" in outputs["gt_output"]:
                    wis3d.add_point_cloud(outputs["gt_output"]["finger_traj_global"][0][i], name="gt-finger-traj")
                    wis3d.add_point_cloud(outputs["gt_output"]["wrist_traj_global"][0][i], name="gt-wrist-traj")

            # if "invtraj_outputs" in outputs:
            #     basis_point = batch["basis_finger_point"][0]  # (M, 3)
            #     obj_mat = batch["obj"][0]  # (L, 2, 4, 4)
            #     scale = batch["scale"][0]  # (1,)
            #     center = batch["center"][0]  # (3,)
            #     global_obj_center_pos = matrix.get_position_from(center[None], obj_mat[..., 0, :, :])  # (L, 3)
            #     global_obj_center_rotmat = matrix.get_rotation(obj_mat[..., 0, :, :])  # (L, 3, 3)
            #     global_obj_center_mat = matrix.get_TRS(global_obj_center_rotmat, global_obj_center_pos)  # (L, 4, 4)

            #     scaled_basis_point = basis_point * scale
            #     global_basis_point = matrix.get_position_from(
            #         scaled_basis_point[None], global_obj_center_mat
            #     )  # (L, M, 3)
            #     finger_dist = outputs["invtraj_outputs"]["finger_dist"][0]  # (L, M*J)
            #     M = finger_dist.shape[-1] // 12
            #     finger_dist = finger_dist.reshape(finger_dist.shape[0], M, 12)
            #     scaled_finger_dist = finger_dist * scale  # (L, M, 12)

            #     for i in range(global_basis_point.shape[0]):
            #         wis3d.set_scene_id(i)
            #         wis3d.add_point_cloud(global_basis_point[i], name="basis_point")
            #         for j_i in range(12):
            #             mask = finger_dist[i, :, j_i] < 0.2
            #             wis3d.add_spheres(
            #                 global_basis_point[i][mask], scaled_finger_dist[i, mask, j_i], name=f"finger_dist_{j_i}"
            #             )

        self.wis_i += 1
        return

    def on_predict_epoch_start(self, trainer, pl_module):
        self.wis_i = 0
        return

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        return


class VisCamsWholeBodyPose(pl.Callback):
    def __init__(self, prefix=""):
        """
        Args:
            emdb_split: 1 to evaluate incam, 2 to evaluate global
        """
        super().__init__()
        # vid->result
        self.metric_aggregator = {}
        self.prefix = prefix

        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    def obj_forward(self, inputs, obj_outputs):
        obj_p0_vertices = inputs["meta"][0]["obj_p0_vertices"]
        obj_p1_vertices = inputs["meta"][0]["obj_p1_vertices"]
        obj_p0_faces = inputs["meta"][0]["obj_p0_faces"]
        obj_p1_faces = inputs["meta"][0]["obj_p1_faces"]
        p1top0_root_transl = inputs["meta"][0]["p1top0_root_transl"]
        axis = inputs["meta"][0]["axis"]

        objmodel = ObjectModel(obj_p0_vertices, obj_p1_vertices, axis, p1top0_root_transl)
        obj_out = objmodel(
            global_orient=obj_outputs["global_orient"][0],
            transl=obj_outputs["transl"][0],
            articulated_angle=obj_outputs["angles"][0],
        )

        obj_verts_p0 = obj_out.vertices_p0
        obj_verts_p1 = obj_out.vertices_p1

        obj_verts = torch.cat([obj_verts_p0, obj_verts_p1], dim=-2)
        obj_p1_faces_ = obj_p1_faces + obj_verts_p0.shape[-2]
        obj_faces = torch.cat([obj_p0_faces, obj_p1_faces_], dim=-2)

        return obj_verts, obj_faces

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        # if dataset_id != self.target_dataset_id:
        #     return

        vid = batch["meta"][0]["vid"]
        seq_length = batch["length"][0].item()
        text = batch["caption"][0]

        # pred
        pred_body_pos = outputs["global_pos"][0]
        pred_lefthand_pos = outputs["lefthand_global_pos"][0]
        pred_righthand_pos = outputs["righthand_global_pos"][0]

        prefix = self.prefix
        pred_data = {
            "transl": outputs["transl"],
            "global_orient": outputs["global_orient"],
            "body_pose": outputs["body_pose"],
            "left_hand_pose": outputs["left_hand_pose"],
            "right_hand_pose": outputs["right_hand_pose"],
        }

        delta_transl = outputs["delta_transl"]

        save_data = {
            "pred_data": pred_data,
            "beta": batch["beta"],
            "obj_name": batch["meta"][0]["obj_name"],
            "obj_mat": batch["obj"],
            "scale": batch["scale"],
            "center": batch["center"],
            "obj_transl": batch["obj_transl"],
            "obj_global_orient": batch["obj_global_orient"],
            "delta_transl": delta_transl,
        }
        save_data["obj_angles"] = batch["obj_angles"]

        save_data["pred_obj_transl"] = outputs["obj_transl"]
        save_data["pred_obj_global_orient"] = outputs["obj_global_orient"]
        save_data["pred_obj_angles"] = outputs["obj_angles"]

        pred_obj_verts = outputs["obj_verts"]  # (L, M, 3)
        pred_obj_faces = outputs["obj_faces"]
        save_data["pred_obj_verts"] = pred_obj_verts
        save_data["pred_obj_faces"] = pred_obj_faces

        save_dir = f"./save_dirs/{prefix}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(save_data, os.path.join(save_dir, f"{self.wis_i:03d}.pth"))

        obj_name = batch["meta"][0]["obj_name"]

        pred_obj_verts = outputs["obj_verts"]  # (L, M, 3)
        pred_obj_faces = outputs["obj_faces"]

        gt_obj_verts = outputs["gt_obj_verts"]
        gt_obj_faces = outputs["gt_obj_faces"]

        wis3d = make_wis3d(name=f"debug_cams_wholebodypose_{self.wis_i:03d}")
        add_motion_as_lines(pred_body_pos, wis3d, name=f"pred-{prefix}-body", radius=0.005)
        add_motion_as_lines(
            pred_lefthand_pos, wis3d, name=f"pred-{prefix}-lefthand", skeleton_type="handtip", radius=0.005
        )
        add_motion_as_lines(
            pred_righthand_pos, wis3d, name=f"pred-{prefix}-righthand", skeleton_type="handtip", radius=0.005
        )
        for i in range(pred_obj_verts.shape[0]):
            wis3d.set_scene_id(i)
            wis3d.add_mesh(gt_obj_verts[i], gt_obj_faces, name=f"gt-obj_{text}_{obj_name}")
            wis3d.add_mesh(pred_obj_verts[i], pred_obj_faces, name=f"pred-{prefix}-obj")

        if "target_finger_traj" in outputs:
            for i in range(outputs["target_finger_traj"][0].shape[0]):
                wis3d.set_scene_id(i)
                wis3d.add_point_cloud(outputs["target_finger_traj"][0][i], name=prefix + "target-finger-traj")
                wis3d.add_point_cloud(outputs["target_wrist_traj"][0][i], name=prefix + "target-wrist-traj")

        self.wis_i += 1
        return

    def on_predict_epoch_start(self, trainer, pl_module):
        self.wis_i = 0
        return

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        return


class VisHot3dWholeBodyPose(pl.Callback):
    def __init__(self, prefix=""):
        """
        Args:
            emdb_split: 1 to evaluate incam, 2 to evaluate global
        """
        super().__init__()
        # vid->result
        self.metric_aggregator = {}
        self.prefix = prefix

        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        # if dataset_id != self.target_dataset_id:
        #     return

        # pred
        pred_body_pos = outputs["global_pos"][0]
        pred_lefthand_pos = outputs["lefthand_global_pos"][0]
        pred_righthand_pos = outputs["righthand_global_pos"][0]

        prefix = self.prefix
        pred_data = {
            "transl": outputs["transl"],
            "global_orient": outputs["global_orient"],
            "body_pose": outputs["body_pose"],
            "left_hand_pose": outputs["left_hand_pose"],
            "right_hand_pose": outputs["right_hand_pose"],
        }

        delta_transl = outputs["delta_transl"]
        obj_pose = batch["meta"][0]["obj_pose"]

        save_data = {
            "pred_data": pred_data,
            "beta": batch["beta"],
            "delta_transl": delta_transl,
            "obj_pose": obj_pose,
        }

        save_data["obj_verts"] = outputs["obj_verts"]
        save_data["obj_faces"] = outputs["obj_faces"]

        save_dir = f"./save_dirs/{prefix}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(save_data, os.path.join(save_dir, f"{self.wis_i:03d}.pth"))

        gt_obj_verts = outputs["obj_verts"]  # (L, M, 3)
        gt_obj_faces = outputs["obj_faces"]

        wis3d = make_wis3d(name=f"debug_hot3d_wholebodypose_{self.wis_i:03d}")
        add_motion_as_lines(pred_body_pos, wis3d, name=f"pred-{prefix}-body", radius=0.005)
        add_motion_as_lines(
            pred_lefthand_pos, wis3d, name=f"pred-{prefix}-lefthand", skeleton_type="handtip", radius=0.005
        )
        add_motion_as_lines(
            pred_righthand_pos, wis3d, name=f"pred-{prefix}-righthand", skeleton_type="handtip", radius=0.005
        )
        for k, v in gt_obj_verts.items():
            for i in range(v.shape[0]):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(v[i], gt_obj_faces[k], name=f"gt-obj_{k}")

        if "target_finger_traj" in outputs:
            for i in range(outputs["target_finger_traj"][0].shape[0]):
                wis3d.set_scene_id(i)
                wis3d.add_point_cloud(outputs["target_finger_traj"][0][i], name=prefix + "target-finger-traj")
                wis3d.add_point_cloud(outputs["target_wrist_traj"][0][i], name=prefix + "target-wrist-traj")

        self.wis_i += 1
        raise NotImplementedError
        return

    def on_predict_epoch_start(self, trainer, pl_module):
        self.wis_i = 0
        return

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        return


class VisHandObjPose(pl.Callback):
    def __init__(self, prefix=""):
        """
        Args:
            emdb_split: 1 to evaluate incam, 2 to evaluate global
        """
        super().__init__()
        # vid->result
        self.target_dataset_id = "ARCTIC_WHOLEBODYPOSE"
        self.metric_aggregator = {}
        self.prefix = prefix

        self.object_tensor = ObjectTensors().cuda()

        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        # dataset_id = batch["meta"][0]["dataset_id"]
        # if dataset_id != self.target_dataset_id:
        #     return

        vid = batch["meta"][0]["vid"]
        seq_length = batch["length"][0].item()
        mask = batch["mask"][0]

        # Groundtruth
        lefthand_pos = outputs["gt_output"]["lefthand_global_pos"][0]
        righthand_pos = outputs["gt_output"]["righthand_global_pos"][0]

        # pred
        pred_lefthand_pos = outputs["lefthand_global_pos"][0]
        pred_righthand_pos = outputs["righthand_global_pos"][0]

        prefix = self.prefix

        obj_name = batch["meta"][0]["obj_name"]

        if "obj" in outputs.keys():
            arctic_obj_data = {}
            arctic_obj_data["angles"] = outputs["obj_angles"][0].cpu()
            arctic_obj_data["transl"] = outputs["obj_transl"][0].cpu()
            arctic_obj_data["global_orient"] = outputs["obj_global_orient"][0].cpu()
            with torch.no_grad():
                obj_ns = [obj_name for _ in range(arctic_obj_data["angles"].shape[0])]
                obj_out = self.object_tensor(**arctic_obj_data, query_names=obj_ns)
            pred_obj_verts = obj_out["v"]
            pred_obj_faces = obj_out["f"][0]
        else:
            pred_obj_verts = None
            pred_obj_faces = None

        arctic_obj_data = {}
        arctic_obj_data["angles"] = batch["obj_angles"][0].cpu()
        arctic_obj_data["transl"] = batch["obj_transl"][0].cpu()
        arctic_obj_data["global_orient"] = batch["obj_global_orient"][0].cpu()
        with torch.no_grad():
            obj_ns = [obj_name for _ in range(arctic_obj_data["angles"].shape[0])]
            obj_out = self.object_tensor(**arctic_obj_data, query_names=obj_ns)
        obj_verts = obj_out["v"]
        obj_faces = obj_out["f"][0]

        wis3d = make_wis3d(name=f"debug_bps_wholebodypose_{self.wis_i:03d}")
        add_motion_as_lines(lefthand_pos, wis3d, name="gt-lefthand", skeleton_type="handtip", radius=0.005)
        add_motion_as_lines(
            pred_lefthand_pos, wis3d, name=f"pred-{prefix}-lefthand", skeleton_type="handtip", radius=0.005
        )
        add_motion_as_lines(righthand_pos, wis3d, name="gt-righthand", skeleton_type="handtip", radius=0.005)
        add_motion_as_lines(
            pred_righthand_pos, wis3d, name=f"pred-{prefix}-righthand", skeleton_type="handtip", radius=0.005
        )
        for i in range(obj_verts.shape[0]):
            wis3d.set_scene_id(i)
            wis3d.add_mesh(obj_verts[i], obj_faces, name="obj")
            if pred_obj_verts is not None:
                wis3d.add_mesh(pred_obj_verts[i], pred_obj_faces, name=f"pred-{prefix}-obj")

        self.wis_i += 1
        return

    def on_predict_epoch_start(self, trainer, pl_module):
        self.wis_i = 0
        return

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        return
