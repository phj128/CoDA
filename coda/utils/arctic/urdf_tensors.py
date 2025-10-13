# https://github.com/zc-alexfan/arctic/blob/13ebdf1e89dcfe47b677a275cfc4280a250b160e/common/object_tensors.py
import json

import os.path as op
import sys

import numpy as np
import torch
import torch.nn as nn
import trimesh
from easydict import EasyDict
from scipy.spatial.distance import cdist

import coda.utils.arctic.thing as thing
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_apply, axis_angle_to_matrix
from coda.utils.arctic.torch_utils import pad_tensor_list
from coda.utils.arctic.xdict import xdict


URDF_OBJECTS = [
    "capsulemachine",
    "box",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "espressomachine",
    "waffleiron",
    # "scissors", # no scissors from artigrasp
    "phone",
]


class URDFTensors(nn.Module):
    def __init__(self):
        super(URDFTensors, self).__init__()
        self.obj_tensors = thing.thing2dev(construct_obj_tensors(URDF_OBJECTS), "cpu")
        self.dev = None

    def forward_7d_batch(
        self,
        angles: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
        query_names: list,
    ):
        self._sanity_check(angles, global_orient, transl, query_names)

        # store output
        out = xdict()

        # meta info
        obj_idx = np.array([self.obj_tensors["names"].index(name) for name in query_names])
        out["bottom_f"] = self.obj_tensors["bottom_f"][obj_idx]
        out["bottom_f_len"] = self.obj_tensors["bottom_f_len"][obj_idx]
        out["bottom_v_len"] = self.obj_tensors["bottom_v_len"][obj_idx]
        bottom_max_len = out["bottom_v_len"].max()
        out["bottom_v"] = self.obj_tensors["bottom_v"][obj_idx][:, :bottom_max_len]
        out["bottom_mask"] = self.obj_tensors["bottom_mask"][obj_idx][:, :bottom_max_len]

        out["top_f"] = self.obj_tensors["top_f"][obj_idx]
        out["top_f_len"] = self.obj_tensors["top_f_len"][obj_idx]
        out["top_v_len"] = self.obj_tensors["top_v_len"][obj_idx]
        top_max_len = out["top_v_len"].max()
        out["top_v"] = self.obj_tensors["top_v"][obj_idx][:, :top_max_len]
        out["top_mask"] = self.obj_tensors["top_mask"][obj_idx][:, :top_max_len]

        # articulation + global rotation
        quat_arti = axis_angle_to_quaternion(self.obj_tensors["z_axis"] * angles)
        quat_global = axis_angle_to_quaternion(global_orient.view(-1, 3))

        # collect entities to be transformed
        tf_dict = xdict()
        tf_dict["v_top"] = out["top_v"].clone()
        tf_dict["v_bottom"] = out["bottom_v"].clone()

        # articulate top parts
        for key, val in tf_dict.items():
            if "top" in key:
                val_rot = quaternion_apply(quat_arti[:, None, :], val)
                tf_dict.overwrite(key, val_rot)

        # global rotation for all
        for key, val in tf_dict.items():
            val_rot = quaternion_apply(quat_global[:, None, :], val)
            if transl is not None:
                val_rot = val_rot + transl[:, None, :]
            tf_dict.overwrite(key, val_rot)

        # prep output
        v_bottom_tensor = tf_dict["v_bottom"].clone()
        v_top_tensor = tf_dict["v_top"].clone()

        out.overwrite("out_bottom_v", v_bottom_tensor)
        out.overwrite("out_top_v", v_top_tensor)
        return out

    def forward(self, angles, global_orient, transl, query_names):
        out = self.forward_7d_batch(angles, global_orient, transl, query_names)
        return out

    def to(self, dev):
        self.obj_tensors = thing.thing2dev(self.obj_tensors, dev)
        self.dev = dev

    def _sanity_check(self, angles, global_orient, transl, query_names):
        # sanity check
        batch_size = angles.shape[0]
        assert angles.shape == (batch_size, 1)
        assert global_orient.shape == (batch_size, 3)
        if transl is not None:
            assert isinstance(transl, torch.Tensor)
            assert transl.shape == (batch_size, 3)
        assert len(query_names) == batch_size


def construct_obj_tensors(object_names):
    obj_list = []
    for k in object_names:
        object_model_p = f"./physhoi/data/assets/arctic/{k}"
        obj = construct_obj(object_model_p)
        obj_list.append(obj)

    bottom_v_list = []
    bottom_f_list = []
    top_v_list = []
    top_f_list = []
    for obj in obj_list:
        bottom_v_list.append(obj.bottom_v)
        bottom_f_list.append(obj.bottom_f)

        top_v_list.append(obj.top_v)
        top_f_list.append(obj.top_f)

    bottom_v_list, bottom_v_len_list = pad_tensor_list(bottom_v_list)
    bottom_f_list, bottom_f_len_list = pad_tensor_list(bottom_f_list)
    top_v_list, top_v_len_list = pad_tensor_list(top_v_list)
    top_f_list, top_f_len_list = pad_tensor_list(top_f_list)

    bottom_max_len = bottom_v_len_list.max()
    bottom_mask = torch.zeros(len(obj_list), bottom_max_len)
    for idx, vlen in enumerate(bottom_v_len_list):
        bottom_mask[idx, :vlen] = 1.0
    top_max_len = top_v_len_list.max()
    top_mask = torch.zeros(len(obj_list), top_max_len)
    for idx, vlen in enumerate(top_v_len_list):
        top_mask[idx, :vlen] = 1.0

    bottom_f_list, bottom_f_len_list = pad_tensor_list(bottom_f_list)
    top_f_list, top_f_len_list = pad_tensor_list(top_f_list)

    obj_tensors = {}
    obj_tensors["names"] = object_names

    obj_tensors["bottom_v"] = bottom_v_list.float()
    obj_tensors["bottom_v_len"] = bottom_v_len_list
    obj_tensors["bottom_f"] = bottom_f_list
    obj_tensors["bottom_f_len"] = bottom_f_len_list
    obj_tensors["bottom_mask"] = bottom_mask

    obj_tensors["top_v"] = top_v_list.float()
    obj_tensors["top_v_len"] = top_v_len_list
    obj_tensors["top_f"] = top_f_list
    obj_tensors["top_f_len"] = top_f_len_list
    obj_tensors["top_mask"] = top_mask

    obj_tensors["z_axis"] = torch.FloatTensor(np.array([0, 0, -1])).view(1, 3)
    return obj_tensors


def construct_obj(object_model_p):
    # load vtemplate
    bottom_mesh_p = op.join(object_model_p, "bottom_watertight_tiny.obj")
    top_mesh_p = op.join(object_model_p, "top_watertight_tiny.obj")

    bottom_mesh = trimesh.exchange.load.load_mesh(bottom_mesh_p, process=False)
    bottom_mesh_v = bottom_mesh.vertices
    bottom_mesh_f = torch.LongTensor(bottom_mesh.faces)

    top_mesh = trimesh.exchange.load.load_mesh(top_mesh_p, process=False)
    top_mesh_v = top_mesh.vertices
    top_mesh_f = torch.LongTensor(top_mesh.faces)

    vsk = object_model_p.split("/")[-1]

    obj = EasyDict()
    obj.name = vsk
    obj.obj_name = "".join([i for i in vsk if not i.isdigit()])
    obj.bottom_v = torch.FloatTensor(bottom_mesh_v)
    obj.bottom_f = bottom_mesh_f
    obj.top_v = torch.FloatTensor(top_mesh_v)
    obj.top_f = top_mesh_f

    return obj
