import torch
import torch.nn.functional as F
import numpy as np
import smplx
import pickle
from smplx import SMPL, SMPLX, SMPLXLayer
from coda.utils.body_model import BodyModelSMPLH, BodyModelSMPLX, BodyModelMANO
from coda.utils.body_model.smplx_lite import SmplxLiteCoco17, SmplxLiteV437Coco17, SmplxLiteSmplN24
from coda import PROJ_ROOT

# fmt: off
SMPLH_PARENTS = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                              16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                              35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50])
# fmt: on


def make_smplx(type="neu_fullpose", **kwargs):
    if type == "neu_fullpose":
        model = smplx.create(
            model_path="inputs/models/smplx/SMPLX_NEUTRAL.npz", use_pca=False, flat_hand_mean=True, **kwargs
        )
    elif type == "supermotion":
        # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": False,
        }
        bm_kwargs.update(kwargs)
        model = BodyModelSMPLX(model_path=PROJ_ROOT / "inputs/checkpoints/body_models", **bm_kwargs)
    elif type == "wholebody":
        bm_kwargs = {
            "model_type": "smplx",
            "use_pca": False,
            "gender": "neutral",
            "flat_hand_mean": True,
            "num_betas": 16,
        }
        bm_kwargs.update(kwargs)
        model = BodyModelSMPLX(model_path="inputs/checkpoints/body_models", **bm_kwargs)
    elif type == "mano":
        bm_kwargs = {
            "model_type": "mano",
            "use_pca": False,
        }
        bm_kwargs.update(kwargs)
        model = BodyModelMANO(model_path="inputs/checkpoints/body_models", **bm_kwargs)
    elif type == "mano_pca":
        bm_kwargs = {
            "model_type": "mano",
            "use_pca": True,
            "num_pca_comps": 15,
        }
        bm_kwargs.update(kwargs)
        model = BodyModelMANO(model_path="inputs/checkpoints/body_models", **bm_kwargs)
    elif type == "supermotion_EVAL3DPW":
        # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": True,
        }
        bm_kwargs.update(kwargs)
        model = BodyModelSMPLX(model_path="inputs/checkpoints/body_models", **bm_kwargs)
    elif type == "supermotion_coco17":
        # Fast but only predicts 17 joints
        model = SmplxLiteCoco17()
    elif type == "supermotion_v437coco17":
        # Predicts 437 verts and 17 joints
        model = SmplxLiteV437Coco17()
    elif type == "supermotion_smpl24":
        model = SmplxLiteSmplN24()
    elif type == "rich-smplx":
        # https://github.com/paulchhuang/rich_toolkit/blob/main/smplx2images.py
        bm_kwargs = {
            "model_type": "smplx",
            "gender": kwargs.get("gender", "male"),
            "num_pca_comps": 12,
            "flat_hand_mean": False,
            # create_expression=True, create_jaw_pose=Ture
        }
        # A /smplx folder should exist under the model_path
        model = BodyModelSMPLX(model_path="inputs/checkpoints/body_models", **bm_kwargs)
    elif type == "rich-smplh":
        bm_kwargs = {
            "model_type": "smplh",
            "gender": kwargs.get("gender", "male"),
            "use_pca": False,
            "flat_hand_mean": True,
        }
        model = BodyModelSMPLH(model_path="inputs/checkpoints/body_models", **bm_kwargs)

    elif type in ["smplx-circle", "smplx-groundlink"]:
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 16,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type == "smplx-motionx":
        layer_args = {
            "create_global_orient": False,
            "create_body_pose": False,
            "create_left_hand_pose": False,
            "create_right_hand_pose": False,
            "create_jaw_pose": False,
            "create_leye_pose": False,
            "create_reye_pose": False,
            "create_betas": False,
            "create_expression": False,
            "create_transl": False,
        }

        bm_kwargs = {
            "model_type": "smplx",
            "model_path": "inputs/checkpoints/body_models",
            "gender": "neutral",
            "use_pca": False,
            "use_face_contour": True,
            **layer_args,
        }
        model = smplx.create(**bm_kwargs)

    elif type == "smplx-samp":
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 10,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type == "smplx-bedlam":
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 11,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type in ["smplx-layer", "smplx-fit3d"]:
        # Use layer
        if type == "smplx-fit3d":
            assert (
                kwargs.get("gender") == "neutral"
            ), "smplx-fit3d use neutral model: https://github.com/sminchisescu-research/imar_vision_datasets_tools/blob/e8c8f83ffac23cc36adf8ec8d0fd1c55679484ef/util/smplx_util.py#L15C34-L15C34"

        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models/smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 10,
            "num_expression": 10,
        }
        model = SMPLXLayer(**bm_kwargs)

    elif type == "smpl":
        bm_kwargs = {
            "model_path": PROJ_ROOT / "inputs/checkpoints/body_models",
            "model_type": "smpl",
            "gender": "neutral",
            "num_betas": 10,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
        }
        bm_kwargs.update(kwargs)
        # model = SMPL(**bm_kwargs)
        model = BodyModelSMPLH(**bm_kwargs)
    elif type == "smplh":
        bm_kwargs = {
            "model_type": "smplh",
            "gender": kwargs.get("gender", "male"),
            "use_pca": False,
            "flat_hand_mean": False,
        }
        model = BodyModelSMPLH(model_path="inputs/checkpoints/body_models", **bm_kwargs)

    else:
        raise NotImplementedError

    return model


def load_parents(npz_path="models/smplx/SMPLX_NEUTRAL.npz"):
    smplx_struct = np.load("models/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
    parents = smplx_struct["kintree_table"][0].astype(np.long)
    parents[0] = -1
    return parents


def load_smpl_faces(npz_path="models/smplh/SMPLH_FEMALE.pkl"):
    with open(npz_path, "rb") as f:
        smpl_model = pickle.load(f, encoding="latin1")
    faces = np.array(smpl_model["f"].astype(np.int64))
    return faces


def decompose_fullpose(fullpose, model_type="smplx"):
    assert model_type == "smplx"

    fullpose_dict = {
        "global_orient": fullpose[..., :3],
        "body_pose": fullpose[..., 3:66],
        "jaw_pose": fullpose[..., 66:69],
        "leye_pose": fullpose[..., 69:72],
        "reye_pose": fullpose[..., 72:75],
        "left_hand_pose": fullpose[..., 75:120],
        "right_hand_pose": fullpose[..., 120:165],
    }

    return fullpose_dict


def compose_fullpose(fullpose_dict, model_type="smplx"):
    assert model_type == "smplx"
    fullpose = torch.cat(
        [
            fullpose_dict[k]
            for k in [
                "global_orient",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "left_hand_pose",
                "right_hand_pose",
            ]
        ],
        dim=-1,
    )
    return fullpose


# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]


def detect_foot_contact(motion, thre=0.002):
    """Label if the foot contact the floor.

    If the movement is large enough, it will be 1.0, otherwise 0.0.
    # TODO: Is this really ok? What if the movement is obvious? It will always be 0?

    ### Args:
    - `motion`(torch.Tensor): ((B), J=22, 3), joints position of each frame
    - `thre`(float): threshold factor to detect the foot contact the floor
    ### Returns:
    - `l_fc_labels`(torch.Tensor): ((B), 2), double foot contact labels of left foot
    - `r_fc_labels`(torch.Tensor): ((B), 2), double foot contact labels of right foot
    """
    device = motion.device
    motion_shape = list(motion.shape)
    vel_factor = torch.tensor([thre, thre]).expand(motion_shape[:-2] + [2]).to(device)

    feet_l_xyz = motion[..., 1:, fid_l, :] - motion[..., :-1, fid_l, :]  # (F-1, 2, 3)
    feet_l_l2dis = torch.norm(feet_l_xyz, dim=-1)  # (F-1, 2)
    feet_l_l2dis = torch.cat(
        [feet_l_l2dis, feet_l_l2dis[..., [-1], :].clone()], dim=-2
    )  # ((B), F, 1), padding by append the last frame again
    feet_l = (feet_l_l2dis**2 < vel_factor).float()  # (F, 2)

    feet_r_xyz = motion[..., 1:, fid_r, :] - motion[..., :-1, fid_r, :]  # (F-1, 3)
    feet_r_l2dis = torch.norm(feet_r_xyz, dim=-1)  # (F-1, 1)
    feet_r_l2dis = torch.cat(
        [feet_r_l2dis, feet_r_l2dis[..., [-1], :].clone()], dim=-2
    )  # ((B), F, 1), padding by append the last frame again
    feet_r = (feet_r_l2dis**2 < vel_factor).float()  # (F, 2)

    return feet_l, feet_r
