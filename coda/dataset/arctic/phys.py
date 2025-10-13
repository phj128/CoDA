import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import json

from tqdm import tqdm
from pathlib import Path
from coda.utils.pylogger import Log
from coda.configs import MainStore, builds
from pytorch3d.transforms import matrix_to_rotation_6d
from coda.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from coda.network.evaluator.word_vectorizer import WordVectorizer
import spacy

from coda.utils.smplx_utils import make_smplx
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines, convert_motion_as_line_mesh
from .utils import *


class BaseDataset(Dataset):
    def __init__(
        self, split="train", is_mirror=False, reverse_augment=False, is_norot=False, contain_name=None, limit_size=None
    ):
        super().__init__()
        self.is_testobj = False
        self.split = split
        self.is_mirror = is_mirror
        self.test_obj = ["box"]
        self.test_seq_dict = json.load(open("./coda/dataset/arctic/split.json"))
        self.test_seq = []
        for v in self.test_seq_dict.values():
            self.test_seq.extend(v)

        self.reverse_augment = reverse_augment
        self.is_norot = is_norot
        print(f"Use norot={is_norot}")
        self.contain_name = contain_name
        self.limit_size = limit_size

        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        self.motion_files = {}
        dataset_path = "./inputs/arctic_neutral"
        # ./inputs/arctic_neutral/s01/xx.pt
        character_obj_dict = {}
        max_length = -1
        all_path = Path(dataset_path).glob("**/*.pt")
        all_path = list(all_path)
        all_path.sort()
        for path in tqdm(all_path):
            if self.contain_name is not None:
                if self.contain_name not in path.name:
                    continue

            if path.parent.name not in character_obj_dict:
                character_obj_dict[path.parent.name] = []
            character_obj_dict[path.parent.name].append(path.name.split("_")[0])

            vid_name = path.parent.name + "_" + path.name

            if self.is_testobj:
                if self.split == "train":
                    if "box" in vid_name:
                        print(f"Filter test seq: {vid_name}")
                        continue
                else:
                    if "box" not in vid_name:
                        continue
            else:
                if self.split == "train":
                    if vid_name in self.test_seq:
                        print(f"Filter test seq: {vid_name}")
                        continue
                else:
                    if vid_name not in self.test_seq:
                        continue

            # print(f"Loading {vid_name}")
            motion_data = load_arctic_data(path)
            humanoid_localmat, humanoid_globalmat = get_humanoid_data(motion_data)
            obj_localmat, obj_globalmat = get_obj_data(motion_data)
            contact = motion_data["contact"]
            angles = motion_data["obj"]["angles"]
            beta = motion_data["humanoid"]["betas"]
            self.motion_files[vid_name] = {
                "humanoid_localmat": humanoid_localmat,
                "humanoid_globalmat": humanoid_globalmat,
                "obj_localmat": obj_localmat,
                "obj_globalmat": obj_globalmat,
                "contact": contact,
                "angles": angles,
                "beta": beta,
                "is_mirror": False,
            }
            # print(f"seq {vid_name} length: {humanoid_localmat.shape[0]}")
            max_length = max(max_length, humanoid_localmat.shape[0])
            if self.split == "train":
                if self.is_mirror:
                    self.motion_files[vid_name + "_mirror"] = {
                        "humanoid_localmat": humanoid_localmat,
                        "humanoid_globalmat": humanoid_globalmat,
                        "obj_localmat": obj_localmat,
                        "obj_globalmat": obj_globalmat,
                        "contact": contact,
                        "angles": angles,
                        "beta": beta,
                        "is_mirror": True,
                    }

                if self.reverse_augment:
                    reverse_humanoid_globalmat = humanoid_globalmat.clone().flip(dims=[0])
                    reverse_obj_globalmat = obj_globalmat.clone().flip(dims=[0])
                    reverse_humanoid_localmat = humanoid_localmat.clone().flip(dims=[0])
                    reverse_obj_localmat = obj_localmat.clone().flip(dims=[0])
                    reverse_contact = {}
                    for k in contact.keys():
                        reverse_contact[k] = contact[k].clone().flip(dims=[0])
                    reverse_angles = angles.clone().flip(dims=[0])
                    reverse_beta = beta.clone().flip(dims=[0])
                    self.motion_files[vid_name + "_reverse"] = {
                        "humanoid_localmat": reverse_humanoid_localmat,
                        "humanoid_globalmat": reverse_humanoid_globalmat,
                        "obj_localmat": reverse_obj_localmat,
                        "obj_globalmat": reverse_obj_globalmat,
                        "contact": reverse_contact,
                        "angles": reverse_angles,
                        "beta": reverse_beta,
                        "is_mirror": False,
                    }
                    if self.is_mirror:
                        self.motion_files[vid_name + "_reverse_mirror"] = {
                            "humanoid_localmat": reverse_humanoid_localmat,
                            "humanoid_globalmat": reverse_humanoid_globalmat,
                            "obj_localmat": reverse_obj_localmat,
                            "obj_globalmat": reverse_obj_globalmat,
                            "contact": reverse_contact,
                            "angles": reverse_angles,
                            "beta": reverse_beta,
                            "is_mirror": True,
                        }

        print(f"max_length: {max_length}")

        self.bps_data = torch.load("./inputs/arctic_bps.pth", map_location="cpu")
        return

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []

        motion_start_id = {}
        self.start_seq_idx = []
        for vid in self.motion_files:
            seq_length = self.motion_files[vid]["humanoid_globalmat"].shape[0]
            start_id = motion_start_id[vid] if vid in motion_start_id else 0
            seq_length = seq_length - start_id
            if seq_length < 25:  # Skip clips that are too short
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            if not self.random1 and not self.get_statistics:
                if self.contain_name is not None:
                    num_samples *= 30
                else:
                    num_samples *= 2
            if self.split == "test":
                for n_i in range(((seq_length - 1) // self.motion_frames) + 1):
                    start_frame = start_id + n_i * self.motion_frames
                    end_frame = min(start_id + (n_i + 1) * self.motion_frames, seq_length)
                    # if end_frame - start_frame < 299:
                    #     continue
                    self.idx2meta.extend([(vid, start_frame, end_frame)])
                    if start_frame == 0:
                        self.start_seq_idx.append(len(self.idx2meta) - 1)
            else:
                self.idx2meta.extend([(vid, start_id, -1)] * num_samples)
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples.")
        Log.info(f"Start seq idx: {self.start_seq_idx}")
        return

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def _load_data(self, idx):
        vid, start_id, end_id = self.idx2meta[idx]
        humanoid_localmat = self.motion_files[vid]["humanoid_localmat"]
        humanoid_globalmat = self.motion_files[vid]["humanoid_globalmat"]
        obj_localmat = self.motion_files[vid]["obj_localmat"]
        obj_globalmat = self.motion_files[vid]["obj_globalmat"]

        raw_len = humanoid_globalmat.shape[0] - start_id
        # Get {tgt_len} frames from data
        # Random select a subset with speed augmentation  [start, end)
        tgt_len = self.motion_frames
        # raw_subset_len = np.random.randint(int(tgt_len / self.l_factor), int(tgt_len * self.l_factor))
        raw_subset_len = tgt_len
        if raw_subset_len <= raw_len:
            start = np.random.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:  # interpolation will use all possible frames (results in a slow motion)
            start = 0
            end = raw_len

        if end_id != -1:
            start = start_id
            end = end_id

        humanoid_globalmat, obj_globalmat = aztoay_hoi(humanoid_globalmat, obj_globalmat)

        obj_frame0 = obj_globalmat[0].clone()

        humanoid_localmat = humanoid_localmat[start:end].clone()
        humanoid_globalmat = humanoid_globalmat[start:end].clone()
        obj_localmat = obj_localmat[start:end].clone()
        obj_globalmat = obj_globalmat[start:end].clone()

        # humanoid_globalmat, obj_globalmat = aztoay_hoi(humanoid_globalmat, obj_globalmat)

        # modify root position
        humanoid_localmat[:, 0] = humanoid_globalmat[:, 0]
        obj_localmat[:, 0] = obj_globalmat[:, 0]

        angles = self.motion_files[vid]["angles"]
        seq_angles = angles[start:end].clone()
        contact = self.motion_files[vid]["contact"]
        seq_contact = []
        for k in [
            "L_Index3",
            "L_Middle3",
            "L_Pinky3",
            "L_Ring3",
            "L_Thumb3",
            "R_Index3",
            "R_Middle3",
            "R_Pinky3",
            "R_Ring3",
            "R_Thumb3",
        ]:
            c = contact[k][start:end].clone()
            seq_contact.append(c)
        seq_contact = torch.stack(seq_contact, dim=-1)  # (N, 10)

        beta = self.motion_files[vid]["beta"]
        seq_beta = beta[start:end].clone()

        # all seqs larger than 300 frames, no need to pad

        data = {
            "humanoid_localmat": humanoid_localmat,
            "humanoid_globalmat": humanoid_globalmat,
            "obj_localmat": obj_localmat,
            "obj_globalmat": obj_globalmat,
            "contact": seq_contact,
            "angles": seq_angles,
            "beta": seq_beta,
            "obj_frame0": obj_frame0,
            "is_mirror": self.motion_files[vid]["is_mirror"],
        }

        if vid in self.bps_data.keys():
            basis_point = self.bps_data["basis_point"]
            if "basis_point_finger" not in self.bps_data.keys():
                basis_finger_point = basis_point
            else:
                basis_finger_point = self.bps_data["basis_point_finger"]
            bps_top = self.bps_data[vid]["top_bps"]
            bps_bottom = self.bps_data[vid]["bottom_bps"]
            finger_dist = self.bps_data[vid]["finger_dist"]
            finger_vert_dist = self.bps_data[vid]["finger_vert_dist"]
            scale = self.bps_data[vid]["scale"]
            center = self.bps_data[vid]["center"]
            idle_bps_top = self.bps_data[vid]["idle_top_bps"]
            idle_bps_bottom = self.bps_data[vid]["idle_bottom_bps"]
            data["basis_point"] = basis_point
            data["basis_finger_point"] = basis_finger_point
            data["bps_top"] = bps_top[start:end].clone()
            data["bps_bottom"] = bps_bottom[start:end].clone()
            data["finger_dist"] = finger_dist[start:end].clone()
            data["finger_vert_dist"] = finger_vert_dist[start:end].clone()
            data["scale"] = scale
            data["center"] = center
            data["close_mask"] = self.bps_data[vid]["close_mask"][start:end].clone()
            data["idle_bps_top"] = idle_bps_top.clone()
            data["idle_bps_bottom"] = idle_bps_bottom.clone()
        return data

    def _process_data(self, data, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        # idx = 1
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data


class ObjFingerBPSDataset(BaseDataset):
    def __init__(self, is_mirror=False, get_statistics=False, random1=False, **kwargs):  # DEBUG
        self.get_statistics = get_statistics
        self.dataset_id = f"ARCTIC_OBJFINGERBPS"
        self.motion_frames = 300 if not get_statistics else 1e6
        if random1:
            self.motion_frames = 1000
        self.random1 = random1
        self.dataset_name = "ARCTIC_OBJFINGERBPS"
        super().__init__(is_mirror=is_mirror, reverse_augment=True, **kwargs)

    def _process_data(self, data, idx):
        length = data["humanoid_globalmat"].shape[0]

        # contact = data["contact"]
        contact = data["close_mask"]  # already mirrored in bps process
        wrist_mask = torch.stack([torch.any(contact[:, :5], dim=-1), torch.any(contact[:, 5:], dim=-1)], dim=-1)
        mask = torch.cat([wrist_mask, contact], dim=-1)  # (N, 12)
        one_mask = torch.ones((length, 10), dtype=torch.bool)  # always supervise contact
        vert_mask = contact.clone()
        mask = torch.cat([mask, one_mask, vert_mask], dim=-1)  # (N, 32)
        return_data = {
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx][0], "idx": idx},
            "length": length,
            "mask": mask,
        }
        obj_globalmat = data["obj_globalmat"].clone()
        obj_frame0 = data["obj_frame0"].clone()
        if data["is_mirror"]:
            obj_globalmat = matrix.mirror_mat_yzplane(obj_globalmat)
            obj_frame0 = matrix.mirror_mat_yzplane(obj_frame0)
        return_data["humanoid"] = data["humanoid_globalmat"]
        return_data["obj"] = obj_globalmat
        return_data["contact"] = contact
        return_data["angles"] = data["angles"]
        return_data["beta"] = data["beta"]
        return_data["basis_point"] = data["basis_point"]
        return_data["basis_finger_point"] = data["basis_finger_point"]
        return_data["bps"] = torch.cat([data["bps_bottom"].flatten(1), data["bps_top"].flatten(1)], dim=-1)
        return_data["finger_dist"] = data["finger_dist"]
        return_data["finger_vert_dist"] = data["finger_vert_dist"]
        return_data["obj_frame0"] = obj_frame0
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]

        vid, _, _ = self.idx2meta[idx]
        obj_name = vid.split("_")[1]
        action = vid.split("_")[2]
        caption = f"a person {action}s a {obj_name}"
        return_data["caption"] = caption
        return return_data


group_name = "train_datasets/arctic"
MainStore.store(name="objfingerbps", node=builds(ObjFingerBPSDataset, split="train"), group=group_name)
MainStore.store(
    name="objfingerbps_mirror", node=builds(ObjFingerBPSDataset, split="train", is_mirror=True), group=group_name
)
MainStore.store(
    name="objfingerbps_norot", node=builds(ObjFingerBPSDataset, split="train", is_norot=True), group=group_name
)
MainStore.store(
    name="objfingerbps_box",
    node=builds(ObjFingerBPSDataset, contain_name="box", split="train"),
    group=group_name,
)
MainStore.store(
    name="objfingerbps_norot_box",
    node=builds(ObjFingerBPSDataset, contain_name="box", split="train", is_norot=True),
    group=group_name,
)
MainStore.store(
    name="objfingerbps_test",
    node=builds(ObjFingerBPSDataset, split="test"),
    group="test_datasets/arctic",
)
MainStore.store(
    name="objfingerbps_norot_test",
    node=builds(ObjFingerBPSDataset, split="test", is_norot=True),
    group="test_datasets/arctic",
)
MainStore.store(
    name="objfingerbps_norot_box_test",
    node=builds(ObjFingerBPSDataset, contain_name="box", split="test", is_norot=True),
    group="test_datasets/arctic",
)
MainStore.store(
    name="objfingerbps_box_test",
    node=builds(ObjFingerBPSDataset, contain_name="box", split="test"),
    group="test_datasets/arctic",
)


class ObjInvFingerDataset(BaseDataset):
    def __init__(
        self,
        get_statistics=False,
        random1=False,  # DEBUG
        **kwargs,
    ):
        self.get_statistics = get_statistics
        self.dataset_id = f"ARCTIC_OBJINVFINGER"
        self.motion_frames = 300 if not get_statistics else 1e6
        if random1:
            self.motion_frames = 1000
        self.random1 = random1
        self.dataset_name = "ARCTIC_OBJINVFINGER"
        super().__init__(**kwargs)

    def _process_data(self, data, idx):
        length = data["humanoid_globalmat"].shape[0]
        contact = data["contact"]
        wrist_mask = torch.stack([torch.any(contact[:, :5], dim=-1), torch.any(contact[:, 5:], dim=-1)], dim=-1)
        mask = torch.cat([wrist_mask, contact], dim=-1)  # (N, 12)
        mask = mask[..., None].expand(-1, -1, 3)  # (N, 12, 3)
        mask = mask.reshape(length, -1)  # (N, 36)
        one_mask = torch.ones((length, 10), dtype=torch.bool)  # always supervise contact
        mask = torch.cat([mask, one_mask], dim=-1)  # (N, 46)
        return_data = {
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx][0], "idx": idx},
            "length": length,
            "mask": mask,
        }
        return_data["humanoid"] = data["humanoid_globalmat"]
        return_data["obj"] = data["obj_globalmat"]
        return_data["contact"] = data["contact"]
        return_data["angles"] = data["angles"]
        return_data["beta"] = data["beta"]
        return_data["basis_point"] = data["basis_point"]
        return_data["basis_finger_point"] = data["basis_finger_point"]
        return_data["bps"] = torch.cat([data["bps_bottom"].flatten(1), data["bps_top"].flatten(1)], dim=-1)
        return_data["finger_dist"] = data["finger_dist"]
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]
        return return_data


group_name = "train_datasets/arctic"
MainStore.store(name="objinvfinger", node=builds(ObjInvFingerDataset, contain_name="box"), group=group_name)
MainStore.store(
    name="objinvfinger_test",
    node=builds(ObjInvFingerDataset, random1=True, contain_name="box"),
    group="test_datasets/arctic",
)


class HandPoseDataset(BaseDataset):
    def __init__(
        self,
        is_right=False,
        get_statistics=False,
        random1=False,  # DEBUG
        **kwargs,
    ):
        self.is_right = is_right
        self.get_statistics = get_statistics
        self.dataset_id = f"ARCTIC_HANDPOSE"
        self.motion_frames = 300 if not get_statistics else 20000
        if random1:
            self.motion_frames = 1000
        self.random1 = random1
        self.dataset_name = "ARCTIC_HANDPOSE"
        super().__init__(**kwargs)

    def _process_data(self, data, idx):
        length = data["humanoid_globalmat"].shape[0]
        mask = torch.ones((length, 1), dtype=torch.bool)  # always supervise full
        humanoid_localmat = data["humanoid_localmat"]
        handpose = get_handpose(humanoid_localmat, is_right=self.is_right)
        handpose_tip = get_handpose(humanoid_localmat, is_right=self.is_right, istip=True)
        if self.is_right:
            base_rotmat = matrix.get_rotation(data["humanoid_globalmat"][:, 41])
            base_pos = matrix.get_position(data["humanoid_globalmat"][:, 41])
        else:
            base_rotmat = matrix.get_rotation(data["humanoid_globalmat"][:, 17])
            base_pos = matrix.get_position(data["humanoid_globalmat"][:, 17])
        hand_localpos = matrix.get_position(handpose)
        handtip_localpos = matrix.get_position(handpose_tip)
        # hand_localpos[:, 0] = 0.0 # wrist should be zero
        return_data = {
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx][0], "idx": idx},
            "length": torch.tensor(length, dtype=torch.int32),
            # "mask": mask,
        }
        return_data["handpose"] = handpose
        return_data["beta"] = data["beta"]
        return_data["base_rotmat"] = base_rotmat
        return_data["base_pos"] = base_pos
        return_data["hand_localpos"] = hand_localpos
        return_data["handtip_localpos"] = handtip_localpos  # do not need this in training
        return_data["obj"] = data["obj_globalmat"][:, 0]
        return return_data


group_name = "train_datasets/arctic"
MainStore.store(name="left_handpose", node=builds(HandPoseDataset, is_right=False), group=group_name)
MainStore.store(
    name="left_handpose_test",
    node=builds(HandPoseDataset, is_right=False, split="test"),
    group="test_datasets/arctic",
)
MainStore.store(name="right_handpose", node=builds(HandPoseDataset, is_right=True), group=group_name)
MainStore.store(
    name="right_handpose_test",
    node=builds(HandPoseDataset, is_right=True, split="test"),
    group="test_datasets/arctic",
)


class WholeBodyPoseDataset(BaseDataset):
    def __init__(
        self,
        get_statistics=False,
        random1=False,  # DEBUG
        **kwargs,
    ):
        self.get_statistics = get_statistics
        self.dataset_id = f"ARCTIC_WHOLEBODYPOSE"
        self.motion_frames = 300
        if random1:
            self.motion_frames = 1000
        if self.get_statistics:
            self.motion_frames = 2000
        self.random1 = random1
        self.dataset_name = "ARCTIC_WHOLEBODYPOSE"
        super().__init__(**kwargs)
        self.smplx = make_smplx(type="wholebody")

        self.token_model = spacy.load("en_core_web_sm")
        self.max_text_len = 20
        self.unit_length = 4
        self.w_vectorizer = WordVectorizer("./inputs/checkpoints/glove", "our_vab")

    def _process_data(self, data, idx):
        length = data["humanoid_globalmat"].shape[0]
        beta = data["beta"]
        mask = torch.ones((length, 1), dtype=torch.bool)  # always supervise full
        humanoid_localmat = data["humanoid_localmat"]
        smplx_localmat = humanoid_localmat[:, HUMANOID2SMPLX]

        local_skeleton = self.smplx.get_local_skeleton_with_finger(beta[:1])
        root_0 = local_skeleton[:, 0]  # (1, 3)
        root_transl = matrix.get_position(smplx_localmat[:, 0])  # (N, 3)
        transl = root_transl - root_0

        global_orient_rotmat = matrix.get_rotation(smplx_localmat[:, 0])  # (N, 3, 3)
        global_orient = matrix.matrix_to_axis_angle(global_orient_rotmat)  # (N, 3)

        body_pose_rotmat = matrix.get_rotation(smplx_localmat[:, 1:22])  # (N, 21, 3, 3)
        body_pose = matrix.matrix_to_axis_angle(body_pose_rotmat)  # (N, 21, 3)
        left_hand_rotmat = matrix.get_rotation(smplx_localmat[:, 22:37])  # (N, 15, 3, 3)
        left_hand_pose = matrix.matrix_to_axis_angle(left_hand_rotmat)  # (N, 15, 3)
        right_hand_rotmat = matrix.get_rotation(smplx_localmat[:, 37:52])  # (N, 15, 3, 3)
        right_hand_pose = matrix.matrix_to_axis_angle(right_hand_rotmat)  # (N, 15, 3)

        local_pos = matrix.get_position(smplx_localmat)  # (N, 52, 3)
        local_skeleton = local_pos[:1, 1:]  # (1, 51, 3)
        local_skeleton = torch.cat([root_0[None], local_skeleton], dim=1)  # (1, 52, 3)
        return_data = {
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx][0], "idx": idx},
            "length": length,
            "mask": mask,
        }
        return_data["gt_global_pos"] = matrix.get_position(data["humanoid_globalmat"])[:, HUMANOID2SMPLX]
        return_data["transl"] = transl
        return_data["global_orient"] = global_orient
        return_data["body_pose"] = body_pose.flatten(-2)
        return_data["left_hand_pose"] = left_hand_pose.flatten(-2)
        return_data["right_hand_pose"] = right_hand_pose.flatten(-2)
        return_data["skeleton"] = local_skeleton
        return_data["beta"] = beta

        left_handpose = get_handpose(humanoid_localmat, is_right=False)
        left_base_rotmat = matrix.get_rotation(data["humanoid_globalmat"][:, 17])
        left_base_pos = matrix.get_position(data["humanoid_globalmat"][:, 17])
        left_hand_localpos = matrix.get_position(left_handpose)
        return_data["left_base_rotmat"] = left_base_rotmat
        return_data["left_base_pos"] = left_base_pos
        return_data["left_handpose"] = left_handpose
        return_data["left_hand_pose"] = matrix.matrix_to_axis_angle(matrix.get_rotation(left_handpose)).flatten(-2)
        return_data["left_hand_localpos"] = left_hand_localpos
        handpose_tip = get_handpose(humanoid_localmat, is_right=False, istip=True)
        handtip_localpos = matrix.get_position(handpose_tip)
        return_data["left_handtip_localpos"] = handtip_localpos

        right_handpose = get_handpose(humanoid_localmat, is_right=True)
        right_base_rotmat = matrix.get_rotation(data["humanoid_globalmat"][:, 41])
        right_base_pos = matrix.get_position(data["humanoid_globalmat"][:, 41])
        righthand_localpos = matrix.get_position(right_handpose)
        return_data["right_base_rotmat"] = right_base_rotmat
        return_data["right_base_pos"] = right_base_pos
        return_data["right_handpose"] = right_handpose
        return_data["right_hand_pose"] = matrix.matrix_to_axis_angle(matrix.get_rotation(right_handpose)).flatten(-2)
        return_data["right_hand_localpos"] = righthand_localpos
        handpose_tip = get_handpose(humanoid_localmat, is_right=True, istip=True)
        handtip_localpos = matrix.get_position(handpose_tip)
        return_data["right_handtip_localpos"] = handtip_localpos

        return_data["obj_angles"] = data["angles"]
        obj_transl = matrix.get_position(data["obj_globalmat"][:, 0])
        obj_global_orient = matrix.matrix_to_axis_angle(matrix.get_rotation(data["obj_globalmat"][:, 0]))
        return_data["obj_transl"] = obj_transl
        return_data["obj_global_orient"] = obj_global_orient

        return_data["humanoid"] = data["humanoid_globalmat"]
        return_data["obj"] = data["obj_globalmat"]
        return_data["contact"] = data["contact"]
        return_data["angles"] = data["angles"]

        return_data["basis_point"] = data["basis_point"]
        return_data["basis_finger_point"] = data["basis_finger_point"]
        return_data["bps"] = torch.cat([data["bps_bottom"].flatten(1), data["bps_top"].flatten(1)], dim=-1)
        return_data["finger_dist"] = data["finger_dist"]
        return_data["finger_vert_dist"] = data["finger_vert_dist"]
        return_data["obj_frame0"] = data["obj_frame0"]
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]

        return_data["idle_bps"] = torch.cat(
            [data["idle_bps_bottom"].flatten(1), data["idle_bps_top"].flatten(1)], dim=-1
        )

        vid, start_frame, _ = self.idx2meta[idx]
        obj_name = vid.split("_")[1]
        action = vid.split("_")[2]
        caption = f"a person {action}s a {obj_name}"
        # caption = ""
        return_data["caption"] = caption
        return_data["meta"]["obj_name"] = obj_name

        if self.split == "test":
            caption = caption.replace("/", " ")
            tokens = self.token_model(caption)
            token_format = " ".join([f"{token.text}/{token.pos_}" for token in tokens])
            tokens = token_format.split(" ")

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens)
                tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[: self.max_text_len]
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            return_data["pos_onehot"] = pos_one_hots.astype(np.float32)
            return_data["word_embs"] = word_embeddings.astype(np.float32)
            return_data["text_len"] = sent_len

            if start_frame == 0:
                return_data["is_start"] = True
            else:
                return_data["is_start"] = False

        return return_data


group_name = "train_datasets/arctic"
MainStore.store(name="wholebody", node=builds(WholeBodyPoseDataset, split="train"), group=group_name)
MainStore.store(
    name="wholebody_test",
    node=builds(WholeBodyPoseDataset, split="test"),
    group="test_datasets/arctic",
)
MainStore.store(
    name="wholebody_box_test",
    node=builds(WholeBodyPoseDataset, split="test", contain_name="box"),
    group="test_datasets/arctic",
)


class BodyPoseDataset(BaseDataset):
    def __init__(self, get_statistics=False, random1=False, **kwargs):  # DEBUG
        self.dataset_id = f"ARCTIC_BODYPOSE"
        self.motion_frames = 300 if not get_statistics else 2000
        if random1:
            self.motion_frames = 1000
        self.random1 = random1
        self.dataset_name = "ARCTIC_BODYPOSE"
        self.get_statistics = get_statistics
        super().__init__(**kwargs)
        self.smplx = make_smplx(type="wholebody")

    def _load_dataset(self):
        super()._load_dataset()
        # s_key = "s01_box_use_01.pt"
        # # s_key = "s01_box_use_02.pt"
        # if self.random1:
        #     self.motion_files = {s_key: self.motion_files[s_key]}
        # else:
        #     self.motion_files = {k: self.motion_files[k] for k in self.motion_files.keys() if k != s_key}
        return

    def _process_data(self, data, idx):
        length = data["humanoid_globalmat"].shape[0]
        beta = data["beta"]
        humanoid_localmat = data["humanoid_localmat"]
        smplx_localmat = humanoid_localmat[:, HUMANOID2SMPLX]

        local_skeleton = self.smplx.get_local_skeleton_with_finger(beta[:1])
        root_0 = local_skeleton[:, 0]  # (1, 3)
        root_transl = matrix.get_position(smplx_localmat[:, 0])  # (N, 3)
        transl = root_transl - root_0

        global_orient_rotmat = matrix.get_rotation(smplx_localmat[:, 0])  # (N, 3, 3)
        global_orient = matrix.matrix_to_axis_angle(global_orient_rotmat)  # (N, 3)

        body_pose_rotmat = matrix.get_rotation(smplx_localmat[:, 1:22])  # (N, 21, 3, 3)
        body_pose = matrix.matrix_to_axis_angle(body_pose_rotmat)  # (N, 21, 3)

        local_pos = matrix.get_position(smplx_localmat)  # (N, 52, 3)
        local_skeleton = local_pos[:1, 1:]  # (1, 51, 3)
        local_skeleton = torch.cat([root_0[None], local_skeleton], dim=1)  # (1, 52, 3)
        local_skeleton = local_skeleton[:, :22, :]

        return_data = {
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx][0], "idx": idx},
            "length": torch.tensor(length, dtype=torch.int32),
        }
        return_data["transl"] = transl
        return_data["global_orient"] = global_orient
        return_data["body_pose"] = body_pose.flatten(-2)
        return_data["skeleton"] = local_skeleton
        return_data["beta"] = beta[:, :10]

        vid, _, _ = self.idx2meta[idx]
        obj_name = vid.split("_")[1]
        action = vid.split("_")[2]
        caption = f"a person {action}s a {obj_name}"
        return_data["caption"] = caption
        return return_data


group_name = "train_datasets/arctic"
MainStore.store(name="body", node=builds(BodyPoseDataset), group=group_name)
MainStore.store(
    name="body",
    node=builds(BodyPoseDataset, split="test"),
    group="test_datasets/arctic",
)


class ObjTrajPoseDataset(BaseDataset):
    def __init__(self, get_statistics=False, random1=False, **kwargs):  # DEBUG
        self.dataset_id = f"ARCTIC_OBJTRAJ"
        self.motion_frames = 300 if not get_statistics else 2000
        if random1:
            self.motion_frames = 1000
        self.random1 = random1
        self.dataset_name = "ARCTIC_OBJTRAJ"
        self.get_statistics = get_statistics
        super().__init__(**kwargs)
        self.smplx = make_smplx(type="wholebody")

    def _process_data(self, data, idx):
        length = data["humanoid_globalmat"].shape[0]
        contact = data["close_mask"]

        vid, start_id, end_id = self.idx2meta[idx]

        # contact = data["contact"]
        return_data = {
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx][0], "idx": idx},
            "length": length,
        }
        return_data["humanoid"] = data["humanoid_globalmat"]
        return_data["obj"] = data["obj_globalmat"]
        return_data["angles"] = data["angles"]
        return_data["beta"] = data["beta"]
        return_data["basis_point"] = data["basis_point"]
        return_data["basis_finger_point"] = data["basis_finger_point"]
        return_data["bps"] = torch.cat([data["bps_bottom"].flatten(1), data["bps_top"].flatten(1)], dim=-1)
        return_data["idle_bps"] = torch.cat(
            [data["idle_bps_bottom"].flatten(1), data["idle_bps_top"].flatten(1)], dim=-1
        )
        return_data["finger_dist"] = data["finger_dist"]
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]
        return_data["obj_frame0"] = data["obj_frame0"]
        return_data["contact"] = contact

        vid, _, _ = self.idx2meta[idx]
        obj_name = vid.split("_")[1]
        action = vid.split("_")[2]
        caption = f"a person {action}s a {obj_name}"
        return_data["caption"] = caption
        return return_data


group_name = "train_datasets/arctic"
MainStore.store(name="objtraj", node=builds(ObjTrajPoseDataset), group=group_name)
MainStore.store(
    name="objtraj_test",
    node=builds(ObjTrajPoseDataset, split="test"),
    group="test_datasets/arctic",
)


class ObjTrajPoseFingerDataset(BaseDataset):
    def __init__(self, get_statistics=False, random1=False, **kwargs):  # DEBUG
        self.dataset_id = f"ARCTIC_OBJTRAJ"
        self.motion_frames = 300 if not get_statistics else 2000
        if random1:
            self.motion_frames = 1000
        self.random1 = random1
        self.dataset_name = "ARCTIC_OBJTRAJ"
        self.get_statistics = get_statistics
        super().__init__(**kwargs)
        self.smplx = make_smplx(type="wholebody")

    def _process_data(self, data, idx):

        length = data["humanoid_globalmat"].shape[0]
        contact = data["close_mask"]
        wrist_mask = torch.stack([torch.any(contact[:, :5], dim=-1), torch.any(contact[:, 5:], dim=-1)], dim=-1)
        mask = torch.cat([wrist_mask, contact], dim=-1)  # (N, 12)
        one_mask = torch.ones((length, 10), dtype=torch.bool)  # always supervise contact
        vert_mask = contact.clone()
        mask = torch.cat([mask, one_mask, vert_mask, one_mask], dim=-1)  # (N, 42)

        return_data = {
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx][0], "idx": idx},
            "length": length,
            "mask": mask,
        }
        return_data["humanoid"] = data["humanoid_globalmat"]
        return_data["obj"] = data["obj_globalmat"]
        return_data["angles"] = data["angles"]
        return_data["beta"] = data["beta"]
        return_data["basis_point"] = data["basis_point"]
        return_data["basis_finger_point"] = data["basis_finger_point"]
        return_data["bps"] = torch.cat([data["bps_bottom"].flatten(1), data["bps_top"].flatten(1)], dim=-1)
        return_data["idle_bps"] = torch.cat(
            [data["idle_bps_bottom"].flatten(1), data["idle_bps_top"].flatten(1)], dim=-1
        )
        return_data["finger_dist"] = data["finger_dist"]
        return_data["finger_vert_dist"] = data["finger_vert_dist"]
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]
        return_data["obj_frame0"] = data["obj_frame0"]
        return_data["contact"] = contact

        vid, start_id, end_id = self.idx2meta[idx]
        vid, _, _ = self.idx2meta[idx]
        obj_name = vid.split("_")[1]
        action = vid.split("_")[2]
        caption = f"a person {action}s a {obj_name}"
        return_data["caption"] = caption
        return return_data


group_name = "train_datasets/arctic"
MainStore.store(name="objtrajfinger", node=builds(ObjTrajPoseFingerDataset), group=group_name)
MainStore.store(
    name="objtrajfinger_test",
    node=builds(ObjTrajPoseFingerDataset, split="test"),
    group="test_datasets/arctic",
)
