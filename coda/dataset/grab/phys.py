import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os

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
from coda.dataset.arctic.utils import *


plural_dict = {
    "offhand": "offhands",
    "pass": "passes",
    "lift": "lifts",
    "drink": "drinks from",
    "brush": "brushes with",
    "eat": "eats",
    "peel": "peels",
    "takepicture": "takes picture with",
    "see": "sees in",
    "wear": "wears",
    "play": "plays",
    "clean": "cleans",
    "browse": "browses on",
    "inspect": "inspects",
    "pour": "pours from",
    "use": "uses",
    "switchON": "switches on",
    "cook": "cooks on",
    "toast": "toasts with",
    "staple": "staples with",
    "squeeze": "squeezes",
    "set": "sets",
    "open": "opens",
    "chop": "chops with",
    "screw": "screws",
    "call": "calls on",
    "shake": "shakes",
    "fly": "flies",
    "stamp": "stamps with",
}


class BaseDataset(Dataset):
    def __init__(self, split="train", is_mirror=False, reverse_augment=False, contain_name=None, limit_size=None):
        super().__init__()
        self.split = split
        self.is_mirror = is_mirror
        self.test_sbj = ["s10"]
        self.test_obj = [
            "apple",
            "mug",
            "train",
            "elephant",
            "alarmclock",
            "pyramidsmall",
            "cylindermedium",
            "toruslarge",
        ]
        self.reverse_augment = reverse_augment
        self.contain_name = contain_name
        self.limit_size = limit_size

        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        self.motion_files = {}
        dataset_path = "./inputs/grab_neutral"
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
            obj_name = path.name.split("_")[0]
            character_obj_dict[path.parent.name].append(obj_name)

            if self.split == "train":
                if path.parent.name in self.test_sbj:
                    continue
            else:
                if path.parent.name not in self.test_sbj:
                    continue

            vid_name = path.parent.name + "_" + path.name
            # print(f"Loading {vid_name}")
            motion_data = load_arctic_data(path)
            humanoid_localmat, humanoid_globalmat = get_humanoid_data(motion_data)
            obj_localmat, obj_globalmat = get_obj_data(motion_data)
            contact = motion_data["contact"]
            beta = motion_data["humanoid"]["betas"]

            ############## IMoS #################################
            grab_original_path = "./inputs/grab_extracted/grab"
            npz_path = os.path.join(grab_original_path, path.parent.name, path.name.replace(".pt", ".npz"))
            npz = np.load(npz_path, allow_pickle=True)
            motion_intent = npz["motion_intent"]
            motion_intent = motion_intent.item()
            caption = f"The person {plural_dict[motion_intent]} the {obj_name}."
            ##################################################
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
                c = contact[k].clone()
                seq_contact.append(c)
            seq_contact = torch.stack(seq_contact, dim=-1)  # (N, 10)
            seq_contact = torch.any(seq_contact, dim=-1)  # (N,)
            last_true_index = torch.argmax(torch.flip(seq_contact.float(), dims=[0]))
            last_true_index = seq_contact.shape[0] - 1 - last_true_index
            seq_length = last_true_index

            self.motion_files[vid_name] = {
                "humanoid_localmat": humanoid_localmat,
                "humanoid_globalmat": humanoid_globalmat,
                "obj_localmat": obj_localmat,
                "obj_globalmat": obj_globalmat,
                "contact": contact,
                "beta": beta,
                "caption": caption,
                "is_mirror": False,
                "seq_length": seq_length,
            }

            max_length = max(max_length, seq_length)
            if self.split == "train":
                if self.is_mirror:
                    self.motion_files[vid_name + "_mirror"] = {
                        "humanoid_localmat": humanoid_localmat,
                        "humanoid_globalmat": humanoid_globalmat,
                        "obj_localmat": obj_localmat,
                        "obj_globalmat": obj_globalmat,
                        "contact": contact,
                        "beta": beta,
                        "caption": caption,
                        "is_mirror": True,
                        "seq_length": seq_length,
                    }

                if self.reverse_augment:
                    reverse_humanoid_globalmat = humanoid_globalmat.clone().flip(dims=[0])
                    reverse_obj_globalmat = obj_globalmat.clone().flip(dims=[0])
                    reverse_humanoid_localmat = humanoid_localmat.clone().flip(dims=[0])
                    reverse_obj_localmat = obj_localmat.clone().flip(dims=[0])
                    reverse_contact = {}
                    for k in contact.keys():
                        reverse_contact[k] = contact[k].clone().flip(dims=[0])
                    reverse_beta = beta.clone().flip(dims=[0])

                    last_true_index = torch.argmax(seq_contact.float())
                    last_true_index = seq_contact.shape[0] - 1 - last_true_index
                    seq_length = last_true_index
                    self.motion_files[vid_name + "_reverse"] = {
                        "humanoid_localmat": reverse_humanoid_localmat,
                        "humanoid_globalmat": reverse_humanoid_globalmat,
                        "obj_localmat": reverse_obj_localmat,
                        "obj_globalmat": reverse_obj_globalmat,
                        "contact": reverse_contact,
                        "beta": reverse_beta,
                        "caption": caption,
                        "is_mirror": False,
                        "seq_length": seq_length,
                    }
                    if self.is_mirror:
                        self.motion_files[vid_name + "_reverse_mirror"] = {
                            "humanoid_localmat": reverse_humanoid_localmat,
                            "humanoid_globalmat": reverse_humanoid_globalmat,
                            "obj_localmat": reverse_obj_localmat,
                            "obj_globalmat": reverse_obj_globalmat,
                            "contact": reverse_contact,
                            "beta": reverse_beta,
                            "caption": caption,
                            "is_mirror": True,
                            "seq_length": seq_length,
                        }

        print(f"max_length: {max_length}")
        print(f"Loaded {len(self.motion_files)} motion files")
        self.bps_data = torch.load("./inputs/grab_bps.pth", map_location="cpu")
        return

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []

        motion_start_id = {}
        for vid in self.motion_files:
            seq_length = self.motion_files[vid]["seq_length"]
            start_id = motion_start_id[vid] if vid in motion_start_id else 0
            seq_length = seq_length - start_id
            if seq_length < 60:  # Skip clips that are too short
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            if self.split == "test":
                for n_i in range(((seq_length - 1) // self.motion_frames) + 1):
                    start_frame = start_id + n_i * self.motion_frames
                    end_frame = min(start_id + (n_i + 1) * self.motion_frames, seq_length)
                    if end_frame - start_frame < 60:
                        continue
                    self.idx2meta.extend([(vid, start_frame, end_frame)])
                    break
            else:
                self.idx2meta.extend([(vid, start_id, -1)] * num_samples)
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples.")
        return

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def _load_data(self, idx):
        vid, start_id, end_id = self.idx2meta[idx]
        humanoid_localmat = self.motion_files[vid]["humanoid_localmat"]
        humanoid_globalmat = self.motion_files[vid]["humanoid_globalmat"]
        seq_length = self.motion_files[vid]["seq_length"]
        obj_localmat = self.motion_files[vid]["obj_localmat"]
        obj_globalmat = self.motion_files[vid]["obj_globalmat"]
        caption = self.motion_files[vid]["caption"]
        raw_len = seq_length - start_id
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
        humanoid_localmat[:, 0] = humanoid_globalmat[:, 0]
        obj_localmat[:, 0] = obj_globalmat[:, 0]

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

        select_length = end - start

        if select_length < self.motion_frames and not self.get_statistics and self.split != "test":
            humanoid_localmat = torch.cat(
                [humanoid_localmat, humanoid_localmat[-1:].repeat(self.motion_frames - select_length, 1, 1, 1)], dim=0
            )
            humanoid_globalmat = torch.cat(
                [humanoid_globalmat, humanoid_globalmat[-1:].repeat(self.motion_frames - select_length, 1, 1, 1)], dim=0
            )
            obj_localmat = torch.cat(
                [obj_localmat, obj_localmat[-1:].repeat(self.motion_frames - select_length, 1, 1, 1)], dim=0
            )
            obj_globalmat = torch.cat(
                [obj_globalmat, obj_globalmat[-1:].repeat(self.motion_frames - select_length, 1, 1, 1)], dim=0
            )
            seq_contact = torch.cat(
                [seq_contact, seq_contact[-1:].repeat(self.motion_frames - select_length, 1)], dim=0
            )
            seq_beta = torch.cat([seq_beta, seq_beta[-1:].repeat(self.motion_frames - select_length, 1)], dim=0)

        data = {
            "humanoid_localmat": humanoid_localmat,
            "humanoid_globalmat": humanoid_globalmat,
            "obj_localmat": obj_localmat,
            "obj_globalmat": obj_globalmat,
            "contact": seq_contact,
            "beta": seq_beta,
            "length": torch.tensor(select_length, dtype=torch.int32),
            "obj_frame0": obj_frame0,
            "caption": caption,
            "is_mirror": self.motion_files[vid]["is_mirror"],
        }

        if vid in self.bps_data.keys():
            basis_point = self.bps_data["basis_point"]
            if "basis_point_finger" not in self.bps_data.keys():
                basis_finger_point = basis_point
            else:
                basis_finger_point = self.bps_data["basis_point_finger"]
            bps = self.bps_data[vid]["bps"][start:end].clone()
            close_mask = self.bps_data[vid]["close_mask"][start:end].clone()
            finger_dist = self.bps_data[vid]["finger_dist"][start:end].clone()
            finger_vert_dist = self.bps_data[vid]["finger_vert_dist"][start:end].clone()
            scale = self.bps_data[vid]["scale"]
            center = self.bps_data[vid]["center"]
            data["basis_point"] = basis_point
            data["basis_finger_point"] = basis_finger_point
            data["scale"] = scale
            data["center"] = center
            data["idle_bps"] = self.bps_data[vid]["idle_bps"].clone()

            if select_length < self.motion_frames and not self.get_statistics and self.split != "test":
                data["bps"] = torch.cat([bps, bps[-1:].repeat(self.motion_frames - select_length, 1)], dim=0)
                data["finger_dist"] = torch.cat(
                    [finger_dist, finger_dist[-1:].repeat(self.motion_frames - select_length, 1, 1)], dim=0
                )
                data["close_mask"] = torch.cat(
                    [close_mask, close_mask[-1:].repeat(self.motion_frames - select_length, 1)], dim=0
                )
                data["finger_vert_dist"] = torch.cat(
                    [finger_vert_dist, finger_vert_dist[-1:].repeat(self.motion_frames - select_length, 1)], dim=0
                )
            else:
                data["bps"] = bps
                data["finger_dist"] = finger_dist
                data["close_mask"] = close_mask
                data["finger_vert_dist"] = finger_vert_dist
        return data

    def _process_data(self, data, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        # idx = 1
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data


class ObjFingerBPSDataset(BaseDataset):
    def __init__(self, is_mirror=False, get_statistics=False, **kwargs):  # DEBUG
        self.get_statistics = get_statistics
        self.dataset_id = f"GRAB_OBJFINGERBPS"
        self.motion_frames = 300 if not get_statistics else 1e6
        self.dataset_name = "GRAB_OBJFINGERBPS"
        super().__init__(is_mirror=is_mirror, reverse_augment=True, **kwargs)

    def _process_data(self, data, idx):
        length = data["length"]

        # contact = data["contact"]
        contact = data["close_mask"]  # already mirrored in bps process
        contact = contact.clone()
        # contact[...] = True
        wrist_mask = torch.stack([torch.any(contact[:, :5], dim=-1), torch.any(contact[:, 5:], dim=-1)], dim=-1)
        mask = torch.cat([wrist_mask, contact], dim=-1)  # (N, 12)
        mask_len = self.motion_frames if (self.split != "test" and not self.get_statistics) else length
        one_mask = torch.ones((mask_len, 10), dtype=torch.bool)  # always supervise contact
        vert_mask = contact.clone()
        mask = torch.cat([mask, one_mask, vert_mask], dim=-1)  # (N, 32)
        mask[length:] = False
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
        return_data["beta"] = data["beta"]
        return_data["basis_point"] = data["basis_point"]
        return_data["basis_finger_point"] = data["basis_finger_point"]
        return_data["bps"] = data["bps"].flatten(1)
        return_data["finger_dist"] = data["finger_dist"]
        return_data["finger_vert_dist"] = data["finger_vert_dist"]
        return_data["obj_frame0"] = obj_frame0
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]

        return_data["caption"] = data["caption"]
        return return_data


group_name = "train_datasets/grab"
MainStore.store(name="objfingerbps", node=builds(ObjFingerBPSDataset, split="train"), group=group_name)
MainStore.store(
    name="objfingerbps_mirror", node=builds(ObjFingerBPSDataset, split="train", is_mirror=True), group=group_name
)
MainStore.store(
    name="objfingerbps_test",
    node=builds(ObjFingerBPSDataset, split="test"),
    group="test_datasets/grab",
)


class ObjTrajPoseDataset(BaseDataset):
    def __init__(self, is_mirror=False, get_statistics=False, **kwargs):  # DEBUG
        self.get_statistics = get_statistics
        self.dataset_id = f"GRAB_OBJTRAJ"
        self.motion_frames = 300 if not get_statistics else 1e6
        self.dataset_name = "GRAB_OBJTRAJ"
        super().__init__(is_mirror=is_mirror, reverse_augment=False, **kwargs)

    def _process_data(self, data, idx):
        length = data["length"]
        contact = data["close_mask"]

        return_data = {
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx][0], "idx": idx},
            "length": length,
        }
        obj_globalmat = data["obj_globalmat"].clone()
        obj_frame0 = data["obj_frame0"].clone()

        return_data["humanoid"] = data["humanoid_globalmat"]
        return_data["obj"] = obj_globalmat
        return_data["beta"] = data["beta"]
        return_data["basis_point"] = data["basis_point"]
        return_data["basis_finger_point"] = data["basis_finger_point"]
        return_data["bps"] = data["bps"].flatten(1)
        return_data["idle_bps"] = data["idle_bps"].flatten(1)
        return_data["finger_dist"] = data["finger_dist"]
        return_data["finger_vert_dist"] = data["finger_vert_dist"]
        return_data["obj_frame0"] = obj_frame0
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]
        return_data["caption"] = data["caption"]
        return_data["contact"] = contact
        return return_data


group_name = "train_datasets/grab"
MainStore.store(name="objtraj", node=builds(ObjTrajPoseDataset, split="train"), group=group_name)
MainStore.store(
    name="objtraj_test",
    node=builds(ObjTrajPoseDataset, split="test"),
    group="test_datasets/grab",
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
        self.dataset_id = f"GRAB_HANDPOSE"
        self.motion_frames = 300 if not get_statistics else 2000
        if random1:
            self.motion_frames = 1000
        self.random1 = random1
        self.dataset_name = "GRAB_HANDPOSE"
        super().__init__(**kwargs)

    def _load_dataset(self):
        super()._load_dataset()
        s_key = "s1_cubemedium_lift.pt"
        if self.random1:
            self.motion_files = {s_key: self.motion_files[s_key]}
        else:
            self.motion_files = {k: self.motion_files[k] for k in self.motion_files.keys() if k != s_key}
        return

    def _process_data(self, data, idx):
        length = data["length"]
        mask = torch.ones((self.motion_frames, 1), dtype=torch.bool)  # always supervise full
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
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx], "idx": idx},
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


group_name = "train_datasets/grab"
MainStore.store(name="left_handpose", node=builds(HandPoseDataset, is_right=False, split="train"), group=group_name)
MainStore.store(
    name="left_handpose_test",
    node=builds(HandPoseDataset, is_right=False, split="test"),
    group="test_datasets/grab",
)
MainStore.store(name="right_handpose", node=builds(HandPoseDataset, is_right=True, split="train"), group=group_name)
MainStore.store(
    name="right_handpose_test",
    node=builds(HandPoseDataset, is_right=True, split="test"),
    group="test_datasets/grab",
)


class BodyPoseDataset(BaseDataset):
    def __init__(
        self,
        get_statistics=False,
        random1=False,  # DEBUG
        **kwargs,
    ):
        self.dataset_id = f"GRAB_BODYPOSE"
        self.motion_frames = 300 if not get_statistics else 2000
        if random1:
            self.motion_frames = 1000
        self.random1 = random1
        self.dataset_name = "GRAB_BODYPOSE"
        self.get_statistics = get_statistics
        super().__init__(**kwargs)
        self.smplx = make_smplx(type="wholebody")

    def _load_dataset(self):
        super()._load_dataset()
        s_key = "s1_cubemedium_lift.pt"
        if self.random1:
            self.motion_files = {s_key: self.motion_files[s_key]}
        else:
            self.motion_files = {k: self.motion_files[k] for k in self.motion_files.keys() if k != s_key}
        return

    def _process_data(self, data, idx):
        length = data["length"]
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
            "meta": {"dataset_id": self.dataset_id, "vid": self.idx2meta[idx], "idx": idx},
            "length": torch.tensor(length, dtype=torch.int32),
        }
        return_data["transl"] = transl
        return_data["global_orient"] = global_orient
        return_data["body_pose"] = body_pose.flatten(-2)
        return_data["skeleton"] = local_skeleton
        return_data["beta"] = beta[:, :10]

        return_data["caption"] = data["caption"]
        return return_data


group_name = "train_datasets/grab"
MainStore.store(name="body", node=builds(BodyPoseDataset), group=group_name)
MainStore.store(
    name="body",
    node=builds(BodyPoseDataset, split="test"),
    group="test_datasets/grab",
)


class WholeBodyPoseDataset(BaseDataset):
    def __init__(
        self,
        get_statistics=False,
        random1=False,  # DEBUG
        **kwargs,
    ):
        self.get_statistics = get_statistics
        self.dataset_id = f"GRAB_WHOLEBODYPOSE"
        self.motion_frames = 300
        if random1:
            self.motion_frames = 1000
        if self.get_statistics:
            self.motion_frames = 2000
        self.random1 = random1
        self.dataset_name = "GRAB_WHOLEBODYPOSE"
        super().__init__(**kwargs)
        self.smplx = make_smplx(type="wholebody")

        self.token_model = spacy.load("en_core_web_sm")
        self.max_text_len = 20
        self.unit_length = 4
        self.w_vectorizer = WordVectorizer("./inputs/checkpoints/glove", "our_vab")

    def _process_data(self, data, idx):
        length = data["length"]
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

        obj_transl = matrix.get_position(data["obj_globalmat"][:, 0])
        obj_global_orient = matrix.matrix_to_axis_angle(matrix.get_rotation(data["obj_globalmat"][:, 0]))
        return_data["obj_transl"] = obj_transl
        return_data["obj_global_orient"] = obj_global_orient

        return_data["humanoid"] = data["humanoid_globalmat"]
        return_data["obj"] = data["obj_globalmat"]
        return_data["contact"] = data["contact"]

        return_data["basis_point"] = data["basis_point"]
        return_data["basis_finger_point"] = data["basis_finger_point"]
        return_data["bps"] = data["bps"].flatten(1)
        return_data["idle_bps"] = data["idle_bps"].flatten(1)
        return_data["finger_dist"] = data["finger_dist"]
        return_data["finger_vert_dist"] = data["finger_vert_dist"]
        return_data["obj_frame0"] = data["obj_frame0"]
        return_data["scale"] = data["scale"].reshape(1)
        return_data["center"] = data["center"]
        close_mask = data["close_mask"]  # already mirrored in bps process

        vid, _, _ = self.idx2meta[idx]
        obj_name = vid.split("_")[1]
        return_data["caption"] = data["caption"]
        return_data["meta"]["obj_name"] = obj_name

        if self.split == "test":
            caption = return_data["caption"]
            # print(caption)
            caption = caption.replace("/", " ")
            tokens = self.token_model(caption)
            token_format = " ".join([f"{token.text}/{token.pos_}" for token in tokens])
            tokens = token_format.split(" ")

            # filter_tokens = []
            # for token in tokens:
            #     try:
            #         word_emb, pos_oh = self.w_vectorizer[token]
            #     except Exception as e:
            #         continue
            #     filter_tokens.append(token)
            # tokens = filter_tokens

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

        return return_data


group_name = "train_datasets/grab"
MainStore.store(name="wholebody", node=builds(WholeBodyPoseDataset, split="train"), group=group_name)
MainStore.store(
    name="wholebody_test",
    node=builds(WholeBodyPoseDataset, split="test"),
    group="test_datasets/grab",
)
