import torch
import torch.nn.functional as F
import json
from torch.utils.data import Dataset
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import os
import codecs as cs
from coda.utils.pylogger import Log
from coda.configs import MainStore, builds
from pytorch3d.transforms import matrix_to_rotation_6d
from coda.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from coda.network.evaluator.word_vectorizer import WordVectorizer

from coda.utils.smplx_utils import make_smplx
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines, convert_motion_as_line_mesh
from coda.dataset.arctic.utils import *


class AmassDataset(Dataset):
    def __init__(self, split, get_statistics=False, limit_size=None):
        super().__init__()
        self.split = split
        self.get_statistics = get_statistics
        # limit_size = 64
        self.limit_size = limit_size
        self.target_fps = 30
        self.min_motion_len = 2 * self.target_fps
        self.max_motion_len = 10 * self.target_fps

        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        self.motion_files = torch.load("./inputs/amass/smplx.pth")
        return

    def _get_idx2meta(self):
        self.idx2meta = []
        for k, v in self.motion_files.items():
            length = v["pose"].shape[0]
            if length < 30:
                continue
            NUM = (length - 1) // self.max_motion_len + 1
            for _ in range(NUM):
                self.idx2meta.append(k)
        return

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def _load_data(self, idx):
        seq_name = self.idx2meta[idx]
        data = self.motion_files[seq_name]
        return_data = {}
        return_data.update(data)
        return return_data

    def _process_data(self, data, idx):
        pose = data["pose"]
        length = pose.shape[0]
        if length <= self.max_motion_len:
            start_frame = 0
            end_frame = length
        else:
            start_frame = np.random.randint(0, length - self.max_motion_len + 1)
            end_frame = start_frame + self.max_motion_len

        pose = pose[int(start_frame) : int(end_frame)]
        N = pose.shape[0]
        transl = pose[:, :3]
        global_orient = pose[:, 3:6]
        body_pose = pose[:, 6:]
        skeleton = data["skeleton"]  # (1, 22, 3)

        caption = ""

        beta = data["beta"].repeat(N, 1)  # (N, 10)

        if N < self.max_motion_len and not self.get_statistics:
            # pad last frame
            transl = torch.cat([transl, transl[-1:].repeat(self.max_motion_len - N, 1)], dim=0)
            global_orient = torch.cat([global_orient, global_orient[-1:].repeat(self.max_motion_len - N, 1)], dim=0)
            body_pose = torch.cat([body_pose, body_pose[-1:].repeat(self.max_motion_len - N, 1)], dim=0)
            beta = torch.cat([beta, beta[-1:].repeat(self.max_motion_len - N, 1)], dim=0)

        return_data = {
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "beta": beta,
            "skeleton": skeleton,
            "caption": caption,
            "length": N,
            "meta": {
                "dataset_id": "AMASS",
                "vid": self.idx2meta[idx],
                "idx": idx,
            },
        }

        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data


group_name = "train_datasets/amass"
MainStore.store(name="amass", node=builds(AmassDataset, split="train"), group=group_name)
MainStore.store(
    name="amass_test",
    node=builds(AmassDataset, split="test"),
    group="test_datasets/amass",
)


class AmassAugDataset(AmassDataset):
    def _load_dataset(self):
        super()._load_dataset()
        self.aug_motion_files = {}
        self._load_arctic_dataset()
        self._load_grab_dataset()
        self.aug_motion_keys = list(self.aug_motion_files.keys())

    def _load_arctic_dataset(self):
        self.arctic_test_seq_dict = json.load(open("./coda/dataset/arctic/split.json"))
        self.arctic_test_seq = []
        for v in self.arctic_test_seq_dict.values():
            self.arctic_test_seq.extend(v)

        dataset_path = "./inputs/arctic_neutral"
        # ./inputs/arctic_neutral/s01/xx.pt
        character_obj_dict = {}
        max_length = -1
        all_path = Path(dataset_path).glob("**/*.pt")
        all_path = list(all_path)
        all_path.sort()
        N = 0
        for path in tqdm(all_path):
            if path.parent.name not in character_obj_dict:
                character_obj_dict[path.parent.name] = []
            character_obj_dict[path.parent.name].append(path.name.split("_")[0])

            vid_name = path.parent.name + "_" + path.name

            # if self.split == "train":
            #     if path.parent.name in self.test_sbj:
            #         continue
            # else:
            #     if path.parent.name not in self.test_sbj:
            #         continue

            if self.split == "train":
                if vid_name in self.arctic_test_seq:
                    print(f"Filter test seq: {vid_name}")
                    continue
            else:
                if vid_name not in self.arctic_test_seq:
                    continue

            # print(f"Loading {vid_name}")
            motion_data = load_arctic_data(path)
            humanoid_localmat, humanoid_globalmat = get_humanoid_data(motion_data)
            obj_localmat, obj_globalmat = get_obj_data(motion_data)
            beta = motion_data["humanoid"]["betas"]
            smplx_localmat = humanoid_localmat[:, HUMANOID2SMPLX]
            wrist_localmat = smplx_localmat[:, [20, 21]]
            wrist_rotmat = matrix.get_rotation(wrist_localmat)
            wrist_aa = matrix.matrix_to_axis_angle(wrist_rotmat)
            wrist_aa = wrist_aa.flatten(-2)
            self.aug_motion_files[vid_name] = {
                "humanoid_localmat": humanoid_localmat,
                "humanoid_globalmat": humanoid_globalmat,
                "beta": beta,
                "wrist_aa": wrist_aa,
            }
            N += 1
        print(f"Loaded {N} motion files from arctic dataset")
        return

    def _load_grab_dataset(self):
        self.grab_test_sbj = ["s10"]
        dataset_path = "./inputs/grab_neutral"
        # ./inputs/arctic_neutral/s01/xx.pt
        character_obj_dict = {}
        max_length = -1
        all_path = Path(dataset_path).glob("**/*.pt")
        all_path = list(all_path)
        all_path.sort()
        N = 0
        for path in tqdm(all_path):
            if path.parent.name not in character_obj_dict:
                character_obj_dict[path.parent.name] = []
            obj_name = path.name.split("_")[0]
            character_obj_dict[path.parent.name].append(obj_name)

            # if self.split == "train":
            #     if obj_name in self.test_obj:
            #         continue
            # else:
            #     if obj_name not in self.test_obj:
            #         continue

            if self.split == "train":
                if path.parent.name in self.grab_test_sbj:
                    continue
            else:
                if path.parent.name not in self.grab_test_sbj:
                    continue

            vid_name = path.parent.name + "_" + path.name
            # print(f"Loading {vid_name}")
            motion_data = load_arctic_data(path)
            humanoid_localmat, humanoid_globalmat = get_humanoid_data(motion_data)
            beta = motion_data["humanoid"]["betas"]
            smplx_localmat = humanoid_localmat[:, HUMANOID2SMPLX]
            wrist_localmat = smplx_localmat[:, [20, 21]]
            wrist_rotmat = matrix.get_rotation(wrist_localmat)
            wrist_aa = matrix.matrix_to_axis_angle(wrist_rotmat)
            wrist_aa = wrist_aa.flatten(-2)
            self.aug_motion_files[vid_name] = {
                "humanoid_localmat": humanoid_localmat,
                "humanoid_globalmat": humanoid_globalmat,
                "beta": beta,
                "wrist_aa": wrist_aa,
            }
            N += 1

        print(f"Loaded {N} motion files from grab dataset")
        return

    def _process_data(self, data, idx):
        pose = data["pose"]
        length = pose.shape[0]
        if length <= self.max_motion_len:
            start_frame = 0
            end_frame = length
        else:
            start_frame = np.random.randint(0, length - self.max_motion_len + 1)
            end_frame = start_frame + self.max_motion_len

        pose = pose[int(start_frame) : int(end_frame)]
        N = pose.shape[0]
        transl = pose[:, :3]
        global_orient = pose[:, 3:6]
        body_pose = pose[:, 6:]
        skeleton = data["skeleton"]  # (1, 22, 3)

        caption = ""

        beta = data["beta"].repeat(N, 1)  # (N, 10)

        if torch.rand(1) < 0.3:
            # 30% of the time, use aug dataset wrist rot
            select_seq = self.aug_motion_keys[np.random.randint(0, len(self.aug_motion_keys))]
            aug_data = self.aug_motion_files[select_seq]
            wrist_aa = aug_data["wrist_aa"]
            M = wrist_aa.shape[0]
            if M > N:
                # if aug motion is longer than original motion, randomly select a part of it
                start_frame = np.random.randint(0, M - N + 1)
                wrist_aa = wrist_aa[start_frame : start_frame + N]
            elif M == N:
                pass
            else:
                # if aug motion is shorter than original motion, pad it, randomly pre-pad or post-pad
                pre_pad = np.random.randint(0, N - M + 1)
                post_pad = N - M - pre_pad
                if pre_pad == 0:
                    wrist_aa = torch.cat([wrist_aa, wrist_aa[-1:].repeat(post_pad, 1)], dim=0)
                elif post_pad == 0:
                    wrist_aa = torch.cat([wrist_aa[:1].repeat(pre_pad, 1), wrist_aa], dim=0)
                else:
                    wrist_aa = torch.cat(
                        [wrist_aa[:1].repeat(pre_pad, 1), wrist_aa, wrist_aa[-1:].repeat(post_pad, 1)], dim=0
                    )
            body_pose[:, -6:] = wrist_aa

        if N < self.max_motion_len and not self.get_statistics:
            # pad last frame
            transl = torch.cat([transl, transl[-1:].repeat(self.max_motion_len - N, 1)], dim=0)
            global_orient = torch.cat([global_orient, global_orient[-1:].repeat(self.max_motion_len - N, 1)], dim=0)
            body_pose = torch.cat([body_pose, body_pose[-1:].repeat(self.max_motion_len - N, 1)], dim=0)
            beta = torch.cat([beta, beta[-1:].repeat(self.max_motion_len - N, 1)], dim=0)

        return_data = {
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "beta": beta,
            "skeleton": skeleton,
            "caption": caption,
            "length": torch.tensor(N, dtype=torch.int32),
            "meta": {
                "dataset_id": "AMASS",
                "vid": self.idx2meta[idx],
                "idx": idx,
            },
        }

        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data


group_name = "train_datasets/amass"
MainStore.store(name="amass_aug", node=builds(AmassAugDataset, split="train"), group=group_name)
